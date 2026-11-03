"""Nutrify AI — FastAPI application."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.analytics import analytics_tracker
from app.exceptions import ConfigurationError, ModelNotReadyError
from app.feedback import feedback_manager
from app.image_utils import validate_and_preprocess
from app.providers.ml_provider import MLModelProvider
from app.providers.mock_provider import MockProvider
from app.rate_limit import RateLimiter
from app.schemas import (
    AnalysisResponse,
    ErrorDetail,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
)

logger = logging.getLogger("nutrify.proxy")

ERROR_CODES = {
    "MISSING_IMAGE": 400,
    "INVALID_IMAGE_TYPE": 400,
    "INVALID_IMAGE": 400,
    "IMAGE_TOO_LARGE": 413,
    "RATE_LIMITED": 429,
    "NO_FOOD_DETECTED": 422,
    "MODEL_NOT_READY": 503,
    "PROVIDER_UNAVAILABLE": 502,
    "ANALYSIS_TIMEOUT": 504,
    "INTERNAL_ERROR": 500,
}


def _request_id() -> str:
    return str(uuid.uuid4())


def _error_response(
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    hint: str | None = None,
    retryable: bool = False,
) -> JSONResponse:
    body = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(code=code, message=message, hint=hint, retryable=retryable),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_client_ip(request: Request) -> str:
    """Extract client IP, honouring X-Forwarded-For from trusted proxies.

    Walks X-Forwarded-For from right to left and returns the first entry
    that is not in TRUSTED_PROXIES. The header is only evaluated when
    the direct connecting socket host is in TRUSTED_PROXIES.
    """
    client_host = request.client.host if (request.client and request.client.host) else "unknown"

    if client_host not in config.TRUSTED_PROXIES:
        return client_host

    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return client_host

    # Walk right-to-left
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    for hop in reversed(hops):
        if hop not in config.TRUSTED_PROXIES:
            return hop

    return client_host


def _resolve_provider(application: FastAPI) -> MockProvider | MLModelProvider:
    provider = getattr(application.state, "provider", None)
    if provider is not None:
        return provider

    if config.AI_PROVIDER == "mock":
        provider = MockProvider()
    elif config.AI_PROVIDER == "ml_model":
        provider = MLModelProvider()
    else:
        raise RuntimeError(f"Unknown AI_PROVIDER: {config.AI_PROVIDER}")

    application.state.provider = provider
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context: preloads the ML provider at startup for zero cold-start."""
    try:
        provider = _resolve_provider(app)
        logger.info(f"Nutrify AI lifespan: Provider '{provider.name}' preloaded successfully (ready={provider.ready}).")
    except Exception as exc:
        logger.warning(f"Nutrify AI lifespan: Provider pre-warming deferred: {exc}")
    yield


class SecurityAndTracingMiddleware:
    """Pure ASGI middleware for security headers, request tracing, and latency monitoring.
    
    Zero body-stream interception, fully compatible with multipart uploads and TestClient.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"x-request-id":
                rid = value.decode("latin1")
                break
        if not rid:
            rid = str(uuid.uuid4())

        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = rid

        start_time = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode("latin1")))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                if config.APP_ENV == "production":
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                headers.append((b"x-process-time", f"{elapsed_ms:.2f}ms".encode("latin1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app() -> FastAPI:
    """Application factory: validates configuration, instantiates FastAPI, and owns RateLimiter."""
    # Single source of truth for production configuration validation
    if config.APP_ENV == "production" and config.AI_PROVIDER == "mock" and not config.ALLOW_MOCK:
        raise ConfigurationError(
            "Mock provider is refused in production unless ALLOW_MOCK=true is explicitly set."
        )

    if config.APP_ENV == "production":
        trust_edge = os.getenv("TRUST_EDGE_PROXY", "false").lower() in ("true", "1", "yes")
        if not trust_edge:
            logger.warning(
                "TRUST_EDGE_PROXY is not enabled in production: all clients behind an edge proxy will share one rate-limit bucket."
            )

    application = FastAPI(
        title="Nutrify AI",
        version=config.APP_VERSION,
        lifespan=lifespan,
    )

    # Pure ASGI Security & Tracing Middleware
    application.add_middleware(SecurityAndTracingMiddleware)

    # Production CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    application.state.rate_limiter = RateLimiter()
    application.state.provider = None

    @application.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        provider = _resolve_provider(application)
        return HealthResponse(
            status="ok",
            version=config.APP_VERSION,
            provider=provider.name,
            provider_ready=provider.ready,
            model_version=getattr(provider, "model_version", None),
        )

    @application.get("/api/v1/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        """Production monitoring endpoint reporting real-time inference analytics."""
        return MetricsResponse(**analytics_tracker.get_metrics())

    @application.post("/api/v1/feedback", response_model=FeedbackResponse)
    async def submit_feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
        """User feedback loop for recording user corrections for model retraining."""
        client_ip = _get_client_ip(request)
        res = feedback_manager.record_feedback(
            request_id=payload.request_id,
            predicted_food=payload.food_name,
            is_correct=payload.is_correct,
            corrected_food=payload.selected_food,
            comments=payload.comments,
            client_ip=client_ip,
        )
        return FeedbackResponse(success=res["success"], message=res["message"])

    @application.post("/api/v1/analyze-food")
    async def analyze_food(
        request: Request,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> JSONResponse:
        t0 = time.perf_counter()
        rid = getattr(request.state, "request_id", _request_id())

        # --- rate limit ---
        rate_limiter: RateLimiter = request.app.state.rate_limiter
        client_ip = _get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            analytics_tracker.log_failure(rid, client_ip, "RATE_LIMITED")
            return _error_response(
                rid,
                429,
                "RATE_LIMITED",
                "Too many requests. Please try again later.",
                hint="Wait 60 seconds before retrying.",
                retryable=True,
            )

        # --- provider check ---
        provider = _resolve_provider(request.app)

        # --- image present? ---
        if image is None or image.filename == "":
            analytics_tracker.log_failure(rid, client_ip, "MISSING_IMAGE")
            return _error_response(
                rid,
                400,
                "MISSING_IMAGE",
                "No image was uploaded.",
                hint="Attach an image file to the 'image' form field.",
            )

        # --- read and validate ---
        try:
            data = await image.read()
        except Exception:
            analytics_tracker.log_failure(rid, client_ip, "INVALID_IMAGE")
            return _error_response(
                rid,
                400,
                "INVALID_IMAGE",
                "Failed to read the uploaded file.",
            )

        try:
            img, _mime = validate_and_preprocess(
                data,
                max_size_bytes=config.MAX_IMAGE_SIZE_BYTES,
                allowed_mime_types=config.ALLOWED_MIME_TYPES,
                min_short_side=config.MIN_SHORT_SIDE_PX,
                max_long_side=config.MAX_LONG_SIDE_PX,
            )
        except ValueError as exc:
            args = exc.args
            code = args[0] if len(args) > 0 else "INVALID_IMAGE"
            msg = args[1] if len(args) > 1 else str(exc)
            hint = args[2] if len(args) > 2 else None
            status = ERROR_CODES.get(code, 400)
            analytics_tracker.log_failure(rid, client_ip, code)
            return _error_response(rid, status, code, msg, hint)

        # --- run provider ---
        try:
            result = provider.analyze(img)
        except ModelNotReadyError as exc:
            analytics_tracker.log_failure(rid, client_ip, "MODEL_NOT_READY")
            return _error_response(
                rid,
                503,
                "MODEL_NOT_READY",
                exc.message,
                hint="Ensure classifier.pt and label_map.json are in ml/models/.",
                retryable=True,
            )
        except Exception as exc:
            analytics_tracker.log_failure(rid, client_ip, "INTERNAL_ERROR")
            return _error_response(rid, 500, "INTERNAL_ERROR", f"Unexpected error: {exc}")

        # --- build response ---
        items_list = result.get("items", [])
        totals_dict = result.get("totals") or {}
        first_item = items_list[0] if items_list else {}

        resp = AnalysisResponse(
            request_id=rid,
            provider=provider.name,
            model_version=getattr(provider, "model_version", None),
            status=result.get("status", "ok"),
            items=items_list,
            totals=result.get("totals"),
            score=result.get("score"),
            advice=result.get("advice", []),
            food_name=first_item.get("name") if isinstance(first_item, dict) else getattr(first_item, "name", None),
            confidence=first_item.get("confidence") if isinstance(first_item, dict) else getattr(first_item, "confidence", None),
            calories=totals_dict.get("calories") if isinstance(totals_dict, dict) else getattr(totals_dict, "calories", None),
            protein=totals_dict.get("protein_g") if isinstance(totals_dict, dict) else getattr(totals_dict, "protein_g", None),
            carbs=totals_dict.get("carbs_g") if isinstance(totals_dict, dict) else getattr(totals_dict, "carbs_g", None),
            fat=totals_dict.get("fat_g") if isinstance(totals_dict, dict) else getattr(totals_dict, "fat_g", None),
            fiber=totals_dict.get("fiber_g") if isinstance(totals_dict, dict) else getattr(totals_dict, "fiber_g", None),
            confidence_tier=result.get("confidence_tier"),
            user_message=result.get("user_message"),
            candidates=result.get("candidates", []),
            serving_weight_g=result.get("serving_weight_g"),
            reference_serving_g=result.get("reference_serving_g"),
            nutrition_per_100g=result.get("nutrition_per_100g"),
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        conf_val = resp.confidence if resp.confidence is not None else 0.0
        analytics_tracker.log_prediction(
            request_id=rid,
            client_ip=client_ip,
            food_name=resp.food_name or "Unknown",
            confidence=conf_val,
            confidence_tier=resp.confidence_tier or "low",
            status=resp.status,
            latency_ms=latency_ms,
            is_confident=(resp.status == "ok"),
        )

        return JSONResponse(status_code=200, content=resp.model_dump())

    return application


app = create_app()
