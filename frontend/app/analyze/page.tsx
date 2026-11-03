"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Sparkles, Shield, Cpu, RefreshCw } from "lucide-react";

import { ImageUploadZone } from "@/components/ImageUploadZone";
import { ProcessingAnimation } from "@/components/ProcessingAnimation";
import { NutritionResultView } from "@/components/NutritionResultView";
import { ErrorAlert } from "@/components/ErrorAlert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  FoodPrediction,
  ErrorEnvelope,
  CandidateFood,
  parseAnalysisResponse,
} from "@/lib/types";

export default function AnalyzePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<ErrorEnvelope | null>(null);
  const [prediction, setPrediction] = useState<FoodPrediction | null>(null);
  const [healthStatus, setHealthStatus] = useState<{
    provider: string;
    ready: boolean;
    version?: string;
  } | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Check backend health on mount and periodically
  useEffect(() => {
    let isMounted = true;
    async function checkHealth() {
      try {
        const res = await fetch("/api/v1/health", { cache: "no-store" }).catch(() => null);
        if (res && res.ok) {
          const data = await res.json();
          if (isMounted) {
            setHealthStatus({
              provider: data.provider || "ml_model",
              ready: Boolean(data.provider_ready),
              version: data.model_version,
            });
          }
        } else {
          if (isMounted) setHealthStatus({ provider: "ml_model", ready: false });
        }
      } catch {
        if (isMounted) setHealthStatus({ provider: "ml_model", ready: false });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Cleanup object URLs on unmount or file replace
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (file: File) => {
    setError(null);
    setPrediction(null);

    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }

    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(url);
  };

  const handleClear = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setPrediction(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/api/analyze-food", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const rawData = await response.json().catch(() => ({}));

      if (!response.ok || rawData.success === false) {
        const errObj = rawData?.error || {};
        setError({
          code: errObj.code || (response.status === 503 ? "BACKEND_UNREACHABLE" : "ANALYSIS_FAILED"),
          message: errObj.message || "Failed to analyze food image with local model.",
          hint: errObj.hint || "Ensure your image clearly displays a single food or meal.",
          retryable: errObj.retryable !== undefined ? errObj.retryable : true,
        });
      } else {
        const parsed = parseAnalysisResponse(rawData);
        setPrediction(parsed);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        setError({
          code: "REQUEST_CANCELLED",
          message: "The analysis request was cancelled.",
          hint: "Click 'Analyze Meal' to restart whenever you are ready.",
          retryable: true,
        });
      } else {
        setError({
          code: "NETWORK_ERROR",
          message:
            err instanceof Error
              ? err.message
              : "Unable to reach the nutrition inference service.",
          hint: "Ensure the local backend server is running on http://127.0.0.1:8000.",
          retryable: true,
        });
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleCandidateSelect = (cand: CandidateFood) => {
    if (!prediction) return;
    // Swap primary prediction with selected candidate
    setPrediction({
      ...prediction,
      food_name: cand.food_name,
      confidence: cand.confidence,
      calories: cand.calories || Math.round(prediction.calories * 0.95),
      protein: cand.protein || prediction.protein,
      carbs: cand.carbs || prediction.carbs,
      fat: cand.fat || prediction.fat,
      fiber: cand.fiber || prediction.fiber,
      status: "ok",
    });
  };

  return (
    <div className="min-h-screen bg-slate-50/70 text-slate-900 pb-20 selection:bg-emerald-100 selection:text-emerald-900">
      {/* ================= STICKY HEADER ================= */}
      <header className="border-b bg-white/90 backdrop-blur sticky top-0 z-30 px-4 sm:px-6 py-3.5 flex items-center justify-between shadow-xs">
        <Link
          href="/"
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Back to Home</span>
        </Link>

        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold text-sm shadow-xs">
            N
          </div>
          <span className="font-bold text-lg tracking-tight">
            Nutrify <span className="text-emerald-600">AI</span>
          </span>
        </div>

        {/* Engine Status Pill */}
        <div className="flex items-center gap-2 text-xs">
          <div
            className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full border font-mono text-[11px] ${
              healthStatus?.ready
                ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                : "bg-amber-50 border-amber-200 text-amber-800"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                healthStatus?.ready ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span>
              {healthStatus?.ready
                ? "FastAPI ML Ready"
                : "Backend Offline (Port 8000)"}
            </span>
          </div>
        </div>
      </header>

      {/* ================= MAIN CONTAINER ================= */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 pt-6 sm:pt-10 space-y-6">
        {/* Page Title & Subtitle */}
        <div className="text-center sm:text-left space-y-1">
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            AI Food Recognition & Nutrition
          </h1>
          <p className="text-sm text-slate-500 max-w-xl">
            Upload or photograph any meal to instantly extract accurate macronutrients, calories, and personalized recommendations with zero cloud APIs.
          </p>
        </div>

        {/* Error Notification Alert */}
        {error && (
          <ErrorAlert
            error={error}
            onRetry={handleAnalyze}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Workflow State Machine */}
        {!prediction ? (
          /* ================= STATE 1 & 2: UPLOAD & PROCESSING ================= */
          <div className="space-y-6">
            {loading ? (
              <ProcessingAnimation
                previewUrl={previewUrl}
                onCancel={handleCancel}
              />
            ) : (
              <ImageUploadZone
                onFileSelect={handleFileSelect}
                onAnalyze={handleAnalyze}
                onClear={handleClear}
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                isLoading={loading}
                onError={(err) =>
                  setError({
                    code: err.code,
                    message: err.message,
                    hint: err.hint,
                    retryable: true,
                  })
                }
              />
            )}

            {/* Feature Highlights Grid */}
            {!loading && !selectedFile && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-2">
                <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-xs flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600 mt-0.5">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-xs text-slate-900">75.6% Neural Top-1</h4>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Fine-tuned EfficientNet-B0 classifier with 98.9% Top-5 coverage.
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-xs flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-blue-50 text-blue-600 mt-0.5">
                    <Cpu className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-xs text-slate-900">80ms Edge Latency</h4>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Blazing fast CPU inference with optimized ONNX / PyTorch kernels.
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-xs flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-purple-50 text-purple-600 mt-0.5">
                    <Shield className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-xs text-slate-900">USDA Ground Truth</h4>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Decoupled reference nutritional database mapped to 1,051 foods.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ================= STATE 3: NUTRITION RESULT SCREEN ================= */
          <NutritionResultView
            prediction={prediction}
            onReset={handleClear}
            onSelectCandidate={handleCandidateSelect}
          />
        )}
      </main>
    </div>
  );
}
