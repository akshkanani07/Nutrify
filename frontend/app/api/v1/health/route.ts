import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const backendUrl = process.env.BACKEND_API_URL || "http://127.0.0.1:8000";
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${backendUrl}/api/v1/health`, {
      cache: "no-store",
      signal: controller.signal,
    });
    clearTimeout(timeout);
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: unknown) {
    return NextResponse.json(
      {
        status: "starting",
        version: "2.0.0",
        provider: "ml_model",
        provider_ready: false,
        message: "FastAPI backend is offline or starting up.",
      },
      { status: 503 }
    );
  }
}
