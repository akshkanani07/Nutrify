import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const primaryUrl = process.env.BACKEND_API_URL || "http://127.0.0.1:8001";
  const candidateUrls = [primaryUrl];
  if (primaryUrl.includes(":8001")) {
    candidateUrls.push("http://127.0.0.1:8000");
  } else if (primaryUrl.includes(":8000")) {
    candidateUrls.push("http://127.0.0.1:8001");
  }

  for (const url of candidateUrls) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${url}/api/v1/health`, {
        cache: "no-store",
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (res.ok) {
        const data = await res.json();
        return NextResponse.json(data, { status: res.status });
      }
    } catch {
      // Continue to next candidate port
    }
  }

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

