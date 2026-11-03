import { NextRequest, NextResponse } from "next/server";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

export async function POST(request: NextRequest): Promise<NextResponse> {
  const backendUrl = process.env.BACKEND_API_URL || DEFAULT_BACKEND_URL;
  try {
    const body = await request.json();
    const res = await fetch(`${backendUrl}/api/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: any) {
    return NextResponse.json(
      { success: false, message: `Feedback proxy error: ${err.message}` },
      { status: 500 }
    );
  }
}
