import { NextRequest, NextResponse } from 'next/server';
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  console.log('前端API路由接收到视频标注识别POST请求');
  try {
    // 从 URL query 透传视频识别参数（VideoPanel 以 query 传参）
    const searchParams = request.nextUrl.searchParams;
    const qs = new URLSearchParams();
    for (const key of ['frame_interval', 'confidence_threshold', 'recognition_mode', 'model_name', 'top_k']) {
      const val = searchParams.get(key);
      if (val) {
        qs.set(key, val);
      }
    }

    const qsStr = qs.toString();
    // 目标走 api-gateway，由网关再转发到 multimedia-service:8002 的
    // /video/recognize-with-overlay（与 /api/video/recognize 同源入口，便于统一鉴权/限流）。
    const backendUrl = `http://127.0.0.1:8080/api/video/recognize-with-overlay${qsStr ? `?${qsStr}` : ''}`;
    console.log('准备转发请求到后端multimedia服务:', backendUrl);

    const authHeader = request.headers.get('authorization');
    const headers: HeadersInit = {
      // 透传原始 Content-Type（含 multipart 边界），保证文件部件完整
      'content-type': request.headers.get('content-type') || 'multipart/form-data',
    };
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    // 修复（2026-08-10）：overlay 标注模式此前无显式路由，只能走 next.config.js
    // 的 rewrite 代理；Next.js 的 rewrite 代理默认请求体上限 10MB，视频 >10MB 时
    // 直接 "Request body exceeded 10MB" + ECONNRESET（见 logs/services/frontend/frontend.err.log）。
    // 这里改为显式路由 + 直接转发原始请求体字节流，绕过该代理上限，与非标注模式一致。
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers,
      body: request.body,
      // Node fetch 流式转发必需
      duplex: 'half',
    } as RequestInit);

    console.log('后端API响应状态:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('后端API返回错误:', response.status, errorText);
      return NextResponse.json({ error: 'Backend API error', detail: errorText }, { status: response.status });
    }

    const result = await response.json();
    console.log('后端API返回结果:', result);
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error('视频标注识别失败:', error);
    return NextResponse.json({ error: 'Video overlay recognition failed' }, { status: 500 });
  }
}
