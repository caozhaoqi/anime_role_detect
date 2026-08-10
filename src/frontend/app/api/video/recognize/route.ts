import { NextRequest, NextResponse } from 'next/server';
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  console.log('前端API路由接收到视频识别POST请求');
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    // 修复（2026-08-10，视频识别 0 结果根因）：
    // VideoPanel 以 URL query 传参（recognition_mode/model_name/frame_interval/...），
    // 后端 multimedia /video/recognize 也用 Query 参数接收。
    // 原实现从 formData 读参数（恒 undefined）且用 formData 转发（后端 Query 读不到）
    // → recognition_mode 恒默认 search → 无 CLIP 索引 → 恒 0 结果。
    // 现在：从 URL query 读取，并透传到后端 URL query。
    const searchParams = request.nextUrl.searchParams;
    const qs = new URLSearchParams();
    for (const key of ['frame_interval', 'confidence_threshold', 'recognition_mode', 'model_name', 'top_k']) {
      const val = searchParams.get(key);
      if (val) {
        qs.set(key, val);
      }
    }

    console.log('请求参数:', {
      hasFile: !!file,
      recognition_mode: qs.get('recognition_mode'),
      frame_interval: qs.get('frame_interval'),
      confidence_threshold: qs.get('confidence_threshold'),
      model_name: qs.get('model_name'),
    });

    if (!file) {
      console.error('没有提供文件');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const qsStr = qs.toString();
    const backendUrl = `http://127.0.0.1:8080/api/video/recognize${qsStr ? `?${qsStr}` : ''}`;
    console.log('准备转发请求到后端API:', backendUrl);

    const backendFormData = new FormData();
    backendFormData.append('file', file);

    const authHeader = request.headers.get('authorization');
    const headers: HeadersInit = {};
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    console.log('开始发送请求到后端API...');
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: backendFormData,
        headers: headers,
      });
      console.log('后端API响应状态:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('后端API返回错误:', response.status, errorText);
        return NextResponse.json({ error: 'Backend API error' }, { status: response.status });
      }

      const result = await response.json();
      console.log('后端API返回结果:', result);

      return NextResponse.json(result, { status: 200 });
    } catch (fetchError) {
      console.error('发送请求到后端API失败:', fetchError);
      return NextResponse.json({ error: 'Failed to connect to backend API' }, { status: 500 });
    }
  } catch (error) {
    console.error('视频识别失败:', error);
    return NextResponse.json({ error: 'Video recognition failed' }, { status: 500 });
  }
}
