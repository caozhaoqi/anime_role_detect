import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('前端API路由接收到视频识别POST请求');
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const frameInterval = formData.get('frame_interval') as string;
    const confidenceThreshold = formData.get('confidence_threshold') as string;

    console.log('请求参数:', {
      hasFile: !!file,
      frameInterval: frameInterval,
      confidenceThreshold: confidenceThreshold
    });

    if (!file) {
      console.error('没有提供文件');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const backendUrl = 'http://127.0.0.1:8001/api/video/recognize';
    console.log('准备转发请求到后端API:', backendUrl);

    const backendFormData = new FormData();
    backendFormData.append('file', file);
    if (frameInterval) {
      backendFormData.append('frame_interval', frameInterval);
    }
    if (confidenceThreshold) {
      backendFormData.append('confidence_threshold', confidenceThreshold);
    }

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