import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('YOLO多目标检测API路由接收到POST请求');
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const yoloModel = formData.get('yolo_model') as string || 'yolov8n.pt';
    const personConfThreshold = parseFloat(formData.get('person_conf_threshold') as string) || 0.5;
    const maxDetections = parseInt(formData.get('max_detections') as string) || 10;

    if (!file) {
      console.error('没有提供文件');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    console.log('YOLO检测请求:', {
      fileName: file.name,
      fileSize: file.size,
      yoloModel,
      personConfThreshold,
      maxDetections,
    });

    const backendUrl = 'http://127.0.0.1:8001/api/model/detect-yolo';
    console.log('转发请求到后端API:', backendUrl);

    const backendFormData = new FormData();
    backendFormData.append('file', file);
    backendFormData.append('yolo_model', yoloModel);
    backendFormData.append('person_conf_threshold', personConfThreshold.toString());
    backendFormData.append('max_detections', maxDetections.toString());

    const authHeader = request.headers.get('authorization');
    const headers: HeadersInit = {};
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    const response = await fetch(backendUrl, {
      method: 'POST',
      body: backendFormData,
      headers: headers,
    });

    console.log('后端API响应状态:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('后端API返回错误:', response.status, errorText);
      return NextResponse.json(
        { error: 'YOLO detection failed', details: errorText },
        { status: response.status }
      );
    }

    const result = await response.json();
    console.log('YOLO检测结果:', result);

    return NextResponse.json({
      data: result,
      success: true,
    }, { status: 200 });

  } catch (error) {
    console.error('YOLO多目标检测失败:', error);
    return NextResponse.json(
      { error: 'YOLO multi-target detection failed', details: String(error) },
      { status: 500 }
    );
  }
}
