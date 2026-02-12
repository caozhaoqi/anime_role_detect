import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const useModel = formData.get('use_model') as string;
    const useDeepdanbooru = formData.get('use_deepdanbooru') as string;

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    console.log('收到分类请求:', {
      fileName: file.name,
      fileSize: file.size,
      useModel: useModel,
      useDeepdanbooru: useDeepdanbooru
    });

    const backendUrl = 'http://127.0.0.1:5001/api/classify';
    const backendFormData = new FormData();
    backendFormData.append('file', file);

    if (useModel === 'true') {
      backendFormData.append('use_model', 'true');
    }

    if (useDeepdanbooru === 'true') {
      backendFormData.append('use_deepdanbooru', 'true');
    }

    console.log('转发请求到后端API:', backendUrl);

    const response = await fetch(backendUrl, {
      method: 'POST',
      body: backendFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('后端API返回错误:', response.status, errorText);
      return NextResponse.json({ error: 'Backend API error' }, { status: response.status });
    }

    const result = await response.json();
    console.log('后端API返回结果:', result);

    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error('分类失败:', error);
    return NextResponse.json({ error: 'Classification failed' }, { status: 500 });
  }
}
