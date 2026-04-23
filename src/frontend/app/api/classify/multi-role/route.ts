import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('前端API路由接收到多角色检测POST请求');
  try {
    console.log('开始处理多角色检测请求...');
    const formData = await request.formData();
    console.log('FormData解析完成');
    const file = formData.get('file') as File;
    const useCoreML = formData.get('use_coreml') as string;
    const useModel = formData.get('use_model') as string;
    const useAttributes = formData.get('use_attributes') as string;
    const modelName = formData.get('model_name') as string;
    const cacheBypass = formData.get('cache_bypass') as string;

    console.log('请求参数:', {
      hasFile: !!file,
      useCoreML: useCoreML,
      useModel: useModel,
      useAttributes: useAttributes,
      modelName: modelName,
      cacheBypass: cacheBypass
    });

    if (!file) {
      console.error('没有提供文件');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    console.log('收到多角色检测请求:', {
      fileName: file.name,
      fileSize: file.size,
      useCoreML: useCoreML,
      useModel: useModel,
      useAttributes: useAttributes,
      modelName: modelName,
      cacheBypass: cacheBypass
    });

    const backendUrl = 'http://127.0.0.1:8000/api/classify/multi-role';
    console.log('准备转发请求到后端API:', backendUrl);
    const backendFormData = new FormData();
    backendFormData.append('file', file);

    if (useCoreML === 'true') {
      backendFormData.append('use_coreml', 'true');
    }

    if (useModel === 'true') {
      backendFormData.append('use_model', 'true');
    }

    if (useAttributes === 'true') {
      backendFormData.append('use_attributes', 'true');
    }

    if (modelName) {
      backendFormData.append('model_name', modelName);
    }

    if (cacheBypass === 'true') {
      backendFormData.append('cache_bypass', 'true');
    }

    console.log('开始发送请求到后端API...');
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: backendFormData,
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
    console.error('多角色检测失败:', error);
    return NextResponse.json({ error: 'Multi-role classification failed' }, { status: 500 });
  }
}
