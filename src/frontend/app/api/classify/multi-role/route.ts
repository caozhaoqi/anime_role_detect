import { NextRequest, NextResponse } from 'next/server';
export const dynamic = "force-dynamic";

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
    const debug = formData.get('debug') === 'true';

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

    const backendUrl = 'http://127.0.0.1:8080/api/classify/multi-role';
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

    if (debug) {
      backendFormData.append('debug', 'true');
    }

    console.log('开始发送请求到后端API...');
    try {
      // 提取并转发Authorization头
      const authHeader = request.headers.get('authorization');
      console.log('接收到的Authorization头:', authHeader);
      
      const headers: HeadersInit = {};
      if (authHeader) {
        headers['Authorization'] = authHeader;
        console.log('转发Authorization头到后端');
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
        return NextResponse.json({ error: 'Backend API error' }, { status: response.status });
      }

      const result = await response.json();
      console.log('后端API返回结果:', result);

      // 后端返回标准信封 {success, data, message}（经 api-gateway 映射 model-service 的 /api/model/detect-multiple），
      // 直接透传，避免外层丢失 success 字段导致前端误判失败（与已修的 /api/classify 路由一致）
      return NextResponse.json(
        { success: result.success, data: result.data, message: result.message },
        { status: 200 }
      );
    } catch (fetchError) {
      console.error('发送请求到后端API失败:', fetchError);
      return NextResponse.json({ error: 'Failed to connect to backend API' }, { status: 500 });
    }
  } catch (error) {
    console.error('多角色检测失败:', error);
    return NextResponse.json({ error: 'Multi-role classification failed' }, { status: 500 });
  }
}
