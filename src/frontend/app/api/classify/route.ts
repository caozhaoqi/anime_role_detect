import { NextRequest, NextResponse } from 'next/server';
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  console.log('前端API路由接收到POST请求');
  try {
    console.log('开始处理请求...');
    const formData = await request.formData();
    console.log('FormData解析完成');
    const file = formData.get('file') as File;

    console.log('请求参数:', {
      hasFile: !!file
    });

    if (!file) {
      console.error('没有提供文件');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    console.log('收到分类请求:', {
      fileName: file.name,
      fileSize: file.size
    });

    const backendUrl = 'http://127.0.0.1:8080/api/classify';
    console.log('准备转发请求到后端API:', backendUrl);

    const backendFormData = new FormData();
    backendFormData.append('file', file);
    backendFormData.append('model_name', formData.get('model_name') as string || 'efficientnet_b3_loli_optimized_v2_20260529_133654');
    backendFormData.append('use_coreml', formData.get('use_coreml') as string || 'false');
    backendFormData.append('use_model', formData.get('use_model') as string || 'true');
    backendFormData.append('use_attributes', formData.get('use_attributes') as string || 'true');
    backendFormData.append('cache_bypass', 'false');
    backendFormData.append('multi_role', formData.get('multi_role') as string || 'false');
    backendFormData.append('use_deepdanbooru', formData.get('use_deepdanbooru') as string || 'true');

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
        
        // 处理认证失败
        if (response.status === 401) {
          let errorMessage = '认证失败，请登录';
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.detail || errorData.error || errorData.message || '认证失败，请登录';
          } catch {
            errorMessage = errorText || '认证失败，请登录';
          }
          return NextResponse.json({ 
            error: errorMessage,
            code: 'UNAUTHORIZED',
            message: '请登录后再试'
          }, { status: 401 });
        }
        
        return NextResponse.json({ error: 'Backend API error' }, { status: response.status });
      }

      const result = await response.json();
      console.log('后端API返回结果:', result);

      // 后端返回的标准信封结构即为 { success, data, message }，
      // 直接透传，避免再包一层 data 导致前端取不到 success 字段
      return NextResponse.json(
        { success: result.success, data: result.data, message: result.message },
        { status: 200 }
      );
    } catch (fetchError) {
      console.error('发送请求到后端API失败:', fetchError);
      return NextResponse.json({ error: 'Failed to connect to backend API' }, { status: 500 });
    }
  } catch (error) {
    console.error('分类失败:', error);
    return NextResponse.json({ error: 'Classification failed' }, { status: 500 });
  }
}
