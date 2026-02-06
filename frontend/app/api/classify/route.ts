import { NextRequest, NextResponse } from 'next/server';

// API路由：处理文件分类请求
export async function GET() {
  // 返回API文档，与Flask相同
  const apiDoc = {
    endpoint: "/api/classify",
    method: "POST",
    description: "角色分类API（支持图片和视频）",
    parameters: {
      file: "媒体文件（必填，支持图片和视频）",
      use_model: "是否使用专用模型 (true/false, 默认false)",
      frame_skip: "视频帧跳过间隔 (默认5)"
    },
    response: {
      filename: "文件名",
      role: "识别的角色",
      similarity: "相似度",
      boxes: "边界框信息",
      fileType: "文件类型 (image/video)",
      videoResults: "视频帧检测结果（仅视频文件）"
    }
  };
  return NextResponse.json(apiDoc);
}

export async function POST(request: NextRequest) {
  try {
    // 解析表单数据
    const formData = await request.formData();
    
    // 检查是否有文件
    const file = formData.get('file') as File;
    if (!file) {
      return NextResponse.json({ error: '没有文件部分' }, { status: 400 });
    }
    
    // 准备转发到Flask的FormData
    const flaskFormData = new FormData();
    flaskFormData.append('file', file);
    flaskFormData.append('use_model', formData.get('use_model') as string || 'false');
    flaskFormData.append('frame_skip', formData.get('frame_skip') as string || '5');
    
    // 转发请求到Flask后端
    const flaskResponse = await fetch('http://127.0.0.1:5001/api/classify', {
      method: 'POST',
      body: flaskFormData
    });
    
    // 检查Flask响应状态
    if (!flaskResponse.ok) {
      const errorData = await flaskResponse.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.error || '后端服务失败' },
        { status: flaskResponse.status }
      );
    }
    
    // 解析Flask响应数据
    const result = await flaskResponse.json();
    
    // 返回结果
    return NextResponse.json(result);
  } catch (error) {
    console.error('API处理失败:', error);
    return NextResponse.json(
      { error: '处理失败: ' + (error as Error).message },
      { status: 500 }
    );
  }
}
