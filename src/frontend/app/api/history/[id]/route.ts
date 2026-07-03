import { NextRequest, NextResponse } from 'next/server';

// 修改点 1：将 params 的类型定义修改为 Promise<{ id: string }>
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    // 修改点 2：使用 await 异步解析出 id
    const { id } = await params;
    
    const authHeader = request.headers.get('authorization');
    const headers: HeadersInit = {};
    
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    
    const response = await fetch(`http://127.0.0.1:8001/api/history/${id}`, {
      method: 'DELETE',
      headers: headers,
    });
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('删除历史记录失败:', error);
    return NextResponse.json(
      { success: false, message: '删除历史记录失败' },
      { status: 500 }
    );
  }
}