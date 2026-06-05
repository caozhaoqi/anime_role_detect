import { NextRequest, NextResponse } from 'next/server';

// 历史记录API路由
// 用于处理历史记录相关的请求，转发到后端API

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    const headers: HeadersInit = {};
    
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    
    const response = await fetch('http://127.0.0.1:8001/api/history', {
      method: 'GET',
      headers: headers,
    });
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('获取历史记录失败:', error);
    return NextResponse.json(
      { success: false, message: '获取历史记录失败' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { id } = params;
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
