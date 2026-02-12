import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as string;
    const model = formData.get('model') as string;

    // 在实际应用中，这里应该将请求转发到后端API
    // 为了演示，我们返回模拟数据
    const mockResponse = {
      role: '初音未来',
      similarity: 0.92,
      confidence: 'high',
    };

    return NextResponse.json(mockResponse, { status: 200 });
  } catch (error) {
    console.error('Classification failed:', error);
    return NextResponse.json({ error: 'Classification failed' }, { status: 500 });
  }
}
