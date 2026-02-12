import { NextRequest, NextResponse } from 'next/server';

// 模拟模型数据
const mockModels = [
  { name: 'default', path: '', files: [] },
  { name: 'augmented_training', path: 'models/augmented_training', files: ['model.pth'] },
  { name: 'efficientnet_b0', path: 'models/efficientnet_b0', files: ['model.pth'] },
  { name: 'mobilenet_v2', path: 'models/mobilenet_v2', files: ['model.pth'] },
];

export async function GET(request: NextRequest) {
  try {
    // 在实际应用中，这里应该从后端API获取模型列表
    // 为了演示，我们返回模拟数据
    return NextResponse.json({ models: mockModels }, { status: 200 });
  } catch (error) {
    console.error('Failed to load models:', error);
    return NextResponse.json({ error: 'Failed to load models' }, { status: 500 });
  }
}
