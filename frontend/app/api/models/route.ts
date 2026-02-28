import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // 从后端API获取模型列表
    const backendUrl = 'http://127.0.0.1:5001/api/models';
    const response = await fetch(backendUrl);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend API error:', response.status, errorText);
      // 如果后端API失败，返回默认模型列表
      const defaultModels = [
        { name: 'default', path: '', description: '默认分类模型', available: true },
        { name: 'augmented_training', path: 'models/augmented_training', description: '增强训练模型', available: false },
        { name: 'arona_plana', path: 'models/arona_plana', description: '阿罗娜普拉娜模型', available: false },
        { name: 'arona_plana_efficientnet', path: 'models/arona_plana_efficientnet', description: 'EfficientNet模型', available: false },
        { name: 'arona_plana_resnet18', path: 'models/arona_plana_resnet18', description: 'ResNet18模型', available: false },
        { name: 'optimized', path: 'models/optimized', description: '优化模型', available: false }
      ];
      return NextResponse.json({ models: defaultModels }, { status: 200 });
    }
    
    const result = await response.json();
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error('Failed to load models:', error);
    // 如果发生错误，返回默认模型列表
    const defaultModels = [
      { name: 'default', path: '', description: '默认分类模型', available: true },
      { name: 'augmented_training', path: 'models/augmented_training', description: '增强训练模型', available: false },
      { name: 'arona_plana', path: 'models/arona_plana', description: '阿罗娜普拉娜模型', available: false },
      { name: 'arona_plana_efficientnet', path: 'models/arona_plana_efficientnet', description: 'EfficientNet模型', available: false },
      { name: 'arona_plana_resnet18', path: 'models/arona_plana_resnet18', description: 'ResNet18模型', available: false },
      { name: 'optimized', path: 'models/optimized', description: '优化模型', available: false }
    ];
    return NextResponse.json({ models: defaultModels }, { status: 200 });
  }
}
