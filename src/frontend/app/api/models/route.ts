import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const backendUrl = 'http://127.0.0.1:8001/api/models';
    console.log('Fetching models from backend:', backendUrl);
    
    // Authorization
    const authHeader = request.headers.get('authorization');
    console.log('Authorization:', authHeader);
    
    const headers: HeadersInit = {};
    if (authHeader) {
      headers['Authorization'] = authHeader;
      console.log('Authorization');
    }
    
    const response = await fetch(backendUrl, {
      headers: headers
    });

    console.log('Backend response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend API error:', response.status, errorText);
      return NextResponse.json({ success: false, models: [], default_model: "default" }, { status: 200 });
    }

    const result = await response.json();
    console.log('Backend response:', result);

    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error('Failed to load models:', error);
    return NextResponse.json({ success: false, models: [], default_model: "default" }, { status: 200 });
  }
}