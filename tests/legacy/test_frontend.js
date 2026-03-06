// 测试脚本：模拟分类请求
const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');

async function testClassification() {
  try {
    const formData = new FormData();
    // 使用一个测试图片文件
    const testImagePath = './data/test_images/test1.jpg';
    if (fs.existsSync(testImagePath)) {
      formData.append('file', fs.createReadStream(testImagePath));
    } else {
      console.error('测试图片文件不存在');
      return;
    }

    console.log('发送分类请求...');
    const response = await fetch('/api/classify', {
      method: 'POST',
      body: formData,
    });

    console.log('响应状态:', response.status);
    console.log('响应头:', Object.fromEntries(response.headers.entries()));
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('响应错误:', errorData);
      return;
    }

    const data = await response.json();
    console.log('分类结果数据:', JSON.stringify(data, null, 2));
    
  } catch (err) {
    console.error('测试错误:', err);
  }
}

testClassification();
