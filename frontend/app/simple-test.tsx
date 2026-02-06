import React, { useState } from 'react';

const SimpleTestPage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResult(null);
      console.log('📁 文件已选择:', file.name);
    }
  };

  const handleUpload = async () => {
    console.log('🔄 handleUpload函数被调用！');
    alert('handleUpload函数被调用！');

    if (!selectedFile) {
      setError('请先选择文件');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      console.log('🌐 开始发送请求...');
      const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData,
      });

      console.log('📡 响应状态:', response.status);

      if (!response.ok) {
        throw new Error(`服务器响应错误: ${response.status}`);
      }

      const data = await response.json();
      console.log('📡 响应数据:', data);
      setResult(data);
      alert('上传成功！');
    } catch (error) {
      console.error('❌ 上传失败:', error);
      setError(`上传失败: ${(error as Error).message}`);
      alert(`上传失败: ${(error as Error).message}`);
    } finally {
      setIsLoading(false);
      console.log('🔚 上传完成');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 w-full max-w-md">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">简单测试页面</h1>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <span className="text-red-700">{error}</span>
          </div>
        )}

        {result && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
            <h2 className="font-medium text-green-900 mb-2">上传成功！</h2>
            <pre className="text-sm text-green-800">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}

        <div className="mb-6">
          <label htmlFor="file-input" className="block text-sm font-medium text-gray-700 mb-2">
            选择文件：
          </label>
          <input
            id="file-input"
            type="file"
            accept="image/*, video/*"
            onChange={handleFileSelect}
            className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none"
          />
          {selectedFile && (
            <p className="mt-2 text-sm text-gray-500">
              已选择：{selectedFile.name}
            </p>
          )}
        </div>

        <button
          onClick={handleUpload}
          disabled={isLoading || !selectedFile}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? '上传中...' : '上传'}
        </button>
      </div>
    </div>
  );
};

export default SimpleTestPage;
