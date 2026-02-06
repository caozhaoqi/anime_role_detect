'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Image as ImageIcon,
  Loader2,
  CheckCircle,
  XCircle,
  ArrowLeft,
  Info,
  ChevronDown,
  Sparkles,
  Search,
  Star,
  Award,
  Zap,
  BarChart2,
  RefreshCw,
  Download,
  Trash2,
  X,
  Clock,
} from 'lucide-react';

interface ClassificationResult {
  filename: string;
  role: string;
  similarity: number;
  boxes: any[];
}

interface HistoryItem extends ClassificationResult {
  timestamp: number;
  imageData?: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 保存结果到本地存储
  const saveToHistory = (result: ClassificationResult) => {
    if (typeof window === 'undefined') return;
    
    const historyItem: HistoryItem = {
      ...result,
      timestamp: Date.now(),
      imageData: previewImage
    };

    try {
      const storedData = localStorage.getItem('classificationHistory');
      const existingHistory = storedData ? JSON.parse(storedData) : [];
      const updatedHistory = [historyItem, ...existingHistory].slice(0, 50); // 保留最近50条
      
      localStorage.setItem('classificationHistory', JSON.stringify(updatedHistory));
      setHistory(updatedHistory);
    } catch (error) {
      console.error('保存历史记录失败:', error);
      // 出错时使用空数组
      const updatedHistory = [historyItem];
      localStorage.setItem('classificationHistory', JSON.stringify(updatedHistory));
      setHistory(updatedHistory);
    }
  };

  // 加载历史记录
  const loadHistory = () => {
    if (typeof window === 'undefined') return;
    
    try {
      const storedData = localStorage.getItem('classificationHistory');
      const existingHistory = storedData ? JSON.parse(storedData) : [];
      setHistory(existingHistory);
    } catch (error) {
      console.error('加载历史记录失败:', error);
      // 出错时使用空数组
      setHistory([]);
    }
  };

  // 清空历史记录
  const clearHistory = () => {
    if (typeof window === 'undefined') return;
    
    localStorage.removeItem('classificationHistory');
    setHistory([]);
  };

  // 导出历史记录
  const exportHistory = () => {
    if (typeof window === 'undefined') return;
    
    try {
      const storedData = localStorage.getItem('classificationHistory');
      const historyData = storedData ? JSON.parse(storedData) : [];
      const dataStr = JSON.stringify(historyData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `classification-history-${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('导出历史记录失败:', error);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('请选择图片文件');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    // 创建预览
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleUpload = async () => {
    console.log('handleUpload函数被调用');
    if (!selectedFile) {
      setError('请先选择图片');
      console.log('没有选择文件');
      return;
    }

    console.log('开始分类流程，文件:', selectedFile.name);
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      console.log('开始分类请求...');
      // 使用fetch API发送请求
      const response = await fetch('http://127.0.0.1:5001/api/classify', {
        method: 'POST',
        body: formData,
        // 注意：不要设置Content-Type，让浏览器自动处理
      });

      console.log('响应状态:', response.status);
      
      // 检查响应是否成功
      if (!response.ok) {
        throw new Error(`服务器响应错误: ${response.status}`);
      }

      // 解析响应数据
      const data = await response.json();
      console.log('分类结果数据:', data);
      
      // 确保数据结构正确
      const classificationResult: ClassificationResult = {
        filename: data.filename || selectedFile.name,
        role: data.role || '未知',
        similarity: data.similarity || 0,
        boxes: data.boxes || []
      };
      
      console.log('处理后的分类结果:', classificationResult);
      
      // 更新result状态
      console.log('更新result状态前:', result);
      setResult(classificationResult);
      console.log('更新result状态后:', classificationResult);
      
      // 保存到历史记录
      saveToHistory(classificationResult);
      console.log('分类结果已保存');
      
    } catch (err) {
      console.error('分类错误:', err);
      setError(`分类失败: ${err.message}`);
    } finally {
      setIsLoading(false);
      console.log('分类请求完成');
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreviewImage(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getAccuracyBadgeClass = (similarity: number) => {
    if (similarity >= 0.8) return 'bg-green-100 text-green-800';
    if (similarity >= 0.5) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getAccuracyText = (similarity: number) => {
    if (similarity >= 0.8) return '高';
    if (similarity >= 0.5) return '中';
    return '低';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      {/* 导航栏 */}
      <nav className="bg-white shadow-lg bg-opacity-95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <motion.div 
              className="flex items-center"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <motion.div
                whileHover={{ rotate: 10, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 300 }}
                className="bg-gradient-to-r from-primary to-secondary p-2 rounded-full"
              >
                <ImageIcon className="h-6 w-6 text-white" />
              </motion.div>
              <span className="ml-3 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600">角色分类系统</span>
            </motion.div>
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "#45a049" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-primary text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                <span className="font-medium">AI 分类</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "#f3f4f6" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-gray-100 text-gray-800 rounded-lg flex items-center shadow-sm hover:shadow-md transition-all"
                onClick={() => {
                  setShowHistory(true);
                  loadHistory();
                }}
              >
                <BarChart2 className="h-4 w-4 mr-2" />
                <span className="font-medium">历史记录</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "#f3f4f6" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-gray-100 text-gray-800 rounded-lg flex items-center shadow-sm hover:shadow-md transition-all"
              >
                <Info className="h-4 w-4 mr-2" />
                <span className="font-medium">关于</span>
              </motion.button>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* 主内容 */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* 标题 */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="text-center mb-16"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary to-secondary rounded-full mb-6 shadow-lg"
          >
            <Sparkles className="h-8 w-8 text-white" />
          </motion.div>
          <motion.h1 
            className="text-5xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-700"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            角色分类系统
          </motion.h1>
          <motion.p 
            className="text-xl text-gray-600 max-w-2xl mx-auto"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            使用先进的 AI 技术，一键识别图片中的游戏角色
          </motion.p>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.8 }}
            className="mt-8 flex flex-wrap justify-center gap-4"
          >
            <div className="flex items-center bg-white px-4 py-2 rounded-full shadow-sm">
              <Star className="h-4 w-4 text-yellow-400 mr-2" />
              <span className="text-sm font-medium text-gray-700">60+ 角色支持</span>
            </div>
            <div className="flex items-center bg-white px-4 py-2 rounded-full shadow-sm">
              <Zap className="h-4 w-4 text-blue-400 mr-2" />
              <span className="text-sm font-medium text-gray-700">实时识别</span>
            </div>
            <div className="flex items-center bg-white px-4 py-2 rounded-full shadow-sm">
              <Award className="h-4 w-4 text-purple-400 mr-2" />
              <span className="text-sm font-medium text-gray-700">高准确率</span>
            </div>
          </motion.div>
        </motion.div>

        {/* 错误消息 */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
          >
            <div className="flex items-center">
              <XCircle className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-red-700">{error}</span>
            </div>
          </motion.div>
        )}

        {/* 上传区域 */}
        {!result && (
          <motion.div
            className="bg-white rounded-2xl shadow-xl p-8 mb-12 overflow-hidden relative"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            {/* 背景装饰 */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-primary/10 to-secondary/10 rounded-full -mr-20 -mt-20" />
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-gradient-to-tr from-blue-100 to-purple-100 rounded-full -ml-16 -mb-16" />
            
            <div className="relative z-10">
              <motion.h2 
                className="text-2xl font-semibold text-gray-900 mb-8 text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Search className="inline-block h-6 w-6 mr-2 text-primary" />
                上传图片识别
              </motion.h2>

              {/* 拖放区域 */}
              <motion.div
                className={`
                  border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer relative overflow-hidden
                  ${isDragging 
                    ? 'border-primary bg-blue-50 ring-2 ring-primary/30' 
                    : 'border-gray-300 hover:border-primary hover:bg-gradient-to-b from-white to-blue-50'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                whileHover={{ y: -5, boxShadow: "0 12px 24px -8px rgba(0, 0, 0, 0.1)" }}
                whileTap={{ y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                
                <motion.div
                  animate={isDragging ? { scale: 1.05, rotate: 5 } : { scale: 1, rotate: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative z-10"
                >
                  <div className={`
                    w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center
                    ${isDragging ? 'bg-primary text-white' : 'bg-gray-100 text-gray-400'}
                  `}>
                    <Upload className="h-10 w-10" />
                  </div>
                  <h3 className="text-xl font-medium text-gray-900 mb-2">
                    {isDragging ? '释放图片开始上传' : '点击或拖拽图片到此处'}
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    支持 PNG, JPG, JPEG, GIF, BMP 格式
                  </p>
                  <div className="inline-block px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium">
                    最大文件大小: 16MB
                  </div>
                </motion.div>
              </motion.div>

              {/* 预览图片 */}
              {previewImage && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="mt-10"
                >
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.4 }}
                    className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 shadow-md"
                  >
                    <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                      <ImageIcon className="h-5 w-5 mr-2 text-primary" />
                      图片预览
                    </h3>
                    <div className="flex justify-center mb-6">
                      <motion.img
                        src={previewImage}
                        alt="预览"
                        className="max-h-80 rounded-lg shadow-lg border border-gray-200"
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                      />
                    </div>
                    <div className="flex justify-center space-x-6">
                      <motion.button
                        whileHover={{ scale: 1.05, backgroundColor: "#e5e7eb" }}
                        whileTap={{ scale: 0.95 }}
                        onClick={resetForm}
                        className="px-8 py-3 bg-gray-200 text-gray-800 rounded-lg flex items-center shadow-sm hover:shadow-md transition-all"
                      >
                        <RefreshCw className="h-5 w-5 mr-2" />
                        <span className="font-medium">重新选择</span>
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05, backgroundColor: "#45a049" }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleUpload}
                        disabled={isLoading}
                        className="px-8 py-3 bg-primary text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                            <span className="font-medium">识别中...</span>
                          </>
                        ) : (
                          <>
                            <Search className="h-5 w-5 mr-2" />
                            <span className="font-medium">开始识别</span>
                          </>
                        )}
                      </motion.button>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}

        {/* 分类结果 */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Result State Debug:</h3>
          <pre className="bg-gray-100 p-4 rounded-lg text-sm">
            Result: {JSON.stringify(result, null, 2)}
          </pre>
        </div>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="bg-white rounded-2xl shadow-xl p-8 mb-12 overflow-hidden relative"
          >
            {/* 背景装饰 */}
            <div className="absolute top-0 left-0 w-40 h-40 bg-gradient-to-br from-primary/10 to-secondary/10 rounded-full -ml-20 -mt-20" />
            <div className="absolute bottom-0 right-0 w-32 h-32 bg-gradient-to-tr from-green-100 to-blue-100 rounded-full -mr-16 -mb-16" />
            
            <div className="relative z-10">
              <motion.div 
                className="flex items-center mb-8"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <motion.button
                  whileHover={{ scale: 1.1, backgroundColor: "#f3f4f6" }}
                  whileTap={{ scale: 0.95 }}
                  onClick={resetForm}
                  className="p-3 rounded-full hover:bg-gray-100 mr-4 shadow-sm"
                >
                  <ArrowLeft className="h-5 w-5 text-gray-700" />
                </motion.button>
                <motion.h2 
                  className="text-2xl font-semibold text-gray-900"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <BarChart2 className="inline-block h-6 w-6 mr-2 text-primary" />
                  识别结果
                </motion.h2>
              </motion.div>

              {/* 图片预览 */}
              <motion.div 
                className="mb-10"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <ImageIcon className="h-5 w-5 mr-2 text-primary" />
                  上传的图片
                </h3>
                <div className="flex justify-center">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.5 }}
                    className="relative"
                  >
                    <img
                      src={previewImage || ''}
                      alt="上传的图片"
                      className="max-h-80 rounded-xl shadow-lg border border-gray-200"
                    />
                    <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full shadow-sm text-sm font-medium text-gray-700">
                      {result.filename}
                    </div>
                  </motion.div>
                </div>
              </motion.div>

              {/* 结果卡片 */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 }}
                className="bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl p-8 border border-green-100 shadow-sm"
              >
                <motion.h3 
                  className="text-xl font-semibold text-gray-900 mb-6 flex items-center"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.7 }}
                >
                  <Sparkles className="h-6 w-6 mr-2 text-primary" />
                  AI 识别结果
                </motion.h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* 角色信息 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.8 }}
                    className="bg-white rounded-xl p-5 shadow-sm hover:shadow-md transition-all"
                  >
                    <h4 className="text-sm font-medium text-gray-500 mb-3">
                      识别角色
                    </h4>
                    <p className="text-lg font-bold text-gray-900">
                      {result.role || '未知'}
                    </p>
                  </motion.div>

                  {/* 置信度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.9 }}
                    className="bg-white rounded-xl p-5 shadow-sm hover:shadow-md transition-all"
                  >
                    <h4 className="text-sm font-medium text-gray-500 mb-3">
                      置信度
                    </h4>
                    <div className="flex items-center mb-3">
                      <span className="text-lg font-bold text-gray-900 mr-3">
                        {(result.similarity * 100).toFixed(2)}%
                      </span>
                      <span
                        className={`
                          px-3 py-1 rounded-full text-xs font-medium
                          ${getAccuracyBadgeClass(result.similarity)}
                        `}
                      >
                        {getAccuracyText(result.similarity)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.similarity * 100}%` }}
                        transition={{ duration: 1.2, ease: "easeOut", delay: 1 }}
                        className={`h-3 rounded-full ${result.similarity >= 0.8 ? 'bg-green-500' : result.similarity >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      />
                    </div>
                  </motion.div>

                  {/* 识别速度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 1 }}
                    className="bg-white rounded-xl p-5 shadow-sm hover:shadow-md transition-all"
                  >
                    <h4 className="text-sm font-medium text-gray-500 mb-3">
                      识别速度
                    </h4>
                    <div className="flex items-center">
                      <Zap className="h-5 w-5 text-yellow-500 mr-2" />
                      <span className="text-lg font-semibold text-gray-900">
                        约 2 秒
                      </span>
                    </div>
                  </motion.div>
                </div>
              </motion.div>

              {/* 操作按钮 */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.1 }}
                className="mt-10 flex justify-center space-x-6"
              >
                <motion.button
                  whileHover={{ scale: 1.05, backgroundColor: "#e5e7eb" }}
                  whileTap={{ scale: 0.95 }}
                  onClick={resetForm}
                  className="px-8 py-4 bg-gray-200 text-gray-800 rounded-xl font-medium shadow-sm hover:shadow-md transition-all"
                >
                  <RefreshCw className="inline-block h-5 w-5 mr-2" />
                  上传另一张
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05, backgroundColor: "#1976d2" }}
                  whileTap={{ scale: 0.95 }}
                  className="px-8 py-4 bg-secondary text-white rounded-xl font-medium shadow-md hover:shadow-lg transition-all"
                >
                  <Search className="inline-block h-5 w-5 mr-2" />
                  查看详情
                </motion.button>
              </motion.div>
            </div>
          </motion.div>
        )}

        {/* 系统信息 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-12 bg-white rounded-xl shadow-md p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              ℹ️ 系统信息
            </h3>
            <ChevronDown className="h-5 w-5 text-gray-500" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-2">
                支持的游戏
              </h4>
              <ul className="text-gray-900 space-y-1">
                <li>• 蔚蓝档案 (Blue Archive)</li>
                <li>• 原神 (Genshin Impact)</li>
                <li>• 鸣潮 (Wuthering Waves)</li>
                <li>• 绝区零 (Zenless Zone Zero)</li>
                <li>• 崩坏三 (Honkai Impact 3rd)</li>
                <li>• 崩坏星穹铁道 (Honkai: Star Rail)</li>
                <li>• 崩坏二 (Guns GirlZ)</li>
                <li>• 幻塔 (Tower of Fantasy)</li>
                <li>• 明日方舟 (Arknights)</li>
                <li>• 终末地 (The End Earth)</li>
                <li>• 我推的孩子 (Oshi no Ko)</li>
                <li>• 间谍过家家 (Spy x Family)</li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-2">
                系统性能
              </h4>
              <ul className="text-gray-900 space-y-1">
                <li>• 支持角色: 60+</li>
                <li>• 平均准确率: 54%</li>
                <li>• 处理速度: ~2秒/张</li>
                <li>• 支持格式: PNG, JPG, JPEG, GIF, BMP</li>
                <li>• 最大文件大小: 16MB</li>
              </ul>
              <h4 className="text-sm font-medium text-gray-500 mt-4 mb-2">
              技术特点
            </h4>
            <ul className="text-gray-900 space-y-1">
              <li>• 基于CLIP模型的特征提取</li>
              <li>• Faiss索引加速相似度搜索</li>
              <li>• YOLOv8目标检测</li>
              <li>• 实时分类结果</li>
              <li>• 响应式Web界面</li>
              <li>• 自动化数据集扩充</li>
              <li>• 模型蒸馏技术</li>
              <li>• 在线学习能力</li>
              <li>• 多模态融合系统</li>
            </ul>
            </div>
          </div>
        </motion.div>
      </main>

      {/* 历史记录模态框 */}
      {showHistory && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.9, y: 20, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.9, y: 20, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
          >
            {/* 模态框头部 */}
            <div className="bg-gradient-to-r from-primary to-secondary p-6 text-white flex justify-between items-center">
              <h2 className="text-2xl font-bold flex items-center">
                <BarChart2 className="h-6 w-6 mr-2" />
                分类历史记录
              </h2>
              <div className="flex space-x-3">
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={exportHistory}
                  className="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg flex items-center"
                >
                  <Download className="h-4 w-4 mr-2" />
                  导出
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={clearHistory}
                  className="bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg flex items-center"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  清空
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowHistory(false)}
                  className="bg-white bg-opacity-30 hover:bg-opacity-40 p-2 rounded-full"
                >
                  <X className="h-5 w-5" />
                </motion.button>
              </div>
            </div>

            {/* 历史记录列表 */}
            <div className="flex-1 overflow-y-auto p-6">
              {history.length === 0 ? (
                <div className="text-center py-20">
                  <Clock className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-xl font-medium text-gray-500">暂无历史记录</h3>
                  <p className="text-gray-400 mt-2">上传并分类图片后，结果会显示在这里</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {history.map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                      className="bg-gray-50 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start space-x-4">
                        {/* 图片预览 */}
                        {item.imageData && (
                          <div className="flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden bg-gray-200">
                            <img 
                              src={item.imageData} 
                              alt={item.filename} 
                              className="w-full h-full object-cover"
                            />
                          </div>
                        )}
                        
                        {/* 分类信息 */}
                        <div className="flex-1">
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="text-lg font-semibold text-gray-900">{item.role}</h4>
                            <span className="text-sm text-gray-500">
                              {new Date(item.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <div className="mb-3">
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-600">置信度</span>
                              <span className="font-medium text-gray-900">
                                {(item.similarity * 100).toFixed(2)}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-gradient-to-r from-primary to-secondary h-2 rounded-full" 
                                style={{ width: `${Math.min(item.similarity * 100, 100)}%` }}
                              />
                            </div>
                          </div>
                          <div className="text-sm text-gray-600">
                            文件名: {item.filename}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* 模态框底部 */}
            <div className="border-t border-gray-200 p-6 flex justify-between items-center">
              <div className="text-sm text-gray-500">
                共 {history.length} 条记录
              </div>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "#e5e7eb" }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowHistory(false)}
                className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg font-medium"
              >
                关闭
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* 页脚 */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <p className="text-lg font-semibold">角色分类系统</p>
              <p className="text-gray-400">让角色识别变得简单！</p>
            </div>
            <div className="text-gray-400">
              © 2026 角色分类系统
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
