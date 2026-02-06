'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Image as ImageIcon,
  Video,
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
  Film,
} from 'lucide-react';

interface ClassificationResult {
  filename: string;
  role: string;
  similarity: number;
  boxes: any[];
  fileType?: 'image' | 'video';
  videoResults?: {
    frame: number;
    role: string;
    similarity: number;
    timestamp: number;
  }[];
}

interface HistoryItem extends ClassificationResult {
  timestamp: number;
  imageData?: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<'image' | 'video' | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewVideo, setPreviewVideo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
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
    // 文件大小检查
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
      setError('文件大小超过限制，请选择小于16MB的文件');
      return;
    }
    
    if (file.type.startsWith('image/')) {
      setFileType('image');
      setError(null);
      setResult(null);
      setSelectedFile(file);
      
      // 创建图片预览
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result as string);
        setPreviewVideo(null);
      };
      reader.onerror = () => {
        setError('文件读取失败，请重试');
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      setFileType('video');
      setError(null);
      setResult(null);
      setSelectedFile(file);
      
      // 创建视频预览
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewVideo(reader.result as string);
        setPreviewImage(null);
      };
      reader.onerror = () => {
        setError('视频文件读取失败，请重试');
      };
      reader.readAsDataURL(file);
    } else {
      setError('请选择图片或视频文件');
      return;
    }
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

  // 添加全局错误监听器
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // 添加全局错误监听器
      const handleGlobalError = (event: ErrorEvent) => {
        console.error('🌐 全局错误:', event.error);
        console.error('🌐 错误堆栈:', event.error?.stack);
        console.error('🌐 错误发生在:', event.filename, '行号:', event.lineno, '列号:', event.colno);
      };

      // 添加全局未捕获Promise错误监听器
      const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
        console.error('🌐 未捕获的Promise错误:', event.reason);
        console.error('🌐 Promise:', event.promise);
      };

      // 添加事件监听器
      window.addEventListener('error', handleGlobalError);
      window.addEventListener('unhandledrejection', handleUnhandledRejection);

      // 全局变量，用于测试函数是否被调用
      (window as any).testHandleUpload = function() {
        console.log('🌐 全局测试函数被调用！');
        alert('全局测试函数被调用！');
      };

      // 清理函数
      return () => {
        window.removeEventListener('error', handleGlobalError);
        window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      };
    }
  }, []);

  const handleUpload = () => {
    // 检查是否选择了文件
    if (!selectedFile) {
      alert('请先选择一个文件！');
      return;
    }
    
    console.log('🔄 handleUpload函数被调用！');
    console.log('🔍 当前状态:', {
      selectedFile: selectedFile ? selectedFile.name : null,
      fileType: fileType,
      isLoading: isLoading,
      error: error
    });
    
    // 设置加载状态
    setIsLoading(true);
    setError(null);
    setProcessingStatus(previewVideo ? '正在处理视频...' : '正在识别图像...');
    
    // 创建FormData对象，用于上传文件
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('use_model', 'true');
    if (previewVideo) {
      formData.append('frame_skip', '5'); // 视频帧跳过间隔
    }
    
    // 发送POST请求到Next.js API路由
    console.log('🌐 尝试调用API...');
    fetch('/api/classify', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      console.log('📡 API响应状态:', response.status);
      if (!response.ok) {
        throw new Error('API响应失败: ' + response.statusText);
      }
      return response.json();
    })
    .then(data => {
      console.log('📡 API响应数据:', data);
      setResult(data);
      setIsLoading(false);
      setProcessingStatus(null);
      alert('处理成功！');
    })
    .catch(error => {
      console.error('❌ API调用失败:', error);
      setError('处理失败: ' + error.message);
      setIsLoading(false);
      setProcessingStatus(null);
      alert('处理失败: ' + error.message);
    });
  };

  const resetForm = () => {
    setSelectedFile(null);
    setFileType(null);
    setPreviewImage(null);
    setPreviewVideo(null);
    setResult(null);
    setError(null);
    setUploadProgress(0);
    setProcessingStatus(null);
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
    <div className="min-h-screen bg-gradient-to-b from-primary-100 to-secondary-100">
      {/* 导航栏 */}
      <nav className="bg-gradient-to-r from-primary-500/95 to-secondary-500/95 backdrop-blur-lg sticky top-0 z-50 shadow-md">
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
                className="bg-white/20 backdrop-blur-sm p-2 rounded-full shadow-lg"
              >
                <Sparkles className="h-6 w-6 text-white" />
              </motion.div>
              <span className="ml-3 text-xl font-bold font-display text-white hidden sm:block">角色智能识别系统</span>
              <span className="ml-3 text-lg font-bold font-display text-white sm:hidden">角色识别</span>
            </motion.div>
            <motion.div 
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-lg hover:shadow-xl transition-all hidden sm:flex"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                <span className="font-medium">AI 分类</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="p-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-lg hover:shadow-xl transition-all sm:hidden"
              >
                <Sparkles className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all hidden sm:flex"
                onClick={() => {
                  setShowHistory(true);
                  loadHistory();
                }}
              >
                <BarChart2 className="h-4 w-4 mr-2" />
                <span className="font-medium">历史记录</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="p-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all sm:hidden"
                onClick={() => {
                  setShowHistory(true);
                  loadHistory();
                }}
              >
                <BarChart2 className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all hidden sm:flex"
              >
                <Info className="h-4 w-4 mr-2" />
                <span className="font-medium">关于</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.2)" }}
                whileTap={{ scale: 0.95 }}
                className="p-2 bg-white/10 backdrop-blur-sm text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all sm:hidden"
              >
                <Info className="h-5 w-5" />
              </motion.button>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* 主内容 */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* 标题 */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="text-center mb-16 sm:mb-20"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="inline-flex items-center justify-center w-20 sm:w-24 h-20 sm:h-24 bg-gradient-to-br from-primary-500 to-secondary-600 rounded-full mb-6 sm:mb-8 shadow-xl animate-float"
          >
            <Sparkles className="h-10 sm:h-12 w-10 sm:w-12 text-white" />
          </motion.div>
          <motion.h1 
            className="text-[clamp(2rem,5vw,4rem)] font-extrabold font-display mb-4 sm:mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            角色智能识别系统
          </motion.h1>
          <motion.p 
            className="text-base sm:text-xl text-dark-600 max-w-2xl mx-auto leading-relaxed"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            利用先进的人工智能技术，一键识别图片和视频中的游戏角色，精准定位并分析角色特征
          </motion.p>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.8 }}
            className="mt-8 sm:mt-10 flex flex-wrap justify-center gap-3 sm:gap-4"
          >
            <motion.div 
              whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-xl shadow-lg border border-light-200"
            >
              <Star className="h-5 sm:h-6 w-5 sm:w-6 text-accent-500 mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-semibold text-dark-700">60+ 角色支持</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-xl shadow-lg border border-light-200"
            >
              <Zap className="h-5 sm:h-6 w-5 sm:w-6 text-primary-500 mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-semibold text-dark-700">实时识别</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-xl shadow-lg border border-light-200"
            >
              <Award className="h-5 sm:h-6 w-5 sm:w-6 text-secondary-500 mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-semibold text-dark-700">高准确率</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-xl shadow-lg border border-light-200"
            >
              <Film className="h-5 sm:h-6 w-5 sm:w-6 text-green-500 mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-semibold text-dark-700">视频支持</span>
            </motion.div>
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
            className="bg-white rounded-2xl shadow-xl p-8 mb-16 overflow-hidden relative"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            {/* 背景装饰 */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-primary-100 to-secondary-100 rounded-full -mr-20 -mt-20" />
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-gradient-to-tr from-blue-50 to-purple-50 rounded-full -ml-16 -mb-16" />
            
            <div className="relative z-10">
              <motion.h2 
                className="text-2xl font-semibold text-dark-900 mb-8 text-center font-display"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Search className="inline-block h-6 w-6 mr-2 text-primary-500" />
                上传文件识别
              </motion.h2>

              {/* 拖放区域 */}
              <motion.div
                className={`
                  border-2 border-dashed rounded-2xl sm:rounded-3xl p-8 sm:p-12 text-center cursor-pointer relative overflow-hidden
                  ${isDragging 
                    ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-300 shadow-xl' 
                    : 'border-light-300 hover:border-primary-500 hover:bg-gradient-to-b from-white to-primary-50'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                whileHover={{ y: -8, boxShadow: "0 20px 40px -15px rgba(0, 0, 0, 0.15)" }}
                whileTap={{ y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*, video/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                
                <motion.div
                  animate={isDragging ? { scale: 1.05, rotate: 5 } : { scale: 1, rotate: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative z-10"
                >
                  <div className={`
                    w-20 sm:w-28 h-20 sm:h-28 mx-auto mb-6 sm:mb-8 rounded-full flex items-center justify-center
                    ${isDragging ? 'bg-gradient-to-br from-primary-500 to-secondary-500 text-white shadow-2xl' : 'bg-gradient-to-br from-light-200 to-light-100 text-primary-600'}
                  `}>
                    <Upload className="h-10 sm:h-14 w-10 sm:w-14" />
                  </div>
                  <h3 className="text-xl sm:text-2xl font-semibold text-dark-900 mb-3 sm:mb-4">
                    {isDragging ? '释放文件开始上传' : '点击或拖拽文件到此处'}
                  </h3>
                  <p className="text-xs sm:text-sm text-dark-500 mb-6 sm:mb-8 max-w-md mx-auto leading-relaxed">
                    支持 PNG, JPG, JPEG, GIF, BMP 图片格式和 MP4, AVI, MOV 视频格式
                  </p>
                  <div className="inline-block px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-light-200 to-light-100 text-dark-700 rounded-lg sm:rounded-xl text-xs sm:text-sm font-semibold shadow-md">
                    最大文件大小: 16MB
                  </div>
                </motion.div>
              </motion.div>

              {/* 预览区域 */}
              {(previewImage || previewVideo) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="mt-12"
                >
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.4 }}
                    className="bg-gradient-to-br from-light-100 to-light-200 rounded-xl p-8 shadow-md border border-light-300"
                  >
                    <h3 className="text-lg font-semibold text-dark-900 mb-6 flex items-center font-display">
                      {previewImage ? (
                        <>
                          <ImageIcon className="h-5 w-5 mr-2 text-primary-500" />
                          图片预览
                        </>
                      ) : (
                        <>
                          <Video className="h-5 w-5 mr-2 text-primary-500" />
                          视频预览
                        </>
                      )}
                    </h3>
                    <div className="flex justify-center mb-8">
                      {previewImage && (
                        <motion.div
                          className="relative"
                          initial={{ scale: 0.9, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.5, delay: 0.2 }}
                        >
                          <img
                            src={previewImage}
                            alt="预览"
                            className="max-h-96 rounded-xl shadow-lg border border-light-300"
                          />
                          <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-dark-700 shadow-sm">
                            图片
                          </div>
                        </motion.div>
                      )}
                      {previewVideo && (
                        <motion.div
                          className="relative max-h-96 rounded-xl shadow-lg border border-light-300 overflow-hidden"
                          initial={{ scale: 0.9, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.5, delay: 0.2 }}
                        >
                          <video
                            src={previewVideo}
                            controls
                            className="w-full h-full"
                          >
                            您的浏览器不支持视频播放。
                          </video>
                          <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-dark-700 shadow-sm">
                            视频
                          </div>
                        </motion.div>
                      )}
                    </div>
                    {/* 上传进度和处理状态 */}
                    {isLoading && (
                      <div className="mt-6 space-y-4">
                        {processingStatus && (
                          <p className="text-sm text-dark-600 animate-pulse flex items-center justify-center">
                            <Clock className="h-4 w-4 mr-2" />
                            {processingStatus}
                          </p>
                        )}
                        {uploadProgress > 0 && uploadProgress < 100 && (
                          <div className="w-full bg-light-300 rounded-full h-3">
                            <motion.div 
                              className="bg-gradient-to-r from-primary-500 to-secondary-500 h-3 rounded-full"
                              initial={{ width: '0%' }}
                              animate={{ width: `${uploadProgress}%` }}
                              transition={{ duration: 0.3 }}
                            />
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="flex justify-center space-x-6 mt-8">
                      <motion.button
                        whileHover={{ scale: 1.05, backgroundColor: "#f1f5f9" }}
                        whileTap={{ scale: 0.95 }}
                        onClick={resetForm}
                        className="px-8 py-3 bg-light-200 text-dark-800 rounded-lg flex items-center shadow-md hover:shadow-lg transition-all"
                      >
                        <RefreshCw className="h-5 w-5 mr-2" />
                        <span className="font-medium">重新选择</span>
                      </motion.button>
                      {/* 简化的测试按钮，使用普通的HTML按钮 */}
                      <motion.button
                        whileHover={{ scale: 1.05, backgroundColor: "rgba(34, 197, 94, 0.9)" }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleUpload}
                        disabled={isLoading}
                        className="px-8 py-3 bg-primary-500 text-white rounded-lg flex items-center shadow-lg hover:shadow-xl transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                            <span className="font-medium">{previewVideo ? '处理中...' : '识别中...'}</span>
                          </>
                        ) : (
                          <>
                            <Search className="h-5 w-5 mr-2" />
                            <span className="font-medium">{previewVideo ? '开始处理' : '开始识别'}</span>
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

              {/* 文件预览 */}
              <motion.div 
                className="mb-10"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  {result.fileType === 'image' ? (
                    <>
                      <ImageIcon className="h-5 w-5 mr-2 text-primary" />
                      上传的图片
                    </>
                  ) : (
                    <>
                      <Video className="h-5 w-5 mr-2 text-primary" />
                      上传的视频
                    </>
                  )}
                </h3>
                <div className="flex justify-center">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.5 }}
                    className="relative"
                  >
                    {result.fileType === 'image' ? (
                      <div className="relative">
                        <img
                          src={previewImage || ''}
                          alt="上传的图片"
                          className="max-h-80 rounded-xl shadow-lg border border-gray-200"
                          ref={(img) => {
                            if (img && result.boxes) {
                              // 计算图片的实际尺寸和缩放比例
                              const imgWidth = img.offsetWidth;
                              const imgHeight = img.offsetHeight;
                              
                              // 清除之前的边界框
                              const existingBoxes = img.parentElement?.querySelectorAll('.bounding-box');
                              existingBoxes?.forEach(box => box.remove());
                              
                              // 绘制边界框
                              result.boxes.forEach((box, index) => {
                                const [x1, y1, x2, y2] = box.bbox;
                                const confidence = box.confidence;
                                
                                // 创建边界框元素
                                const boxElement = document.createElement('div');
                                boxElement.className = 'bounding-box absolute border-2 border-red-500 rounded-md';
                                boxElement.style.left = `${x1}px`;
                                boxElement.style.top = `${y1}px`;
                                boxElement.style.width = `${x2 - x1}px`;
                                boxElement.style.height = `${y2 - y1}px`;
                                boxElement.style.zIndex = '10';
                                
                                // 创建标签元素
                                const labelElement = document.createElement('div');
                                labelElement.className = 'absolute -top-6 left-0 bg-red-500 text-white text-xs px-2 py-1 rounded';
                                labelElement.textContent = `${result.role || '未知'} (${(confidence * 100).toFixed(1)}%)`;
                                
                                // 添加到DOM
                                boxElement.appendChild(labelElement);
                                img.parentElement?.appendChild(boxElement);
                              });
                            }
                          }}
                        />
                        {result.boxes && result.boxes.length > 0 && (
                          <div className="absolute inset-0 pointer-events-none">
                            {/* 边界框会通过ref动态添加 */}
                          </div>
                        )}
                      </div>
                    ) : (
                      <video
                        src={previewVideo || ''}
                        controls
                        className="max-h-80 rounded-xl shadow-lg border border-gray-200"
                      >
                        您的浏览器不支持视频播放。
                      </video>
                    )}
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
                className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-2xl sm:rounded-3xl p-6 sm:p-8 border border-primary-100 shadow-lg"
              >
                <motion.h3 
                  className="text-lg sm:text-xl font-semibold text-dark-900 mb-4 sm:mb-6 flex items-center"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.7 }}
                >
                  <Sparkles className="h-5 sm:h-6 w-5 sm:w-6 mr-2 text-primary-500" />
                  AI 识别结果
                </motion.h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
                  {/* 角色信息 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.8 }}
                    whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
                    className="bg-white rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-lg border border-light-200 hover:border-primary-200 transition-all"
                  >
                    <h4 className="text-xs sm:text-sm font-medium text-dark-500 mb-2 sm:mb-3">
                      {result.fileType === 'image' ? '识别角色' : '主要角色'}
                    </h4>
                    <p className="text-xl sm:text-2xl font-bold text-dark-900">
                      {result.role || '未知'}
                    </p>
                  </motion.div>

                  {/* 置信度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.9 }}
                    whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
                    className="bg-white rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-lg border border-light-200 hover:border-primary-200 transition-all"
                  >
                    <h4 className="text-xs sm:text-sm font-medium text-dark-500 mb-2 sm:mb-3">
                      置信度
                    </h4>
                    <div className="flex items-center mb-3 sm:mb-4">
                      <span className="text-xl sm:text-2xl font-bold text-dark-900 mr-2 sm:mr-3">
                        {(result.similarity * 100).toFixed(2)}%
                      </span>
                      <span
                        className={`
                          px-3 sm:px-4 py-1 rounded-full text-xs font-medium
                          ${getAccuracyBadgeClass(result.similarity)}
                        `}
                      >
                        {getAccuracyText(result.similarity)}
                      </span>
                    </div>
                    <div className="w-full bg-light-200 rounded-full h-3 sm:h-4">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.similarity * 100}%` }}
                        transition={{ duration: 1.2, ease: "easeOut", delay: 1 }}
                        className={`h-3 sm:h-4 rounded-full ${result.similarity >= 0.8 ? 'bg-gradient-to-r from-primary-500 to-primary-600' : result.similarity >= 0.5 ? 'bg-gradient-to-r from-accent-500 to-accent-600' : 'bg-gradient-to-r from-red-500 to-red-600'}`}
                      />
                    </div>
                  </motion.div>

                  {/* 识别速度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 1 }}
                    whileHover={{ y: -8, boxShadow: "0 15px 30px -10px rgba(0, 0, 0, 0.15)" }}
                    className="bg-white rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-lg border border-light-200 hover:border-primary-200 transition-all"
                  >
                    <h4 className="text-xs sm:text-sm font-medium text-dark-500 mb-2 sm:mb-3">
                      处理速度
                    </h4>
                    <div className="flex items-center">
                      <Zap className="h-5 sm:h-6 w-5 sm:w-6 text-accent-500 mr-2 sm:mr-3" />
                      <span className="text-lg sm:text-xl font-semibold text-dark-900">
                        {result.fileType === 'image' ? '约 2 秒' : '约 10 秒'}
                      </span>
                    </div>
                  </motion.div>
                </div>

                {/* 视频帧检测结果 */}
                {result.fileType === 'video' && result.videoResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 1.1 }}
                    className="mt-10"
                  >
                    <h4 className="text-lg font-semibold text-dark-900 mb-4 flex items-center">
                      <Film className="h-6 w-6 mr-3 text-primary-500" />
                      视频帧检测结果
                    </h4>
                    <div className="bg-white rounded-2xl p-6 shadow-lg border border-light-200 max-h-96 overflow-y-auto">
                      <div className="space-y-4">
                        {result.videoResults.map((frameResult, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: 1.2 + index * 0.1 }}
                            whileHover={{ y: -4, boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
                            className="flex items-center justify-between p-4 border border-light-200 rounded-xl hover:bg-light-50 transition-all"
                          >
                            <div className="flex items-center">
                              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-100 to-secondary-100 flex items-center justify-center mr-4">
                                <span className="text-sm font-semibold text-primary-600">{frameResult.frame}</span>
                              </div>
                              <div>
                                <p className="font-semibold text-dark-900">{frameResult.role}</p>
                                <p className="text-xs text-dark-500">时间: {frameResult.timestamp.toFixed(1)}秒</p>
                              </div>
                            </div>
                            <div className="flex items-center">
                              <div className="w-32 bg-light-200 rounded-full h-3 mr-4">
                                <div 
                                  className={`h-3 rounded-full ${frameResult.similarity >= 0.8 ? 'bg-gradient-to-r from-primary-500 to-primary-600' : frameResult.similarity >= 0.5 ? 'bg-gradient-to-r from-accent-500 to-accent-600' : 'bg-gradient-to-r from-red-500 to-red-600'}`} 
                                  style={{ width: `${(frameResult.similarity * 100).toFixed(0)}%` }}
                                />
                              </div>
                              <span className="text-sm font-semibold text-dark-900">
                                {(frameResult.similarity * 100).toFixed(1)}%
                              </span>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
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
                  {result.fileType === 'image' ? '上传另一张' : '上传另一个视频'}
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
                <li>• 支持格式: PNG, JPG, JPEG, GIF, BMP, MP4, AVI, MOV</li>
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
