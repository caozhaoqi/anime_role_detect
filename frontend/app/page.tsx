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
  generatedVideoUrl?: string; // URL of the video with bounding boxes generated locally
}

interface HistoryItem extends ClassificationResult {
  timestamp: number;
  imageData?: string;
}

type WorkflowStep = 'upload' | 'preview' | 'processing' | 'result';

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
  const [workflowStep, setWorkflowStep] = useState<WorkflowStep>('upload');
  const [videoFrameProgress, setVideoFrameProgress] = useState<{current: number, total: number}>({current: 0, total: 0});
  const [models, setModels] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
      setHistory([]);
    }
  };

  // 处理视频播放和边界框绘制
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas || !result || result.fileType !== 'video' || !result.videoResults) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置canvas尺寸与视频匹配
    const resizeCanvas = () => {
      canvas.width = video.offsetWidth;
      canvas.height = video.offsetHeight;
    };

    // 初始调整尺寸
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // 查找当前时间对应的帧结果
    const findCurrentFrameResult = (currentTime: number) => {
      return result.videoResults?.reduce((closest, frameResult) => {
        const closestDiff = Math.abs(closest.timestamp - currentTime);
        const currentDiff = Math.abs(frameResult.timestamp - currentTime);
        return currentDiff < closestDiff ? frameResult : closest;
      }, result.videoResults[0]) || null;
    };

    // 绘制边界框和标签
    const drawBoxes = () => {
      if (!ctx) return;
      
      // 清空画布
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const currentFrameResult = findCurrentFrameResult(video.currentTime);
      if (!currentFrameResult) return;

      // 假设每个视频帧结果都有对应的边界框信息
      // 这里简化处理，实际应用中需要根据API返回的边界框数据
      const box = {
        bbox: [50, 50, canvas.width - 50, canvas.height - 50], // 示例边界框
        confidence: currentFrameResult.similarity
      };

      if (box) {
        const [x1, y1, x2, y2] = box.bbox;
        const confidence = box.confidence;
        
        // 绘制边界框
        ctx.strokeStyle = '#409EFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // 绘制标签背景
        ctx.fillStyle = '#409EFF';
        const label = `${currentFrameResult.role} (${(confidence * 100).toFixed(1)}%)`;
        const labelWidth = ctx.measureText(label).width + 12;
        const labelHeight = 20;
        
        ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);
        
        // 绘制标签文本
        ctx.fillStyle = 'white';
        ctx.font = '12px sans-serif';
        ctx.fillText(label, x1 + 6, y1 - 5);
      }
    };

    // 监听视频时间更新
    video.addEventListener('timeupdate', drawBoxes);
    video.addEventListener('play', drawBoxes);
    video.addEventListener('seeked', drawBoxes);

    // 清理函数
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      video.removeEventListener('timeupdate', drawBoxes);
      video.removeEventListener('play', drawBoxes);
      video.removeEventListener('seeked', drawBoxes);
    };
  }, [result]);

  // 加载历史记录
  useEffect(() => {
    loadHistory();
    fetchModels();
  }, []);

  // 获取模型列表
  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
        // 默认选择第一个模型
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0].name);
        }
      }
    } catch (error) {
      console.error('获取模型列表失败:', error);
    } finally {
      setLoadingModels(false);
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
      setWorkflowStep('preview');
      
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
      setWorkflowStep('preview');
      
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
    setWorkflowStep('processing');
    
    // 模拟视频帧处理进度
    let progressInterval: NodeJS.Timeout | null = null;
    if (previewVideo) {
      setVideoFrameProgress({ current: 0, total: 50 }); // 假设处理50帧
      progressInterval = setInterval(() => {
        setVideoFrameProgress(prev => {
          const newCurrent = prev.current + 1;
          if (newCurrent >= prev.total) {
            if (progressInterval) clearInterval(progressInterval);
            return prev;
          }
          return { ...prev, current: newCurrent };
        });
      }, 150); // 每150ms更新一帧
    }
    
    // 创建FormData对象，用于上传文件
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('use_model', 'true');
    formData.append('model', selectedModel); // 添加模型参数
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
      if (progressInterval) clearInterval(progressInterval);
      setResult(data);
      setIsLoading(false);
      setProcessingStatus(null);
      setWorkflowStep('result');
      alert('处理成功！');
    })
    .catch(error => {
      console.error('❌ API调用失败:', error);
      if (progressInterval) clearInterval(progressInterval);
      setError('处理失败: ' + error.message);
      setIsLoading(false);
      setProcessingStatus(null);
      // 保持在当前步骤以便用户可以重试
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
    setWorkflowStep('upload');
    setVideoFrameProgress({ current: 0, total: 0 });
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
    <div className="min-h-screen bg-background">
      {/* 导航栏 - DeepSeek and Element UI style */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
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
                className="bg-primary-light p-2 rounded-full"
              >
                <Sparkles className="h-6 w-6 text-primary" />
              </motion.div>
              <span className="ml-3 text-xl font-bold font-display text-text-primary hidden sm:block">角色智能识别系统</span>
              <span className="ml-3 text-lg font-bold font-display text-text-primary sm:hidden">角色识别</span>
            </motion.div>
            <motion.div 
              className="flex items-center space-x-2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--primary-light)" }}
                whileTap={{ scale: 0.98 }}
                className="px-4 py-2 bg-white border border-primary text-primary rounded-md flex items-center transition-all hover:shadow-sm hidden sm:flex"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                <span className="font-medium">AI 分类</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--primary-light)" }}
                whileTap={{ scale: 0.98 }}
                className="p-2 bg-white border border-primary text-primary rounded-md flex items-center transition-all hover:shadow-sm sm:hidden"
              >
                <Sparkles className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                whileTap={{ scale: 0.98 }}
                className="px-4 py-2 bg-white border border-border text-text-secondary rounded-md flex items-center transition-all hover:shadow-sm hidden sm:flex"
                onClick={() => {
                  setShowHistory(true);
                  loadHistory();
                }}
              >
                <BarChart2 className="h-4 w-4 mr-2" />
                <span className="font-medium">历史记录</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                whileTap={{ scale: 0.98 }}
                className="p-2 bg-white border border-border text-text-secondary rounded-md flex items-center transition-all hover:shadow-sm sm:hidden"
                onClick={() => {
                  setShowHistory(true);
                  loadHistory();
                }}
              >
                <BarChart2 className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                whileTap={{ scale: 0.98 }}
                className="px-4 py-2 bg-white border border-border text-text-secondary rounded-md flex items-center transition-all hover:shadow-sm hidden sm:flex"
              >
                <Info className="h-4 w-4 mr-2" />
                <span className="font-medium">关于</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                whileTap={{ scale: 0.98 }}
                className="p-2 bg-white border border-border text-text-secondary rounded-md flex items-center transition-all hover:shadow-sm sm:hidden"
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
            className="inline-flex items-center justify-center w-20 sm:w-24 h-20 sm:h-24 bg-primary rounded-full mb-6 sm:mb-8 shadow-lg"
          >
            <Sparkles className="h-10 sm:h-12 w-10 sm:w-12 text-white" />
          </motion.div>
          <motion.h1 
            className="text-[clamp(2rem,5vw,4rem)] font-extrabold font-display mb-4 sm:mb-6 text-text-primary"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            角色智能识别系统
          </motion.h1>
          <motion.p 
            className="text-base sm:text-xl text-text-secondary max-w-2xl mx-auto leading-relaxed"
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
              whileHover={{ y: -4, boxShadow: "var(--card-shadow-hover)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-lg shadow-sm border border-border transition-all"
            >
              <Star className="h-5 sm:h-6 w-5 sm:w-6 text-accent mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-medium text-text-secondary">60+ 角色支持</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -4, boxShadow: "var(--card-shadow-hover)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-lg shadow-sm border border-border transition-all"
            >
              <Zap className="h-5 sm:h-6 w-5 sm:w-6 text-primary mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-medium text-text-secondary">实时识别</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -4, boxShadow: "var(--card-shadow-hover)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-lg shadow-sm border border-border transition-all"
            >
              <Award className="h-5 sm:h-6 w-5 sm:w-6 text-secondary mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-medium text-text-secondary">高准确率</span>
            </motion.div>
            <motion.div 
              whileHover={{ y: -4, boxShadow: "var(--card-shadow-hover)" }}
              className="flex items-center bg-white px-4 sm:px-6 py-3 sm:py-4 rounded-lg shadow-sm border border-border transition-all"
            >
              <Film className="h-5 sm:h-6 w-5 sm:w-6 text-secondary mr-2 sm:mr-3" />
              <span className="text-sm sm:text-base font-medium text-text-secondary">视频支持</span>
            </motion.div>
          </motion.div>
        </motion.div>

        {/* 错误消息 */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6 shadow-md"
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="flex items-center flex-1">
                <XCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />
                <span className="text-red-700">{error}</span>
              </div>
              <div className="flex space-x-3">
                {workflowStep === 'processing' && (
                  <motion.button
                    whileHover={{ scale: 1.05, backgroundColor: "rgba(34, 197, 94, 0.9)" }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleUpload}
                    className="px-4 py-2 bg-green-500 text-white rounded-lg flex items-center shadow-md hover:shadow-lg transition-all text-sm"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    <span>重试</span>
                  </motion.button>
                )}
                <motion.button
                  whileHover={{ scale: 1.05, backgroundColor: "#f1f5f9" }}
                  whileTap={{ scale: 0.95 }}
                  onClick={resetForm}
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg flex items-center shadow-sm hover:shadow-md transition-all text-sm"
                >
                  <X className="h-4 w-4 mr-2" />
                  <span>清除</span>
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* 工作流指示器 */}
        <motion.div
          className="flex justify-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        >
          <div className="flex items-center space-x-1 sm:space-x-4 w-full max-w-2xl">
            {/* 上传步骤 */}
            <motion.div 
              className="flex flex-col items-center flex-1"
              animate={{
                scale: workflowStep === 'upload' ? 1.05 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 sm:mb-3 ${workflowStep === 'upload' || workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? 'bg-primary-500 text-white shadow-lg' : 'bg-gray-200 text-gray-500'}`}>
                {workflowStep === 'upload' || workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <Upload className="h-5 w-5" />
                )}
              </div>
              <span className={`text-xs sm:text-sm font-medium ${workflowStep === 'upload' ? 'text-primary-500 font-bold' : workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? 'text-dark-700' : 'text-gray-500'}`}>
                上传文件
              </span>
            </motion.div>

            {/* 连接线 */}
            <div className={`h-1 flex-1 ${workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? 'bg-primary-500' : 'bg-gray-200'}`} />

            {/* 预览步骤 */}
            <motion.div 
              className="flex flex-col items-center flex-1"
              animate={{
                scale: workflowStep === 'preview' ? 1.05 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 sm:mb-3 ${workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? 'bg-primary-500 text-white shadow-lg' : 'bg-gray-200 text-gray-500'}`}>
                {workflowStep === 'preview' || workflowStep === 'processing' || workflowStep === 'result' ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <ImageIcon className="h-5 w-5" />
                )}
              </div>
              <span className={`text-xs sm:text-sm font-medium ${workflowStep === 'preview' ? 'text-primary-500 font-bold' : workflowStep === 'processing' || workflowStep === 'result' ? 'text-dark-700' : 'text-gray-500'}`}>
                预览文件
              </span>
            </motion.div>

            {/* 连接线 */}
            <div className={`h-1 flex-1 ${workflowStep === 'processing' || workflowStep === 'result' ? 'bg-primary-500' : 'bg-gray-200'}`} />

            {/* 处理步骤 */}
            <motion.div 
              className="flex flex-col items-center flex-1"
              animate={{
                scale: workflowStep === 'processing' ? 1.05 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 sm:mb-3 ${workflowStep === 'processing' || workflowStep === 'result' ? 'bg-primary-500 text-white shadow-lg' : 'bg-gray-200 text-gray-500'}`}>
                {workflowStep === 'processing' ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : workflowStep === 'result' ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <Search className="h-5 w-5" />
                )}
              </div>
              <span className={`text-xs sm:text-sm font-medium ${workflowStep === 'processing' ? 'text-primary-500 font-bold' : workflowStep === 'result' ? 'text-dark-700' : 'text-gray-500'}`}>
                处理中
              </span>
            </motion.div>

            {/* 连接线 */}
            <div className={`h-1 flex-1 ${workflowStep === 'result' ? 'bg-primary-500' : 'bg-gray-200'}`} />

            {/* 结果步骤 */}
            <motion.div 
              className="flex flex-col items-center flex-1"
              animate={{
                scale: workflowStep === 'result' ? 1.05 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 sm:mb-3 ${workflowStep === 'result' ? 'bg-primary-500 text-white shadow-lg' : 'bg-gray-200 text-gray-500'}`}>
                {workflowStep === 'result' ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <BarChart2 className="h-5 w-5" />
                )}
              </div>
              <span className={`text-xs sm:text-sm font-medium ${workflowStep === 'result' ? 'text-primary-500 font-bold' : 'text-gray-500'}`}>
                查看结果
              </span>
            </motion.div>
          </div>
        </motion.div>

        {/* 上传区域 - DeepSeek and Element UI style */}
        {!result && (
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-border p-8 mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            <div className="relative z-10">
              <motion.h2 
                className="text-xl font-semibold text-text-primary mb-8 text-center font-display"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Search className="inline-block h-5 w-5 mr-2 text-primary" />
                上传文件识别
              </motion.h2>

              {/* 拖放区域 */}
              <motion.div
                className={`
                  border-2 border-dashed rounded-lg p-8 sm:p-10 text-center cursor-pointer
                  ${isDragging 
                    ? 'border-primary bg-primary-light' 
                    : 'border-border hover:border-primary hover:bg-primary-light'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                whileHover={{ y: -2, boxShadow: "var(--card-shadow-hover)" }}
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
                  animate={isDragging ? { scale: 1.02 } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                  className="relative z-10"
                >
                  <div className={`
                    w-16 sm:w-20 h-16 sm:h-20 mx-auto mb-6 sm:mb-8 rounded-full flex items-center justify-center
                    ${isDragging ? 'bg-primary text-white' : 'bg-gray-100 text-primary'}
                  `}>
                    <Upload className="h-8 sm:h-10 w-8 sm:w-10" />
                  </div>
                  <h3 className="text-lg sm:text-xl font-medium text-text-primary mb-6 sm:mb-8">
                    {isDragging ? '释放文件开始上传' : '点击或拖拽文件到此处'}
                  </h3>
                  <div className="space-y-4 mb-8 sm:mb-10 max-w-md mx-auto">
                    <p className="text-xs sm:text-sm text-text-light leading-relaxed text-center">
                      支持以下文件格式：
                    </p>
                    <div className="grid grid-cols-2 gap-3 text-center">
                      <motion.div 
                        whileHover={{ y: -2, boxShadow: "var(--card-shadow-hover)" }}
                        className="bg-white rounded-lg p-3 shadow-sm border border-border"
                      >
                        <ImageIcon className="h-5 w-5 text-primary mx-auto mb-2" />
                        <span className="text-xs text-text-secondary font-medium">图片格式</span>
                        <p className="text-xs text-text-light mt-1">PNG, JPG, JPEG, GIF, BMP</p>
                      </motion.div>
                      <motion.div 
                        whileHover={{ y: -2, boxShadow: "var(--card-shadow-hover)" }}
                        className="bg-white rounded-lg p-3 shadow-sm border border-border"
                      >
                        <Video className="h-5 w-5 text-primary mx-auto mb-2" />
                        <span className="text-xs text-text-secondary font-medium">视频格式</span>
                        <p className="text-xs text-text-light mt-1">MP4, AVI, MOV</p>
                      </motion.div>
                    </div>
                    <div className="bg-primary-light rounded-lg p-4 border border-primary/20">
                      <div className="flex items-center justify-center">
                        <Info className="h-4 w-4 text-primary mr-2" />
                        <span className="text-sm font-medium text-primary">最大文件大小: 16MB</span>
                      </div>
                      <p className="text-xs text-text-light mt-2 text-center">
                        建议使用清晰、正面的角色图像以获得最佳识别效果
                      </p>
                    </div>
                  </div>
                </motion.div>
              </motion.div>

              {/* 预览区域 - DeepSeek and Element UI style */}
              {(previewImage || previewVideo) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="mt-8"
                >
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.4 }}
                    className="bg-white rounded-lg p-6 shadow-sm border border-border"
                  >
                    <h3 className="text-lg font-medium text-text-primary mb-6 flex items-center font-display">
                      {previewImage ? (
                        <>
                          <ImageIcon className="h-5 w-5 mr-2 text-primary" />
                          图片预览
                        </>
                      ) : (
                        <>
                          <Video className="h-5 w-5 mr-2 text-primary" />
                          视频预览
                        </>
                      )}
                    </h3>
                    <div className="flex flex-col items-center space-y-6">
                      {/* 模型选择 */}
                      <div className="w-full max-w-md">
                        <label className="block text-sm font-medium text-text-primary mb-2">
                          选择模型
                        </label>
                        <div className="relative">
                          {loadingModels ? (
                            <div className="flex items-center justify-center p-3 border border-border rounded-lg bg-gray-50">
                              <Loader2 className="h-4 w-4 animate-spin text-primary mr-2" />
                              <span className="text-sm text-text-secondary">加载模型列表...</span>
                            </div>
                          ) : (
                            <select
                              value={selectedModel}
                              onChange={(e) => setSelectedModel(e.target.value)}
                              className="w-full px-4 py-2 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                            >
                              <option value="">-- 选择模型 --</option>
                              {models.map((model) => (
                                <option key={model.name} value={model.name}>
                                  {model.name}
                                </option>
                              ))}
                            </select>
                          )}
                        </div>
                      </div>
                      
                      {/* 文件预览 */}
                      <div>
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
                              className="max-h-80 rounded-lg border border-border"
                            />
                            <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-text-secondary shadow-sm">
                              图片
                            </div>
                          </motion.div>
                        )}
                        {previewVideo && (
                          <motion.div
                            className="relative max-h-80 rounded-lg border border-border overflow-hidden"
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
                            <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-text-secondary shadow-sm">
                              视频
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>
                    {/* 上传进度和处理状态 */}
                    {isLoading && (
                      <div className="mt-6 space-y-6">
                        {processingStatus && (
                          <p className="text-sm text-text-secondary animate-pulse flex items-center justify-center">
                            <Clock className="h-4 w-4 mr-2" />
                            {processingStatus}
                          </p>
                        )}
                        
                        {/* 视频帧处理进度 */}
                        {previewVideo && videoFrameProgress.total > 0 && (
                          <div className="space-y-3">
                            <div className="flex justify-between items-center">
                              <span className="text-xs text-text-light">
                                处理帧: {videoFrameProgress.current}/{videoFrameProgress.total}
                              </span>
                              <span className="text-xs text-text-light">
                                {Math.round((videoFrameProgress.current / videoFrameProgress.total) * 100)}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                              <motion.div 
                                className="bg-primary h-2 rounded-full"
                                initial={{ width: '0%' }}
                                animate={{ width: `${(videoFrameProgress.current / videoFrameProgress.total) * 100}%` }}
                                transition={{ duration: 0.3 }}
                              />
                            </div>
                          </div>
                        )}
                        
                        {/* 通用上传进度 */}
                        {uploadProgress > 0 && uploadProgress < 100 && !previewVideo && (
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <motion.div 
                              className="bg-primary h-2 rounded-full"
                              initial={{ width: '0%' }}
                              animate={{ width: `${uploadProgress}%` }}
                              transition={{ duration: 0.3 }}
                            />
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="flex justify-center space-x-4 mt-6">
                      <motion.button
                        whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                        whileTap={{ scale: 0.98 }}
                        onClick={resetForm}
                        className="px-6 py-2 bg-white border border-border text-text-secondary rounded-md flex items-center transition-all hover:shadow-sm"
                      >
                        <RefreshCw className="h-4 w-4 mr-2" />
                        <span className="font-medium">重新选择</span>
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.02, backgroundColor: "var(--primary-dark)" }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleUpload}
                        disabled={isLoading}
                        className="px-6 py-2 bg-primary text-white rounded-md flex items-center transition-all hover:shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            <span className="font-medium">{previewVideo ? '处理中...' : '识别中...'}</span>
                          </>
                        ) : (
                          <>
                            <Search className="h-4 w-4 mr-2" />
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

          {/* 分类结果 - DeepSeek and Element UI style */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="bg-white rounded-lg shadow-sm border border-border p-6 mb-12"
          >
            <div className="relative z-10">
              <motion.div 
                className="flex items-center mb-6"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <motion.button
                  whileHover={{ scale: 1.05, backgroundColor: "var(--info-light)" }}
                  whileTap={{ scale: 0.95 }}
                  onClick={resetForm}
                  className="p-2 rounded-md hover:bg-gray-100 mr-4 transition-all"
                >
                  <ArrowLeft className="h-4 w-4 text-text-secondary" />
                </motion.button>
                <motion.h2 
                  className="text-xl font-semibold text-text-primary"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <BarChart2 className="inline-block h-5 w-5 mr-2 text-primary" />
                  识别结果
                </motion.h2>
              </motion.div>

              {/* 文件预览 */}
              <motion.div 
                className="mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <h3 className="text-base font-medium text-text-primary mb-4 flex items-center">
                  {result.fileType === 'image' ? (
                    <>
                      <ImageIcon className="h-4 w-4 mr-2 text-primary" />
                      上传的图片
                    </>
                  ) : (
                    <>
                      <Video className="h-4 w-4 mr-2 text-primary" />
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
                          className="max-h-72 rounded-lg border border-border"
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
                                boxElement.className = 'bounding-box absolute border-2 border-primary rounded-md';
                                boxElement.style.left = `${x1}px`;
                                boxElement.style.top = `${y1}px`;
                                boxElement.style.width = `${x2 - x1}px`;
                                boxElement.style.height = `${y2 - y1}px`;
                                boxElement.style.zIndex = '10';
                                
                                // 创建标签元素
                                const labelElement = document.createElement('div');
                                labelElement.className = 'absolute -top-6 left-0 bg-primary text-white text-xs px-2 py-1 rounded';
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
                      <div className="relative">
                        <video
                          src={result.generatedVideoUrl || previewVideo || ''}
                          controls
                          className="max-h-72 rounded-lg border border-border"
                        >
                          您的浏览器不支持视频播放。
                        </video>
                      </div>
                    )}
                    <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm px-2 py-1 rounded-full text-xs font-medium text-text-secondary shadow-sm">
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
                className="bg-primary-light rounded-lg p-6 border border-primary/20"
              >
                <motion.h3 
                  className="text-base font-medium text-text-primary mb-4 flex items-center"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.7 }}
                >
                  <Sparkles className="h-4 w-4 mr-2 text-primary" />
                  AI 识别结果
                </motion.h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* 角色信息 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.8 }}
                    whileHover={{ y: -4, boxShadow: "0 8px 16px rgba(0, 0, 0, 0.1)" }}
                    className="bg-white rounded-md p-4 shadow-sm border border-border hover:border-primary/30 transition-all"
                  >
                    <h4 className="text-xs font-medium text-text-tertiary mb-2">
                      {result.fileType === 'image' ? '识别角色' : '主要角色'}
                    </h4>
                    <p className="text-lg font-semibold text-text-primary">
                      {result.role || '未知'}
                    </p>
                  </motion.div>

                  {/* 置信度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.9 }}
                    whileHover={{ y: -4, boxShadow: "0 8px 16px rgba(0, 0, 0, 0.1)" }}
                    className="bg-white rounded-md p-4 shadow-sm border border-border hover:border-primary/30 transition-all"
                  >
                    <h4 className="text-xs font-medium text-text-tertiary mb-2">
                      置信度
                    </h4>
                    <div className="flex items-center mb-3">
                      <span className="text-lg font-semibold text-text-primary mr-2">
                        {(result.similarity * 100).toFixed(2)}%
                      </span>
                      <span
                        className={`
                          px-2 py-1 rounded-full text-xs font-medium
                          ${getAccuracyBadgeClass(result.similarity)}
                        `}
                      >
                        {getAccuracyText(result.similarity)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.similarity * 100}%` }}
                        transition={{ duration: 1.2, ease: "easeOut", delay: 1 }}
                        className={`h-2 rounded-full ${result.similarity >= 0.8 ? 'bg-primary' : result.similarity >= 0.5 ? 'bg-warning' : 'bg-danger'}`}
                      />
                    </div>
                  </motion.div>

                  {/* 识别速度 */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 1 }}
                    whileHover={{ y: -4, boxShadow: "0 8px 16px rgba(0, 0, 0, 0.1)" }}
                    className="bg-white rounded-md p-4 shadow-sm border border-border hover:border-primary/30 transition-all"
                  >
                    <h4 className="text-xs font-medium text-text-tertiary mb-2">
                      处理速度
                    </h4>
                    <div className="flex items-center">
                      <Zap className="h-4 w-4 text-primary mr-2" />
                      <span className="text-base font-medium text-text-primary">
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
                    className="mt-8"
                  >
                    <h4 className="text-base font-medium text-text-primary mb-4 flex items-center">
                      <Film className="h-4 w-4 mr-2 text-primary" />
                      视频帧检测结果
                    </h4>
                    <div className="bg-white rounded-lg p-4 shadow-sm border border-border max-h-80 overflow-y-auto">
                      <div className="space-y-3">
                        {result.videoResults.map((frameResult, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: 1.2 + index * 0.1 }}
                            whileHover={{ y: -2, boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)" }}
                            className="flex items-center justify-between p-3 border border-border rounded-md hover:bg-gray-50 transition-all"
                          >
                            <div className="flex items-center">
                              <div className="w-8 h-8 rounded-full bg-primary-light flex items-center justify-center mr-3">
                                <span className="text-xs font-medium text-primary">{frameResult.frame}</span>
                              </div>
                              <div>
                                <p className="font-medium text-text-primary">{frameResult.role}</p>
                                <p className="text-xs text-text-tertiary">时间: {frameResult.timestamp.toFixed(1)}秒</p>
                              </div>
                            </div>
                            <div className="flex items-center">
                              <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                                <div 
                                  className={`h-2 rounded-full ${frameResult.similarity >= 0.8 ? 'bg-primary' : frameResult.similarity >= 0.5 ? 'bg-warning' : 'bg-danger'}`} 
                                  style={{ width: `${(frameResult.similarity * 100).toFixed(0)}%` }}
                                />
                              </div>
                              <span className="text-xs font-medium text-text-primary">
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

              {/* 操作按钮 - DeepSeek and Element UI style */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.1 }}
                className="mt-8 flex justify-center space-x-4"
              >
                <motion.button
                  whileHover={{ scale: 1.02, backgroundColor: "var(--info-light)" }}
                  whileTap={{ scale: 0.98 }}
                  onClick={resetForm}
                  className="px-6 py-3 bg-white text-text-secondary rounded-md font-medium shadow-sm border border-border hover:border-primary/30 transition-all"
                >
                  <RefreshCw className="inline-block h-4 w-4 mr-2" />
                  {result.fileType === 'image' ? '上传另一张' : '上传另一个视频'}
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02, backgroundColor: "var(--primary-dark)" }}
                  whileTap={{ scale: 0.98 }}
                  className="px-6 py-3 bg-primary text-white rounded-md font-medium shadow-sm hover:shadow transition-all"
                >
                  <Search className="inline-block h-4 w-4 mr-2" />
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
