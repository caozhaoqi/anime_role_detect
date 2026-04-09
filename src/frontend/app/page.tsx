"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Moon, Sun, Trash2, RotateCcw, RotateCw, Crop, Check, ArrowLeft } from "lucide-react";
import { Message, Model } from "./types";
import { useHistory } from "./hooks/useHistory";
import { List } from "react-window";
import Cropper from "react-easy-crop";
import axios from 'axios';
import MessageItem from './components/MessageItem';

// 初始化Web Worker
let worker: Worker | null = null;
if (typeof window !== 'undefined') {
  try {
    worker = new Worker(new URL('./workers/processing.worker.ts', import.meta.url));
  } catch (error) {
    console.error('Failed to create worker:', error);
    worker = null;
  }
}



export default function AnimeRoleDetect() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
      timestamp: Date.now(),
    },
  ]);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // 批量上传状态
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("default");
  const [useMultiRole, setUseMultiRole] = useState<boolean>(false);
  const [models, setModels] = useState<Model[]>([
    { name: "default", path: "", files: [], available: true, description: "默认分类模型" },
    { name: "mobilenet_v2", path: "models/incremental", files: [], available: true, description: "MobileNetV2模型 (准确率: 81.13%)" },
    { name: "efficientnet_b0", path: "models/incremental_efficientnet_b0", files: [], available: true, description: "EfficientNet-B0模型 (准确率: 64.15%)" },
    { name: "resnet50", path: "models/incremental_resnet50", files: [], available: true, description: "ResNet50模型 (准确率: 52.83%)" },
  ]);
  const [inputText, setInputText] = useState<string>("");
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false); // 默认隐藏侧边栏
  const [darkMode, setDarkMode] = useState(false);
  
  // 图片编辑状态
  const [isEditing, setIsEditing] = useState(false);
  const [imageToEdit, setImageToEdit] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [cropSize, setCropSize] = useState({ width: 300, height: 300 });
  
  // 模型比较状态
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonResults, setComparisonResults] = useState<Map<string, any>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(false);

  // 使用历史记录Hook
  const { history, filteredHistory, filters, loadHistory, addToHistory, clearHistory, applyFilters } = useHistory();

  // 组件挂载时执行
  useEffect(() => {
    isMountedRef.current = true;
    // 立即执行loadModels函数
    (async () => {
      await loadModels();
    })();

    // 加载历史记录
    loadHistory();

    // 监听键盘事件
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowSidebar(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      isMountedRef.current = false;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [loadHistory]);

  // 加载模型列表
  const loadModels = async () => {
    try {
      console.log('开始加载模型');
      const response = await fetch('/api/models');
      if (!response.ok) {
        throw new Error('加载模型失败');
      }
      const data = await response.json();
      if (isMountedRef.current) {
        setModels(data.models);
      }
    } catch (error) {
      console.error('加载模型失败:', error);
    }
  };

  // 处理图片选择
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        if (isMountedRef.current) {
          setImagePreview(reader.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // 处理批量图片选择
  const handleBatchImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
      setSelectedImages(imageFiles);
      
      const previews: string[] = [];
      imageFiles.forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          if (isMountedRef.current) {
            previews.push(reader.result as string);
            if (previews.length === imageFiles.length) {
              setImagePreviews(previews);
            }
          }
        };
        reader.readAsDataURL(file);
      });
    }
  };

  // 移除单张批量上传的图片
  const removeBatchImage = (index: number) => {
    const newImages = selectedImages.filter((_, i) => i !== index);
    const newPreviews = imagePreviews.filter((_, i) => i !== index);
    setSelectedImages(newImages);
    setImagePreviews(newPreviews);
  };

  // 清空批量上传的图片
  const clearBatchImages = () => {
    setSelectedImages([]);
    setImagePreviews([]);
  };

  // 处理模型比较
  const handleModelComparison = async () => {
    if (!selectedImage || isComparing) {
      return;
    }

    setIsComparing(true);
    
    // 使用Web Worker处理模型比较
    if (worker) {
      worker.postMessage({
        type: 'processModelComparison',
        data: {
          image: selectedImage,
          models: ['mobilenet_v2', 'efficientnet_b0', 'resnet50']
        }
      });

      worker.onmessage = async (event) => {
        if (event.data.success) {
          const results = new Map<string, any>();
          const modelsToCompare = ['mobilenet_v2', 'efficientnet_b0', 'resnet50'];

          for (const modelName of modelsToCompare) {
            try {
              // 构建FormData
              const formData = new FormData();
              formData.append('file', selectedImage);
              formData.append('use_model', 'true');
              formData.append('use_attributes', 'true');
              formData.append('model_name', modelName);
              formData.append('cache_bypass', Date.now().toString());

              // 发送请求到后端API
              const response = await axios.post('/api/classify', formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
              });

              results.set(modelName, response.data);
            } catch (error) {
              console.error(`模型 ${modelName} 比较失败:`, error);
              results.set(modelName, { error: '比较失败' });
            }
          }

          setComparisonResults(results);
        }
        setIsComparing(false);
      };
    } else {
      // 降级方案：直接在主线程处理
      const results = new Map<string, any>();
      const modelsToCompare = ['mobilenet_v2', 'efficientnet_b0', 'resnet50'];

      for (const modelName of modelsToCompare) {
        try {
          // 构建FormData
          const formData = new FormData();
          formData.append('file', selectedImage);
          formData.append('use_model', 'true');
          formData.append('use_attributes', 'true');
          formData.append('model_name', modelName);
          formData.append('cache_bypass', Date.now().toString());

          // 发送请求到后端API
          const response = await axios.post('/api/classify', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          results.set(modelName, response.data);
        } catch (error) {
          console.error(`模型 ${modelName} 比较失败:`, error);
          results.set(modelName, { error: '比较失败' });
        }
      }

      setComparisonResults(results);
      setIsComparing(false);
    }
  };

  // 处理图片编辑
  const handleEditImage = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setImageToEdit(reader.result as string);
      setIsEditing(true);
    };
    reader.readAsDataURL(file);
  };

  // 应用图片编辑
  const handleApplyEdit = () => {
    setIsEditing(false);
    setImagePreview(imageToEdit);
  };

  // 取消图片编辑
  const handleCancelEdit = () => {
    setIsEditing(false);
    setImageToEdit(null);
  };

  // 处理图片旋转
  const handleRotate = (direction: 'left' | 'right') => {
    setRotation(prev => direction === 'left' ? (prev - 90) % 360 : (prev + 90) % 360);
  };

  // 移除图片
  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
  };

  // 处理拖拽事件
  const [dragCounter, setDragCounter] = useState(0);

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragCounter(prev => prev + 1);
    if (dragCounter === 0) {
      setIsDragging(true);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragCounter(prev => prev - 1);
    if (dragCounter === 1) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragCounter(0);
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        if (isMountedRef.current) {
          setImagePreview(reader.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // 处理发送消息
  const handleSend = useCallback(async () => {
    if ((!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing) {
      return;
    }

    // 清空输入
    setInputText("");

    // 处理批量上传
    if (selectedImages.length > 0) {
      // 开始处理
      setIsProcessing(true);

      // 使用Web Worker处理批量上传
      if (worker) {
        worker.postMessage({
          type: 'processBatchUpload',
          data: {
            images: selectedImages,
            imagePreviews: imagePreviews,
            selectedModel: selectedModel
          }
        });

        worker.onmessage = async (event) => {
          if (event.data.success) {
            // 依次处理每张图片
            for (let i = 0; i < selectedImages.length; i++) {
              const image = selectedImages[i];
              const imagePreview = imagePreviews[i];

              // 创建用户消息
              const userMessage: Message = {
                id: Date.now().toString(),
                role: "user",
                content: `上传图片 ${i + 1}/${selectedImages.length}`,
                image: imagePreview || undefined,
                timestamp: Date.now(),
              };

              // 添加用户消息到消息列表
              setMessages(prev => [...prev, userMessage]);

              // 创建处理中消息
              const processingMessage: Message = {
                id: `processing_${Date.now()}`,
                role: "assistant",
                content: `正在识别图片 ${i + 1}/${selectedImages.length}...`,
                isThinking: true,
                thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色..."],
                isThinkingFinished: false,
                timestamp: Date.now(),
              };

              setMessages(prev => [...prev, processingMessage]);

              try {
                // 构建FormData
                const formData = new FormData();
                formData.append('file', image);
                formData.append('use_model', 'true');
                formData.append('use_attributes', 'true');
                formData.append('model_name', selectedModel);
                formData.append('cache_bypass', Date.now().toString());

                // 发送请求到后端API
                console.log(`开始发送请求到API，处理图片 ${i + 1}`);
                const response = await axios.post('/api/classify', formData, {
                  headers: {
                    'Content-Type': 'multipart/form-data',
                  },
                });

                console.log('API响应:', response.data);

                const data = response.data;

                // 构建助手消息
                const assistantMessage: Message = {
                  id: Date.now().toString(),
                  role: "assistant",
                  content: `图片 ${i + 1} 识别完成！`,
                  classification: {
                    role: data.ai_predicted_role || data.predicted_role || "未知角色",
                    similarity: data.similarity || 0,
                    confidence: data.confidence || "medium",
                  },
                  attributes: data.attributes || [],
                  ai_predicted_role: data.ai_predicted_role,
                  thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色...", "识别完成！"],
                  isThinkingFinished: true,
                  timestamp: Date.now(),
                };

                // 更新消息列表，移除处理中消息，添加助手消息
                setMessages(prev => {
                  const newMessages = prev.filter(msg => !msg.isThinking);
                  return [...newMessages, assistantMessage];
                });

                // 添加到历史记录
                addToHistory({
                  image: image,
                  result: {
                    role: data.ai_predicted_role || data.predicted_role || "未知角色",
                    similarity: data.similarity || 0,
                    confidence: data.confidence || "medium",
                  },
                  timestamp: Date.now(),
                });

              } catch (error) {
                console.error('API请求失败:', error);

                // 构建错误消息
                const errorMessage: Message = {
                  id: Date.now().toString(),
                  role: "assistant",
                  content: `图片 ${i + 1} 识别失败`,
                  error: "识别过程中出现错误，请重试。",
                  timestamp: Date.now(),
                };

                // 更新消息列表，移除处理中消息，添加错误消息
                setMessages(prev => {
                  const newMessages = prev.filter(msg => !msg.isThinking);
                  return [...newMessages, errorMessage];
                });
              }
            }
          }

          // 结束处理
          setIsProcessing(false);
          // 清空批量上传的图片
          clearBatchImages();
        };
      } else {
        // 降级方案：直接在主线程处理
        // 依次处理每张图片
        for (let i = 0; i < selectedImages.length; i++) {
          const image = selectedImages[i];
          const imagePreview = imagePreviews[i];

          // 创建用户消息
          const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: `上传图片 ${i + 1}/${selectedImages.length}`,
            image: imagePreview || undefined,
            timestamp: Date.now(),
          };

          // 添加用户消息到消息列表
          setMessages(prev => [...prev, userMessage]);

          // 创建处理中消息
          const processingMessage: Message = {
            id: `processing_${Date.now()}`,
            role: "assistant",
            content: `正在识别图片 ${i + 1}/${selectedImages.length}...`,
            isThinking: true,
            thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色..."],
            isThinkingFinished: false,
            timestamp: Date.now(),
          };

          setMessages(prev => [...prev, processingMessage]);

          try {
            // 构建FormData
            const formData = new FormData();
            formData.append('file', image);
            formData.append('use_model', 'true');
            formData.append('use_attributes', 'true');
            formData.append('model_name', selectedModel);
            formData.append('cache_bypass', Date.now().toString());

            // 发送请求到后端API
            console.log(`开始发送请求到API，处理图片 ${i + 1}`);
            const response = await axios.post('/api/classify', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });

            console.log('API响应:', response.data);

            const data = response.data;

            // 构建助手消息
            const assistantMessage: Message = {
              id: Date.now().toString(),
              role: "assistant",
              content: `图片 ${i + 1} 识别完成！`,
              classification: {
                role: data.ai_predicted_role || data.predicted_role || "未知角色",
                similarity: data.similarity || 0,
                confidence: data.confidence || "medium",
              },
              attributes: data.attributes || [],
              ai_predicted_role: data.ai_predicted_role,
              thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色...", "识别完成！"],
              isThinkingFinished: true,
              timestamp: Date.now(),
            };

            // 更新消息列表，移除处理中消息，添加助手消息
            setMessages(prev => {
              const newMessages = prev.filter(msg => !msg.isThinking);
              return [...newMessages, assistantMessage];
            });

            // 添加到历史记录
            addToHistory({
              image: image,
              result: {
                role: data.ai_predicted_role || data.predicted_role || "未知角色",
                similarity: data.similarity || 0,
                confidence: data.confidence || "medium",
              },
              timestamp: Date.now(),
            });

          } catch (error) {
            console.error('API请求失败:', error);

            // 构建错误消息
            const errorMessage: Message = {
              id: Date.now().toString(),
              role: "assistant",
              content: `图片 ${i + 1} 识别失败`,
              error: "识别过程中出现错误，请重试。",
              timestamp: Date.now(),
            };

            // 更新消息列表，移除处理中消息，添加错误消息
            setMessages(prev => {
              const newMessages = prev.filter(msg => !msg.isThinking);
              return [...newMessages, errorMessage];
            });
          }
        }

        // 结束处理
        setIsProcessing(false);
        // 清空批量上传的图片
        clearBatchImages();
      }
    } else if (selectedImage) {
      // 处理单张图片
      // 创建用户消息
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputText.trim(),
        image: selectedImage ? imagePreview || undefined : undefined,
        timestamp: Date.now(),
      };

      // 添加用户消息到消息列表
      setMessages(prev => [...prev, userMessage]);

      // 开始处理
      setIsProcessing(true);

      // 创建处理中消息
      const processingMessage: Message = {
        id: `processing_${Date.now()}`,
        role: "assistant",
        content: "正在识别...",
        isThinking: true,
        thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色..."],
        isThinkingFinished: false,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, processingMessage]);

      try {
        // 构建FormData
        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('use_model', 'true');
        formData.append('use_attributes', 'true');
        formData.append('model_name', selectedModel);
        formData.append('cache_bypass', Date.now().toString());

        // 发送请求到后端API
        console.log('开始发送请求到API');
        const response = await axios.post('/api/classify', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        console.log('API响应:', response.data);

        const data = response.data;

        // 构建助手消息
        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: `识别完成！`,
          classification: {
            role: data.ai_predicted_role || data.predicted_role || "未知角色",
            similarity: data.similarity || 0,
            confidence: data.confidence || "medium",
          },
          attributes: data.attributes || [],
          ai_predicted_role: data.ai_predicted_role,
          thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色...", "识别完成！"],
          isThinkingFinished: true,
          timestamp: Date.now(),
        };

        // 更新消息列表，移除处理中消息，添加助手消息
        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, assistantMessage];
        });

        // 添加到历史记录
        addToHistory({
          image: selectedImage,
          result: {
            role: data.ai_predicted_role || data.predicted_role || "未知角色",
            similarity: data.similarity || 0,
            confidence: data.confidence || "medium",
          },
          timestamp: Date.now(),
        });

      } catch (error) {
        console.error('API请求失败:', error);

        // 构建错误消息
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: "识别失败",
          error: "识别过程中出现错误，请重试。",
          timestamp: Date.now(),
        };

        // 更新消息列表，移除处理中消息，添加错误消息
        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, errorMessage];
        });
      } finally {
        // 结束处理
        setIsProcessing(false);
        // 移除选中的图片
        removeImage();
      }
    }
  }, [inputText, selectedImage, imagePreview, selectedImages, imagePreviews, isProcessing, removeImage, clearBatchImages, selectedModel, addToHistory]);

  // 处理键盘按键
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 复制消息内容
  const handleCopyMessage = useCallback((content: string) => {
    if (!content) return;

    navigator.clipboard.writeText(content)
      .then(() => {
        setCopySuccess("复制成功！");
        setTimeout(() => setCopySuccess(null), 3000);
      })
      .catch(err => {
        console.error("复制失败:", err);
      });
  }, []);

  // 下载消息内容
  const handleDownloadMessage = useCallback((content: string, role: string) => {
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${role === "user" ? "用户" : "助手"}_消息_${new Date().toISOString().slice(0, 19).replace(/[-:]/g, "")}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  return (
    <div 
      className={`flex flex-col h-screen font-sans overflow-hidden ${darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* 拖拽上传覆盖层 */}
      {isDragging && (
        <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-[9999] border-2 border-dashed border-blue-500 rounded-lg animate-pulse">
          <div className="text-center p-8 bg-white dark:bg-gray-800 rounded-xl shadow-2xl transform transition-transform duration-300 hover:scale-105">
            <Upload className="h-16 w-16 mx-auto mb-4 text-blue-500 animate-bounce" />
            <h3 className="text-xl font-semibold mb-2">拖拽图片到这里</h3>
            <p className="text-gray-600 dark:text-gray-400">松开鼠标即可上传图片进行识别</p>
            <div className="mt-4 flex justify-center space-x-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}
      
      {/* 顶部导航栏 */}
      <header className={`sticky top-0 z-50 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b transition-all duration-300`}>
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className={`p-2.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors transform hover:scale-105`}
            title="显示侧边栏"
          >
            <Menu className="h-6 w-6" />
          </button>
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">动漫角色识别</h1>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors transform hover:scale-105`}
            title={darkMode ? "切换到浅色模式" : "切换到深色模式"}
          >
            {darkMode ? <Sun className="h-6 w-6" /> : <Moon className="h-6 w-6" />}
          </button>
        </div>
      </header>
      
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧边栏 */}
        <aside className={`fixed top-16 left-0 z-40 w-72 h-[calc(100vh-4rem)] ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-r transform transition-transform duration-300 ease-in-out ${showSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          <div className="p-4 md:p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-blue-500" />
              <span>模型选择</span>
            </h2>
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.name}
                  className={`flex items-center space-x-3 p-3 rounded-lg transition-all cursor-pointer transform hover:scale-[1.02] ${selectedModel === model.name ? (darkMode ? 'bg-blue-900/30 border-blue-700' : 'bg-blue-50 border-blue-200') : (darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100')} border`}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <div className={`w-2 h-2 rounded-full ${model.available ? 'bg-green-500' : 'bg-yellow-500'}`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{model.name}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{model.description}</p>
                  </div>
                  {selectedModel === model.name && (
                    <CheckCircle className="h-4 w-4 text-blue-500 animate-pulse" />
                  )}
                </div>
              ))}
            </div>
            
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
                <span>识别设置</span>
              </h3>
              <div className={`flex items-center justify-between p-3 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors cursor-pointer transform hover:scale-[1.02]`}>
                <div>
                  <p className="text-sm font-medium">多角色识别</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">同时识别图片中的多个角色</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useMultiRole}
                    onChange={(e) => setUseMultiRole(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className={`w-10 h-5 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-500`}></div>
                </label>
              </div>
            </div>
            
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2 flex items-center space-x-2">
                <span>历史记录</span>
              </h3>
              
              {/* 历史记录筛选 */}
              <div className="space-y-3 mb-4">
                {/* 角色搜索 */}
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">角色搜索</label>
                  <input
                    type="text"
                    value={filters.role}
                    onChange={(e) => applyFilters({ ...filters, role: e.target.value })}
                    placeholder="输入角色名称"
                    className={`w-full px-3 py-2 text-sm rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all`}
                  />
                </div>
                
                {/* 时间范围筛选 */}
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">时间范围</label>
                  <select
                    value={filters.timeRange}
                    onChange={(e) => applyFilters({ ...filters, timeRange: e.target.value as 'all' | 'today' | 'week' | 'month' })}
                    className={`w-full px-3 py-2 text-sm rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all`}
                  >
                    <option value="all">全部时间</option>
                    <option value="today">今天</option>
                    <option value="week">本周</option>
                    <option value="month">本月</option>
                  </select>
                </div>
              </div>
              
              {/* 历史记录列表 */}
              <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {filteredHistory.length === 0 ? (
                  <p className="text-xs text-gray-500 dark:text-gray-400 text-center py-4">
                    暂无历史记录
                  </p>
                ) : (
                  filteredHistory.map((item) => (
                    <div key={item.id} className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} text-xs`}>
                      <div className="flex justify-between items-center">
                        <span>{item.result?.role || item.result?.ai_predicted_role || item.result?.predicted_role || '未知'}</span>
                        <span>{item.result?.similarity ? (item.result.similarity * 100).toFixed(1) : '0.0'}%</span>
                      </div>
                      <div className="text-gray-500 dark:text-gray-400 mt-1">
                        {new Date(item.timestamp).toLocaleString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              <button
                onClick={clearHistory}
                className={`w-full flex items-center justify-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors transform hover:scale-[1.02] mt-3`}
              >
                <Trash2 className="h-4 w-4" />
                <span className="text-sm">清除历史记录</span>
              </button>
            </div>
          </div>
        </aside>
        
        {/* 主内容区 */}
        <main className="flex-1 overflow-y-auto ml-0 md:ml-72">
          <div className="flex h-full">
            {/* 中间内容区 */}
            <div className="flex-1 overflow-y-auto">
              <div className="container mx-auto px-4 md:px-6 py-6 md:py-8">
                <div className={`max-w-4xl mx-auto ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-lg border ${darkMode ? 'border-gray-700' : 'border-gray-200'} transform transition-all duration-300 hover:shadow-xl`}>
                  <div className="p-4 md:p-6 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                    <h2 className="text-lg md:text-xl font-semibold">动漫角色识别</h2>
                  </div>
                  <div className="p-4 md:p-6 max-h-[calc(100vh-24rem)] overflow-hidden">
                    <List
                      rowCount={messages.length}
                      rowHeight={300}
                      overscanCount={5}
                      style={{ width: '100%', height: 600 }}
                      rowProps={{}}
                      rowComponent={({ index, style }) => (
                        <div style={style} className="space-y-4 md:space-y-6">
                          <MessageItem
                            message={messages[index]}
                            darkMode={darkMode}
                            handleCopyMessage={handleCopyMessage}
                            handleDownloadMessage={handleDownloadMessage}
                          />
                        </div>
                      )}
                    />
                    <div ref={messagesEndRef} />
                  </div>
                  <div className="p-4 md:p-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                    <div className="flex flex-col md:flex-row items-stretch md:items-center space-y-3 md:space-y-0 md:space-x-4">
                      {/* 直接的文件输入元素 */}
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageSelect}
                        className={`px-3 py-2 md:px-4 md:py-3 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
                      />
                      
                      {/* 批量上传文件输入元素 */}
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleBatchImageSelect}
                        multiple
                        className={`px-3 py-2 md:px-4 md:py-3 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
                      />
                      <div className="flex-1 relative">
                        <input
                          type="text"
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          onKeyPress={handleKeyPress}
                          placeholder="输入消息或上传图片..."
                          className={`w-full px-4 py-2 md:px-5 md:py-3 pr-12 md:pr-16 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
                          disabled={isProcessing}
                        />
                        <button
                          onClick={() => setInputText("")}
                          className={`absolute right-8 top-1/2 transform -translate-y-1/2 p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                          title="清空输入"
                          disabled={!inputText.trim() || isProcessing}
                        >
                          <X className={`h-4 w-4 ${inputText.trim() && !isProcessing ? '' : 'opacity-50 cursor-not-allowed'}`} />
                        </button>

                      </div>
                      <button
                        onClick={handleSend}
                        disabled={(!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing}
                        className={`bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 md:px-6 md:py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-1 md:space-x-2 ${(!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing ? 'opacity-50 cursor-not-allowed' : 'transform hover:scale-105 hover:shadow-lg'}`}
                      >
                        {isProcessing ? (
                          <>
                            <svg className="h-4 w-4 md:h-5 md:w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span className="text-sm font-medium">识别中</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-4 w-4 md:h-5 md:w-5" />
                            <span className="text-sm font-medium">发送</span>
                          </>
                        )}
                      </button>
                    </div>
                    {selectedImage && imagePreview && (
                      <div className={`mt-3 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-3 flex items-center space-x-3 animate-fade-in`}>
                        <div className="w-16 h-16 rounded-lg overflow-hidden shadow-md">
                          <img
                            src={imagePreview}
                            alt="Selected image"
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium">
                            已选择图片: {selectedImage.name}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            大小: {Math.round(selectedImage.size / 1024)} KB
                          </p>
                        </div>
                        <button
                          onClick={() => handleEditImage(selectedImage!)}
                          className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-blue-900/20' : 'hover:bg-blue-50'} text-blue-500 transition-colors transform hover:scale-110`}
                          title="编辑图片"
                        >
                          <Crop className="h-4 w-4" />
                        </button>
                        <button
                          onClick={handleModelComparison}
                          className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-green-900/20' : 'hover:bg-green-50'} text-green-500 transition-colors transform hover:scale-110`}
                          title="模型比较"
                        >
                          <Sparkles className="h-4 w-4" />
                        </button>
                        <button
                          onClick={removeImage}
                          className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors transform hover:scale-110`}
                          title="移除图片"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                    
                    {/* 批量上传预览 */}
                    {selectedImages.length > 0 && (
                      <div className={`mt-3 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-3 animate-fade-in`}>
                        <div className="flex justify-between items-center mb-3">
                          <p className="text-sm font-medium">
                            已选择 {selectedImages.length} 张图片
                          </p>
                          <button
                            onClick={clearBatchImages}
                            className={`text-xs ${darkMode ? 'text-red-400 hover:text-red-300' : 'text-red-600 hover:text-red-500'} transition-colors`}
                            title="清空所有图片"
                          >
                            清空
                          </button>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                          {selectedImages.map((image, index) => (
                            <div key={index} className="relative group">
                              <div className="w-full aspect-square rounded-lg overflow-hidden shadow-md">
                                <img
                                  src={imagePreviews[index]}
                                  alt={`Selected image ${index + 1}`}
                                  className="w-full h-full object-cover"
                                />
                              </div>
                              <button
                                onClick={() => removeBatchImage(index)}
                                className={`absolute top-1 right-1 p-1 rounded-full bg-red-500 text-white opacity-0 group-hover:opacity-100 transition-opacity transform hover:scale-110`}
                                title="移除图片"
                              >
                                <X className="h-3 w-3" />
                              </button>
                              <div className="text-xs mt-1 truncate">
                                {image.name}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            {/* 右侧边栏 */}
            <div className="w-0 lg:w-80 border-l border-gray-200 dark:border-gray-700 overflow-y-auto">
              <div className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                  <Sparkles className="h-5 w-5 text-blue-500" />
                  <span>识别结果</span>
                </h3>
                {(() => {
                  const lastMessage = messages[messages.length - 1];
                  if (!lastMessage?.classification) return null;
                  return (
                    <div className={`p-5 rounded-xl ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} shadow-md transition-all hover:shadow-lg animate-fade-in`}>
                      <div className="space-y-4">
                        <div className="transform hover:scale-[1.02] transition-transform">
                          <p className="text-sm text-gray-500 dark:text-gray-400">角色</p>
                          <p className="text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">{lastMessage.classification.role}</p>
                        </div>
                        <div className="transform hover:scale-[1.02] transition-transform">
                          <p className="text-sm text-gray-500 dark:text-gray-400">相似度</p>
                          <p className="text-sm font-medium">{(lastMessage.classification.similarity * 100).toFixed(1)}%</p>
                        </div>
                        {lastMessage.classification.confidence && (
                          <div className="transform hover:scale-[1.02] transition-transform">
                            <p className="text-sm text-gray-500 dark:text-gray-400">置信度</p>
                            <p className="text-sm font-medium">{lastMessage.classification.confidence}</p>
                          </div>
                        )}
                        {lastMessage.attributes && lastMessage.attributes.length > 0 && (
                          <div className="transform hover:scale-[1.02] transition-transform">
                            <p className="text-sm text-gray-500 dark:text-gray-400">角色属性</p>
                            <div className="flex flex-wrap gap-2 mt-2">
                              {lastMessage.attributes.map((attr, index) => (
                                <span key={index} className={`px-3 py-1.5 rounded-full text-xs ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} transform hover:scale-105 transition-transform`}>
                                  {attr.tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </main>
      </div>
      
      {/* 图片编辑模态框 */}
      {isEditing && imageToEdit && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold">图片编辑</h3>
              <button
                onClick={handleCancelEdit}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
                title="取消编辑"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="mb-6">
              <div className="aspect-square relative w-full max-w-md mx-auto rounded-lg overflow-hidden">
                <Cropper
                  image={imageToEdit}
                  crop={crop}
                  zoom={zoom}
                  rotation={rotation}
                  cropSize={cropSize}
                  onCropChange={setCrop}
                  onZoomChange={setZoom}
                  onRotationChange={setRotation}
                />
              </div>
            </div>
            
            <div className="flex justify-center space-x-4 mb-6">
              <button
                onClick={() => handleRotate('left')}
                className={`p-3 rounded-full ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="向左旋转"
              >
                <RotateCcw className="h-5 w-5" />
              </button>
              <button
                onClick={() => handleRotate('right')}
                className={`p-3 rounded-full ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="向右旋转"
              >
                <RotateCw className="h-5 w-5" />
              </button>
              <button
                onClick={() => setZoom(prev => Math.max(0.5, prev - 0.1))}
                className={`p-3 rounded-full ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="缩小"
              >
                <span className="text-lg font-bold">-</span>
              </button>
              <button
                onClick={() => setZoom(prev => Math.min(2, prev + 0.1))}
                className={`p-3 rounded-full ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="放大"
              >
                <span className="text-lg font-bold">+</span>
              </button>
            </div>
            
            <div className="flex space-x-4">
              <button
                onClick={handleCancelEdit}
                className={`flex-1 py-3 px-6 border ${darkMode ? 'border-gray-600 text-gray-300' : 'border-gray-300 text-gray-700'} rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors`}
              >
                取消
              </button>
              <button
                onClick={handleApplyEdit}
                className={`flex-1 py-3 px-6 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors transform hover:scale-[1.02]`}
              >
                应用编辑
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 模型比较结果模态框 */}
      {isComparing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold">模型比较</h3>
              <button
                onClick={() => setIsComparing(false)}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
                title="关闭"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="mb-6">
              <div className="w-full max-w-md mx-auto">
                {imagePreview && (
                  <img
                    src={imagePreview}
                    alt="Comparison image"
                    className="w-full h-auto rounded-lg shadow-md"
                  />
                )}
              </div>
            </div>
            
            <div className="space-y-6">
              <h4 className="text-lg font-medium">模型识别结果对比</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Array.from(comparisonResults.entries()).map(([modelName, result]) => (
                  <div key={modelName} className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} shadow-md transition-all hover:shadow-lg`}>
                    <h5 className="font-semibold mb-3">
                      {modelName === 'mobilenet_v2' && 'MobileNetV2'}
                      {modelName === 'efficientnet_b0' && 'EfficientNet-B0'}
                      {modelName === 'resnet50' && 'ResNet50'}
                    </h5>
                    {result.error ? (
                      <p className="text-red-500">{result.error}</p>
                    ) : (
                      <div className="space-y-2">
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">识别角色</p>
                          <p className="font-medium">{result?.ai_predicted_role || result?.predicted_role || '未知'}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">相似度</p>
                          <p className="font-medium">{result?.similarity ? (result.similarity * 100).toFixed(1) : '0.0'}%</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">置信度</p>
                          <p className="font-medium">{result?.confidence === 'high' ? '高' : result?.confidence === 'medium' ? '中' : '低'}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 模型比较结果展示 */}
      {comparisonResults.size > 0 && !isComparing && (
        <div className={`mt-6 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-4 animate-fade-in`}>
          <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2">
            <Sparkles className="h-5 w-5 text-green-500" />
            <span>模型比较结果</span>
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Array.from(comparisonResults.entries()).map(([modelName, result]) => (
              <div key={modelName} className={`p-3 rounded-lg ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} shadow-sm transition-all hover:shadow-md`}>
                <h5 className="font-medium mb-2">
                  {modelName === 'mobilenet_v2' && 'MobileNetV2'}
                  {modelName === 'efficientnet_b0' && 'EfficientNet-B0'}
                  {modelName === 'resnet50' && 'ResNet50'}
                </h5>
                {result.error ? (
                  <p className="text-red-500 text-sm">{result.error}</p>
                ) : (
                  <div className="space-y-1 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">角色:</span> {result.ai_predicted_role || result.predicted_role || '未知'}
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">相似度:</span> {(result.similarity * 100).toFixed(1)}%
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">置信度:</span> {result.confidence === 'high' ? '高' : result.confidence === 'medium' ? '中' : '低'}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* 页脚 */}
      <footer className={`py-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'} transition-all duration-300`}>
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <p className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">动漫角色识别助手 © zhaoqi.cao arona 2026</p>
          <p className="mt-1">基于深度学习的动漫角色识别系统</p>
        </div>
      </footer>
    </div>
  );
}
