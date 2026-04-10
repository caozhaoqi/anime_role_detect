"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Moon, Sun, Trash2, RotateCcw, RotateCw, Crop, Check, ArrowLeft } from "lucide-react";
import { Message } from "./types";
import axios from 'axios';
import MessageItem from './components/MessageItem';

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
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputText, setInputText] = useState<string>("");
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(false);

  // 组件挂载时执行
  useEffect(() => {
    isMountedRef.current = true;

    // 监听键盘事件
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        // 可以添加其他快捷键处理
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      isMountedRef.current = false;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

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
    if ((!inputText.trim() && !selectedImage) || isProcessing) {
      return;
    }

    // 清空输入
    setInputText("");

    if (selectedImage) {
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
        formData.append('model_name', 'default');
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
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage]);

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
        <div className="container mx-auto px-6 py-4 flex items-center justify-center">
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">动漫角色识别</h1>
        </div>
      </header>
      
      <div className="flex-1 flex overflow-hidden">
        {/* 主内容区 */}
        <main className="flex-1 overflow-y-auto">
          <div className="flex-1 overflow-y-auto">
            <div className="container mx-auto px-4 md:px-6 py-6 md:py-8">
              <div className={`w-full ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-lg border ${darkMode ? 'border-gray-700' : 'border-gray-200'} transform transition-all duration-300 hover:shadow-xl`}>
                <div className={`p-4 md:p-6 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <h2 className="text-lg md:text-xl font-semibold">动漫角色识别</h2>
                </div>
                <div className="p-4 md:p-6 max-h-[calc(100vh-24rem)] overflow-y-auto">
                  {messages.map((message, index) => (
                    <div key={message.id} className="space-y-4 md:space-y-6 mb-6">
                      <MessageItem
                        message={message}
                        darkMode={darkMode}
                        handleCopyMessage={handleCopyMessage}
                        handleDownloadMessage={handleDownloadMessage}
                      />
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
                <div className={`p-4 md:p-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <div className="flex flex-col md:flex-row items-stretch md:items-center space-y-3 md:space-y-0 md:space-x-4">
                    {/* 文件输入元素 */}
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      className={`w-full md:w-1/4 px-3 py-2 md:px-4 md:py-3 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
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
                      disabled={(!inputText.trim() && !selectedImage) || isProcessing}
                      className={`w-full md:w-auto min-w-[120px] bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 md:px-6 md:py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-1 md:space-x-2 ${(!inputText.trim() && !selectedImage) || isProcessing ? 'opacity-50 cursor-not-allowed' : 'transform hover:scale-105 hover:shadow-lg'}`}
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
                        onClick={removeImage}
                        className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors transform hover:scale-110`}
                        title="移除图片"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
      

      
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
