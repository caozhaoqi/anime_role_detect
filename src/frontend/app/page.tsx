"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Moon, Sun, Trash2 } from "lucide-react";
import { Message, Model } from "./types";
import { useHistory } from "./hooks/useHistory";
import axios from 'axios';

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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(false);

  // 使用历史记录Hook
  const { history, loadHistory, addToHistory, clearHistory } = useHistory();

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

  // 移除图片
  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
  };

  // 处理拖拽事件
  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
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

    // 清空输入
    setInputText("");

    if (selectedImage) {
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
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage, selectedModel, addToHistory]);

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
        <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-[9999] border-2 border-dashed border-blue-500 rounded-lg">
          <div className="text-center p-8 bg-white dark:bg-gray-800 rounded-xl shadow-2xl">
            <Upload className="h-16 w-16 mx-auto mb-4 text-blue-500" />
            <h3 className="text-xl font-semibold mb-2">拖拽图片到这里</h3>
            <p className="text-gray-600 dark:text-gray-400">松开鼠标即可上传图片进行识别</p>
          </div>
        </div>
      )}
      
      {/* 顶部导航栏 */}
      <header className={`sticky top-0 z-50 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b`}>
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className={`p-2.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
            title="显示侧边栏"
          >
            <Menu className="h-6 w-6" />
          </button>
          <h1 className="text-2xl font-semibold">动漫角色识别</h1>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
            title={darkMode ? "切换到浅色模式" : "切换到深色模式"}
          >
            {darkMode ? <Sun className="h-6 w-6" /> : <Moon className="h-6 w-6" />}
          </button>
        </div>
      </header>
      
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧边栏 */}
        <aside className={`fixed top-16 left-0 z-40 w-72 h-[calc(100vh-4rem)] ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-r transform transition-transform duration-300 ${showSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          <div className="p-4 md:p-6">
            <h2 className="text-lg font-semibold mb-4">模型选择</h2>
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.name}
                  className={`flex items-center space-x-3 p-2 rounded-lg transition-colors cursor-pointer ${selectedModel === model.name ? (darkMode ? 'bg-blue-900/30 border-blue-700' : 'bg-blue-50 border-blue-200') : (darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100')} border`}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <div className={`w-2 h-2 rounded-full ${model.available ? 'bg-green-500' : 'bg-yellow-500'}`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{model.name}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{model.description}</p>
                  </div>
                  {selectedModel === model.name && (
                    <CheckCircle className="h-4 w-4 text-blue-500" />
                  )}
                </div>
              ))}
            </div>
            
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2">识别设置</h3>
              <div className={`flex items-center justify-between p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors cursor-pointer`}>
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
              <h3 className="text-sm font-medium mb-2">历史记录</h3>
              <button
                onClick={clearHistory}
                className={`w-full flex items-center justify-center space-x-2 p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors`}
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
                <div className={`max-w-4xl mx-auto ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                  <div className="p-4 md:p-6 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                    <h2 className="text-lg md:text-xl font-semibold">动漫角色识别</h2>
                  </div>
                  <div className="p-4 md:p-6 max-h-[calc(100vh-24rem)] overflow-y-auto space-y-4 md:space-y-6">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`flex-shrink-0 mr-2 ml-2 ${message.role === "user" ? "order-2" : "order-1"}`}
                        >
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${message.role === "user" ? 'bg-blue-500 text-white' : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')}`}>
                            {message.role === "user" ? (
                              <User className="h-4 w-4" />
                            ) : (
                              <Bot className="h-4 w-4" />
                            )}
                          </div>
                        </div>
                        <div
                          className={`max-w-[80%] ${message.role === "user" ? "order-1" : "order-2"}`}
                        >
                          <div
                            className={`rounded-lg p-4 ${message.role === "user" ? 'bg-blue-500 text-white' : (darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900')}`}
                          >
                            {message.image && (
                              <div className="mb-3 rounded overflow-hidden">
                                <img
                                  src={message.image}
                                  alt="User uploaded image"
                                  className="w-full h-auto object-cover"
                                />
                              </div>
                            )}
                            <p className="whitespace-pre-wrap break-words">{message.content}</p>

                            {message.classification && (
                              <div className="mt-3 space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <h4 className="font-semibold text-xs">识别结果</h4>
                                </div>
                                <div className={`grid grid-cols-2 gap-2 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                                  <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">角色</p>
                                    <p className="text-sm font-medium">{message.classification.role}</p>
                                  </div>
                                  <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">相似度</p>
                                    <p className="text-sm font-medium">{(message.classification.similarity * 100).toFixed(1)}%</p>
                                  </div>
                                  <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded col-span-2`}>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">置信度</p>
                                    <div className="flex items-center space-x-2">
                                      <p className="text-sm font-medium">
                                        {message.classification.confidence === "high" ? "高" : message.classification.confidence === "medium" ? "中" : "低"}
                                      </p>
                                      <div
                                        className={`w-2 h-2 rounded-full ${message.classification.confidence === "high" ? "bg-green-500" : message.classification.confidence === "medium" ? "bg-yellow-500" : "bg-red-500"}`}
                                      />
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {message.multi_roles && message.multi_roles.length > 0 && (
                              <div className="mt-3 space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <h4 className="font-semibold text-xs">多角色识别结果</h4>
                                </div>
                                <div className="space-y-2">
                                  {message.multi_roles.map((role, index) => (
                                    <div key={index} className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                      <div className="flex justify-between items-center">
                                        <p className="text-sm font-medium">{role.role}</p>
                                        <div className="flex items-center space-x-2">
                                          <p className="text-xs">{(role.similarity * 100).toFixed(1)}%</p>
                                          <div
                                            className={`w-1.5 h-1.5 rounded-full ${role.similarity >= 0.8 ? "bg-green-500" : role.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                                          />
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {message.attributes && message.attributes.length > 0 && (
                              <div className="mt-3 space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <h4 className="font-semibold text-xs">角色属性</h4>
                                </div>
                                <div className="flex flex-wrap gap-1">
                                  {message.attributes.map((attr, index) => (
                                    <span
                                      key={index}
                                      className={`px-2 py-1 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-xs font-medium`}
                                    >
                                      {attr.tag}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {message.text_detections && message.text_detections.length > 0 && (
                              <div className="mt-3 space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <h4 className="font-semibold text-xs">文本检测</h4>
                                </div>
                                <div className="space-y-1">
                                  {message.text_detections.map((text, index) => (
                                    <div key={index} className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                      <p className="text-sm font-medium">{text.text}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {message.ai_predicted_role && (
                              <div className="mt-3 space-y-2">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-green-500" />
                                  <h4 className="font-semibold text-xs">AI预测角色</h4>
                                </div>
                                <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                  <p className="text-sm font-medium">{message.ai_predicted_role}</p>
                                </div>
                              </div>
                            )}

                            {message.thoughts && !message.isThinkingFinished && (
                              <div className="mt-3 space-y-1">
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <h4 className="font-semibold text-xs">识别过程</h4>
                                </div>
                                <div className="space-y-1">
                                  {message.thoughts.map((thought, index) => (
                                    <div key={index} className="flex items-center space-x-2">
                                      <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                                      <p className="text-xs">{thought}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            <div className="flex items-center justify-between mt-3 text-xs text-gray-400 dark:text-gray-500">
                              <span suppressHydrationWarning={true}>{new Date(message.timestamp).toLocaleTimeString()}</span>
                              <div className="flex items-center space-x-2">
                                <button
                                  onClick={() => handleCopyMessage(message.content)}
                                  className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                                  title="复制内容"
                                >
                                  <Copy className="h-3 w-3" />
                                </button>
                                <button
                                  onClick={() => handleDownloadMessage(message.content, message.role)}
                                  className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                                  title="下载内容"
                                >
                                  <Download className="h-3 w-3" />
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                  <div className="p-4 md:p-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                    <div className="flex flex-col md:flex-row items-stretch md:items-center space-y-3 md:space-y-0 md:space-x-4">
                      {/* 直接的文件输入元素 */}
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageSelect}
                        className={`px-3 py-2 md:px-4 md:py-3 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm`}
                      />
                      <div className="flex-1 relative">
                        <input
                          type="text"
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          onKeyPress={handleKeyPress}
                          placeholder="输入消息或上传图片..."
                          className={`w-full px-4 py-2 md:px-5 md:py-3 pr-12 md:pr-16 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm`}
                          disabled={isProcessing}
                        />
                        <button
                          onClick={() => setInputText("")}
                          className={`absolute right-8 top-1/2 transform -translate-y-1/2 p-1 ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} rounded transition-colors`}
                          title="清空输入"
                          disabled={!inputText.trim() || isProcessing}
                        >
                          <X className={`h-4 w-4 ${inputText.trim() && !isProcessing ? '' : 'opacity-50 cursor-not-allowed'}`} />
                        </button>

                      </div>
                      <button
                        onClick={handleSend}
                        disabled={(!inputText.trim() && !selectedImage) || isProcessing}
                        className={`bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 md:px-6 md:py-3 rounded-lg font-medium transition-colors flex items-center justify-center space-x-1 md:space-x-2 ${(!inputText.trim() && !selectedImage) || isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
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
                      <div className={`mt-3 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-3 flex items-center space-x-3`}>
                        <div className="w-16 h-16 rounded overflow-hidden">
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
                          className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors`}
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
            {/* 右侧边栏 */}
            <div className="w-0 lg:w-80 border-l border-gray-200 dark:border-gray-700 overflow-y-auto">
              <div className="p-6">
                <h3 className="text-lg font-semibold mb-4">识别结果</h3>
                {(() => {
                  const lastMessage = messages[messages.length - 1];
                  if (!lastMessage?.classification) return null;
                  return (
                    <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">角色</p>
                          <p className="text-lg font-semibold">{lastMessage.classification.role}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">相似度</p>
                          <p className="text-sm font-medium">{(lastMessage.classification.similarity * 100).toFixed(1)}%</p>
                        </div>
                        {lastMessage.classification.confidence && (
                          <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">置信度</p>
                            <p className="text-sm font-medium">{lastMessage.classification.confidence}</p>
                          </div>
                        )}
                        {lastMessage.attributes && lastMessage.attributes.length > 0 && (
                          <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">角色属性</p>
                            <div className="flex flex-wrap gap-2 mt-2">
                              {lastMessage.attributes.map((attr, index) => (
                                <span key={index} className={`px-2 py-1 rounded-full text-xs ${darkMode ? 'bg-gray-600' : 'bg-gray-200'}`}>
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
      
      {/* 页脚 */}
      <footer className={`py-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>动漫角色识别助手 © zhaoqi.cao arona 2026</p>
          <p className="mt-1">基于深度学习的动漫角色识别系统</p>
        </div>
      </footer>
    </div>
  );
}
