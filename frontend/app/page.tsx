"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, Image as ImageIcon, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Search, Settings, HelpCircle, Moon, Sun, Zap, Layers } from "lucide-react";
import { Message, Model } from "./types";
import { useHistory } from "./hooks/useHistory";

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
  const [models, setModels] = useState<Model[]>([{ name: "default", path: "", files: [] }]);
  const [inputText, setInputText] = useState<string>("");
  const [showUploadOptions, setShowUploadOptions] = useState(false);
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMountedRef = useRef(false);

  // 使用历史记录Hook
  const { history, loadHistory, addToHistory, clearHistory } = useHistory();

  useEffect(() => {
    isMountedRef.current = true;
    loadModels();
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 快捷键功能
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+H 打开/关闭历史记录
      if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
        e.preventDefault();
        setShowHistory(!showHistory);
      }
      
      // Esc 关闭上传选项
      if (e.key === 'Escape') {
        setShowUploadOptions(false);
        setIsDragging(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showHistory]);

  const loadModels = async () => {
    try {
      // 由于后端API没有提供获取模型列表的端点，我们直接在前端硬编码一个模型列表
      const modelList = [
        { name: "default", path: "", files: [] },
        { name: "efficientnet-b3", path: "", files: [] },
        { name: "mobilenet-v2", path: "", files: [] }
      ];
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0].name);
      }
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  };

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    }
  }, []);

  const classifyImage = async (imageData: string): Promise<any> => {
    // 将base64编码的图像数据转换为Blob对象
    const response = await fetch(imageData);
    const blob = await response.blob();
    const file = new File([blob], "uploaded_image.jpg", { type: "image/jpeg" });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("use_model", "true");

    const apiResponse = await fetch("http://localhost:5002/api/classify", {
      method: "POST",
      body: formData,
    });

    if (!apiResponse.ok) {
      const errorData = await apiResponse.json().catch(() => ({}));
      throw new Error(errorData.error || "Classification failed");
    }

    return await apiResponse.json();
  };

  const handleSend = useCallback(async () => {
    if ((!inputText.trim() && !selectedImage) || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputText,
      image: imagePreview || undefined,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    const currentImage = selectedImage;
    const currentImagePreview = imagePreview;
    removeImage();

    if (currentImage && currentImagePreview) {
      setIsProcessing(true);

      const processingMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        thoughts: ["正在分析图片特征...", "提取角色关键信息...", "匹配数据库中的角色..."],
        isThinkingFinished: false,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, processingMessage]);

      try {
        const result = await classifyImage(currentImagePreview);

        const assistantMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content: `识别完成！识别结果：${result.role || "未知角色"}，相似度：${(result.similarity * 100).toFixed(1)}%`,
          classification: {
            role: result.role || "未知角色",
            similarity: result.similarity || 0,
            confidence: (result.similarity || 0) >= 0.8 ? "high" : (result.similarity || 0) >= 0.5 ? "medium" : "low",
          },
          thoughts: ["正在分析图片特征...", "提取角色关键信息...", "匹配数据库中的角色...", "识别完成！"],
          isThinkingFinished: true,
          timestamp: Date.now(),
        };

        setMessages((prev) => prev.filter((m) => m.id !== processingMessage.id).concat(assistantMessage));
        addToHistory(assistantMessage);
      } catch (error) {
        console.error("Classification error:", error);
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content: `抱歉，识别过程中出现错误：${error instanceof Error ? error.message : "未知错误"}，请重试。`,
          timestamp: Date.now(),
        };
        setMessages((prev) => prev.filter((m) => m.id !== processingMessage.id).concat(errorMessage));
      } finally {
        setIsProcessing(false);
      }
    } else if (inputText.trim()) {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "请上传动漫角色图片，我会帮你识别角色名称。",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage, classifyImage, addToHistory]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const getConfidenceText = useCallback((confidence: string) => {
    switch (confidence) {
      case "high":
        return "高置信度";
      case "medium":
        return "中等置信度";
      case "low":
        return "低置信度";
      default:
        return "未知";
    }
  }, []);

  const handleCopyMessage = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopySuccess("复制成功！");
      setTimeout(() => setCopySuccess(null), 3000);
    } catch (err) {
      console.error("复制失败:", err);
    }
  }, []);

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
      className="flex flex-col h-screen font-sans overflow-hidden bg-[#0a0a0a] text-[#ededed]"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* 拖拽上传覆盖层 */}
      {isDragging && (
        <div className="fixed inset-0 bg-[#0a0a0a]/80 backdrop-blur-sm flex items-center justify-center z-50 border-2 border-dashed border-[#6366f1] rounded-lg animate-pulse-glow">
          <div className="text-center p-8 glass rounded-xl">
            <Upload className="h-16 w-16 mx-auto mb-4 text-[#6366f1] animate-bounce" />
            <h3 className="text-xl font-semibold mb-2 text-[#fafafa]">拖拽图片到这里</h3>
            <p className="text-[#a1a1aa]">松开鼠标即可上传图片进行识别</p>
          </div>
        </div>
      )}

      {/* 移动端顶部导航栏 */}
      <div className="md:hidden h-14 border-b border-[#27272a] flex items-center justify-between px-4 flex-shrink-0 glass">
        <div className="flex items-center gap-2.5">
          <button 
            className="p-1.5 rounded-lg hover:bg-[#27272a] transition-colors"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <Menu size={18} className="text-[#a1a1aa]" />
          </button>
          <h2 className="text-base font-semibold gradient-text">动漫角色识别</h2>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-1.5 rounded-lg hover:bg-[#27272a] transition-colors">
            <Moon className="h-4 w-4 text-[#a1a1aa]" />
          </button>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧边栏（仅在中等及以上屏幕显示） */}
        <div className={`fixed md:relative top-14 left-0 z-50 flex flex-col items-center lg:items-start w-16 lg:w-60 h-[calc(100%-3.5rem)] glass border-r border-[#27272a] p-3 transition-all duration-300 transform ${showSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          {/* Logo */}
          <div className="flex items-center gap-2.5 mb-6">
            <div className="p-2.5 gradient-bg rounded-lg shadow-lg animate-pulse-glow">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-lg font-bold gradient-text hidden lg:block">动漫角色识别</h1>
          </div>

          {/* 导航菜单 */}
          <nav className="flex-1 w-full">
            <ul className="space-y-1.5">
              <li>
                <button className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg bg-[#6366f1]/10 text-[#6366f1] font-medium hover:bg-[#6366f1]/20 transition-all duration-300">
                  <Search className="h-4 w-4" />
                  <span className="hidden lg:block">识别</span>
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setShowHistory(!showHistory)}
                  className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg hover:bg-[#27272a] text-[#a1a1aa] transition-all duration-300"
                >
                  <Layers className="h-4 w-4" />
                  <span className="hidden lg:block">历史记录</span>
                  {history.length > 0 && (
                    <span className="ml-auto bg-[#6366f1] text-white text-xs px-2 py-0.5 rounded-full">
                      {history.length}
                    </span>
                  )}
                </button>
              </li>
              <li>
                <button className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg hover:bg-[#27272a] text-[#a1a1aa] transition-all duration-300">
                  <Settings className="h-4 w-4" />
                  <span className="hidden lg:block">设置</span>
                </button>
              </li>
              <li>
                <button className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg hover:bg-[#27272a] text-[#a1a1aa] transition-all duration-300">
                  <HelpCircle className="h-4 w-4" />
                  <span className="hidden lg:block">帮助</span>
                </button>
              </li>
            </ul>
          </nav>

          {/* 底部设置 */}
          <div className="w-full mt-auto">
            <button className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-lg hover:bg-[#27272a] text-[#a1a1aa] transition-all duration-300">
              <Moon className="h-4 w-4" />
              <span className="hidden lg:block">深色模式</span>
            </button>
          </div>
        </div>

        {/* 主内容区域 */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          {/* 顶部导航栏（仅在中等及以上屏幕显示） */}
          <div className="hidden md:flex h-14 border-b border-[#27272a] items-center justify-between px-6 flex-shrink-0 glass">
            <h2 className="text-base font-semibold gradient-text">{showHistory ? "历史记录" : "动漫角色识别"}</h2>
            <div className="flex items-center gap-3">
              {!showHistory && (
                <div className="relative">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="appearance-none pl-3 pr-8 py-1.5 border border-[#3f3f46] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6366f1] focus:border-transparent bg-[#18181b] text-[#fafafa] text-sm transition-all duration-300"
                  >
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name === "default" ? "默认模型" : model.name}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-[#71717a]">
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              )}
              {showHistory && (
                <button 
                  onClick={clearHistory}
                  className="px-3 py-1.5 bg-[#ef4444]/10 text-[#ef4444] rounded-lg text-sm hover:bg-[#ef4444]/20 transition-all duration-300"
                >
                  清空历史
                </button>
              )}
            </div>
          </div>

          {/* 移动端模型选择 */}
          <div className="md:hidden px-4 py-2.5 border-b border-[#27272a] glass">
            {!showHistory && (
              <div className="relative">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full appearance-none pl-3 pr-8 py-1.5 border border-[#3f3f46] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6366f1] focus:border-transparent bg-[#18181b] text-[#fafafa] text-sm transition-all duration-300"
                >
                  {models.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name === "default" ? "默认模型" : model.name}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-[#71717a]">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            )}
            {showHistory && (
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-[#fafafa]">历史记录</h3>
                <button 
                  onClick={clearHistory}
                  className="px-3 py-1 bg-[#ef4444]/10 text-[#ef4444] rounded-lg text-xs hover:bg-[#ef4444]/20 transition-all duration-300"
                >
                  清空
                </button>
              </div>
            )}
          </div>

          {/* 消息列表或历史记录 */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth">
            {copySuccess && (
              <div className="fixed top-20 right-4 left-4 md:left-auto md:right-4 bg-[#10b981] text-white px-6 py-3 rounded-xl shadow-lg animate-slide-up z-50">
                {copySuccess}
              </div>
            )}
            
            {/* 历史记录显示 */}
            {showHistory ? (
              <div className="max-w-3xl mx-auto space-y-6 pb-12">
                {history.length === 0 ? (
                  <div className="glass p-8 rounded-xl text-center">
                    <Layers className="h-12 w-12 mx-auto mb-4 text-[#71717a]" />
                    <h3 className="text-lg font-semibold text-[#fafafa] mb-2">暂无历史记录</h3>
                    <p className="text-[#a1a1aa]">上传图片进行识别后，结果将显示在这里</p>
                  </div>
                ) : (
                  history.map((record, idx) => (
                    <div key={idx} className="glass border border-[#27272a] rounded-xl p-4 shadow-lg animate-slide-up">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs text-[#71717a]">
                          {new Date(record.timestamp).toLocaleString()}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full border ${record.classification?.confidence === "high" ? "bg-[#10b981]/10 text-[#10b981] border-[#10b981]/30" : record.classification?.confidence === "medium" ? "bg-[#f59e0b]/10 text-[#f59e0b] border-[#f59e0b]/30" : "bg-[#ef4444]/10 text-[#ef4444] border-[#ef4444]/30"}`}>
                          {record.classification && getConfidenceText(record.classification.confidence)}
                        </span>
                      </div>
                      {record.classification && (
                        <div className="mb-3">
                          <div className="text-lg font-bold text-[#fafafa] mb-1">{record.classification.role}</div>
                          <div className="text-sm text-[#a1a1aa] mb-2">相似度: {(record.classification.similarity * 100).toFixed(1)}%</div>
                          <div className="progress-bar h-2 mb-3">
                            <div
                              className="progress-bar-fill h-2"
                              style={{ width: `${record.classification.similarity * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={() => {
                            setShowHistory(false);
                            // 可以选择将历史记录重新添加到消息列表中
                          }}
                          className="px-3 py-1.5 bg-[#6366f1]/10 text-[#6366f1] rounded-lg text-xs hover:bg-[#6366f1]/20 transition-all duration-300"
                        >
                          查看详情
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : (
              /* 消息列表显示 */
              <div className="max-w-3xl mx-auto space-y-6 pb-12">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"} animate-slide-up`}>
                    {msg.role === "assistant" && (
                      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center shadow gradient-bg animate-float">
                        <Bot size={16} className="text-white" />
                      </div>
                    )}

                    <div className={`max-w-[80%] sm:max-w-[75%] md:max-w-[70%] rounded-xl px-4 py-3 text-sm leading-6 shadow message-bubble transition-all duration-300 ${msg.role === "user" ? "gradient-bg text-white" : "glass border border-[#27272a] text-[#fafafa]"}`}>
                      {msg.role === "assistant" ? (
                        <div className="flex flex-col gap-3">
                          {/* 思考过程展示 */}
                          {msg.thoughts && msg.thoughts.length > 0 && (
                    <div className="mb-4">
                      <div className="flex items-center gap-2 text-xs px-4 py-2 rounded-full transition-all cursor-pointer w-fit select-none border border-[#6366f1]/30 bg-[#6366f1]/10 text-[#6366f1] shadow-sm">
                        <div className="relative">
                          <Sparkles size={14} className="text-[#6366f1]" />
                          {!msg.isThinkingFinished && (
                            <span className="absolute -top-1 -right-1 flex h-2 w-2">
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#6366f1] opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#6366f1]"></span>
                            </span>
                          )}
                        </div>
                        <span className="font-medium whitespace-nowrap">
                          {msg.isThinkingFinished ? "思考完成" : "正在思考..."}
                        </span>
                      </div>
                      <div className="relative pl-4 border-l-2 border-[#3f3f46] py-1">
                        <div className="text-sm text-[#a1a1aa] leading-relaxed font-serif italic whitespace-pre-wrap">
                          {msg.thoughts.join("\n")}
                          {!msg.isThinkingFinished && (
                            <span className="inline-flex items-center gap-1 ml-1">
                              <span className="w-1.5 h-1.5 bg-[#6366f1] rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
                              <span className="w-1.5 h-1.5 bg-[#6366f1] rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                              <span className="w-1.5 h-1.5 bg-[#6366f1] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                          {/* 回复正文 */}
                          {msg.content && (
                            <div className="prose prose-sm max-w-none animate-fade-in">
                              <p className="whitespace-pre-wrap">{msg.content}</p>
                            </div>
                          )}

                          {/* 识别结果 */}
                          {msg.classification && (
                            <div className="mt-4 pt-3 border-t border-[#27272a]">
                              <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                                <span className="text-xs font-medium text-[#a1a1aa]">识别结果</span>
                                <span className={`text-xs px-3 py-1.5 rounded-full border ${msg.classification.confidence === "high" ? "bg-[#10b981]/10 text-[#10b981] border-[#10b981]/30" : msg.classification.confidence === "medium" ? "bg-[#f59e0b]/10 text-[#f59e0b] border-[#f59e0b]/30" : "bg-[#ef4444]/10 text-[#ef4444] border-[#ef4444]/30"}`}>
                                  {getConfidenceText(msg.classification.confidence)}
                                </span>
                              </div>
                              <div className="glass p-4 rounded-xl shadow-inner">
                                <div className="flex flex-col sm:flex-row items-center space-x-3">
                                  <div className="flex-1 w-full sm:w-auto">
                                    <div className="text-sm font-bold text-[#fafafa]">{msg.classification.role}</div>
                                    <div className="text-xs text-[#a1a1aa] mt-1">相似度: {(msg.classification.similarity * 100).toFixed(1)}%</div>
                                    <div className="mt-2">
                                      <div className="progress-bar h-2">
                                        <div
                                          className="progress-bar-fill h-2"
                                          style={{ width: `${msg.classification.similarity * 100}%` }}
                                        ></div>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="w-12 h-12 rounded-full bg-[#6366f1]/10 flex items-center justify-center shadow-md mt-3 sm:mt-0">
                                    <CheckCircle
                                      className={`w-6 h-6 ${msg.classification.confidence === "high" ? "text-[#10b981]" : msg.classification.confidence === "medium" ? "text-[#f59e0b]" : "text-[#ef4444]"}`}
                                    />
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="flex flex-col gap-3">
                          {msg.image && (
                            <div className="mb-3">
                              <div className="relative">
                                <img src={msg.image} alt="Uploaded" className="max-w-full h-auto rounded-xl shadow-lg animate-fade-in" />
                                <div className="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded-lg">{selectedImage?.name}</div>
                              </div>
                            </div>
                          )}
                          <div className="whitespace-pre-wrap">{msg.content}</div>
                        </div>
                      )}

                      {/* 复制和下载按钮 */}
                      {msg.content && (
                        <div className="flex justify-end gap-2 mt-2">
                          <button
                            onClick={() => handleCopyMessage(msg.content)}
                            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs transition-all duration-300 ${msg.role === "user" ? "bg-white/10 hover:bg-white/20 text-white" : "bg-[#27272a] hover:bg-[#3f3f46] text-[#a1a1aa]"}`}
                            title="复制消息内容"
                          >
                            <Copy size={12} />
                            <span className="hidden sm:inline">复制</span>
                          </button>
                          <button
                            onClick={() => handleDownloadMessage(msg.content, msg.role)}
                            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs transition-all duration-300 ${msg.role === "user" ? "bg-white/10 hover:bg-white/20 text-white" : "bg-[#27272a] hover:bg-[#3f3f46] text-[#a1a1aa]"}`}
                            title="下载消息内容"
                          >
                            <Download size={12} />
                            <span className="hidden sm:inline">下载</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {msg.role === "user" && (
                      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center shadow bg-[#27272a] text-[#fafafa]">
                        <User size={16} />
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} className="h-4" />
              </div>
            )}
          </div>

          {/* 输入区域 */}
          <div className="border-t border-[#27272a] glass shadow-2xl">
            <div className="px-4 sm:px-6 lg:px-8 py-4">
              {imagePreview && (
                <div className="mb-3 inline-flex items-center space-x-3 px-4 py-3 glass rounded-xl shadow-sm w-full max-w-xs">
                  <img src={imagePreview} alt="Preview" className="w-12 h-12 object-cover rounded-lg shadow" />
                  <span className="text-sm font-medium text-[#a1a1aa] truncate flex-1">{selectedImage?.name}</span>
                  <button onClick={removeImage} className="p-1.5 hover:bg-[#3f3f46] rounded-full transition-colors">
                    <X className="w-4 h-4 text-[#a1a1aa]" />
                  </button>
                </div>
              )}

              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0 relative">
                  <button
                    onClick={() => setShowUploadOptions(!showUploadOptions)}
                    className="p-2.5 rounded-lg hover:bg-[#27272a] transition-all duration-300"
                  >
                    <Upload className="h-5 w-5 text-[#6366f1]" />
                  </button>
                  {showUploadOptions && (
                    <div className="absolute bottom-full left-0 right-0 mb-4 glass rounded-xl shadow-2xl border border-[#27272a] p-2 z-50">
                      <button
                        className="flex items-center px-4 py-3 hover:bg-[#27272a] rounded-lg w-full transition-all duration-300"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <ImageIcon className="h-4 w-4 mr-2 text-[#a1a1aa]" />
                        <span className="text-sm text-[#fafafa]">上传图片</span>
                      </button>
                      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
                    </div>
                  )}
                </div>
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="输入消息或上传图片..."
                    className="w-full px-4 py-2.5 pr-10 glass border border-[#3f3f46] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6366f1] focus:border-transparent text-[#fafafa] placeholder-[#52525b] input-glow transition-all duration-300"
                    disabled={isProcessing}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 p-1.5 hover:bg-[#27272a] rounded transition-colors"
                    disabled={isProcessing}
                  >
                    <ImageIcon className="h-4 w-4 text-[#71717a]" />
                  </button>
                </div>
                <div className="flex-shrink-0">
                  <button
                    onClick={handleSend}
                    disabled={(!inputText.trim() && !selectedImage) || isProcessing}
                    className={`btn-primary px-4 py-2.5 rounded-lg font-medium transition-all duration-300 flex items-center space-x-2 shadow-lg ${(!inputText.trim() && !selectedImage) || isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {isProcessing ? (
                      <>
                        <svg className="loading-spinner h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span className="hidden sm:inline">识别中</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        <span className="hidden sm:inline">发送</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="mt-2 text-xs text-[#71717a] text-center">按 Enter 发送，Shift + Enter 换行</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
