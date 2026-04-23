"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Moon, Sun, Trash2, RotateCcw, RotateCw, Crop, Check, ArrowLeft, LogOut } from "lucide-react";
import { Message, AuthState } from "./types";
import axios from 'axios';
import MessageItem from './components/MessageItem';
import Login from './components/Login';

export default function AnimeRoleDetect() {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    accessToken: null,
    refreshToken: null
  });
  const [isLoginLoading, setIsLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  
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
  const [isMacPlatform, setIsMacPlatform] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("default");
  const [useCoreML, setUseCoreML] = useState(false);
  const [useAttributes, setUseAttributes] = useState(true);
  const [multiRole, setMultiRole] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(false);

  // 组件挂载时执行
  useEffect(() => {
    isMountedRef.current = true;

    // 检查登录状态
    const savedAccessToken = localStorage.getItem('accessToken');
    const savedRefreshToken = localStorage.getItem('refreshToken');
    const savedUser = localStorage.getItem('currentUser');
    
    console.log('组件挂载时，检查localStorage:', {
      savedAccessToken: savedAccessToken,
      savedRefreshToken: savedRefreshToken,
      savedUser: savedUser
    });
    
    if (savedAccessToken && savedRefreshToken && savedUser) {
      console.log('从localStorage加载登录状态');
      setAuthState({
        isAuthenticated: true,
        user: JSON.parse(savedUser),
        accessToken: savedAccessToken,
        refreshToken: savedRefreshToken
      });
    } else {
      console.log('localStorage中没有登录信息，用户未登录');
    }

    // 检测平台
    const platform = navigator.platform.toLowerCase();
    const isMac = platform.includes('mac') || platform.includes('darwin');
    setIsMacPlatform(isMac);
    if (isMac) {
      setUseCoreML(true);
    }

    // 获取可用模型列表
    if (savedAccessToken) {
      fetchAvailableModels();
    }

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

  // 获取可用模型列表
  const fetchAvailableModels = async () => {
    try {
      const headers = authState.accessToken ? { Authorization: `Bearer ${authState.accessToken}` } : {};
      const response = await axios.get('/api/models', { headers });
      if (response.data.success) {
        const models = response.data.models || [];
        setAvailableModels(['default', ...models]);
        console.log('可用模型:', models);
      }
    } catch (error) {
      console.error('获取模型列表失败:', error);
    }
  };

  // 处理登录
  const handleLogin = async (username: string, password: string) => {
    setIsLoginLoading(true);
    setLoginError(null);
    
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      console.log('发送登录请求到:', '/api/auth/login');
      const response = await axios.post('/api/auth/login', formData);
      console.log('登录响应:', response.data);
      
      if (response.data.success) {
        const { access_token, refresh_token, username: userName, role } = response.data.data;
        
        console.log('登录成功，获取到token:', access_token);
        
        const user = { username: userName, role };
        
        setAuthState({
          isAuthenticated: true,
          user,
          accessToken: access_token,
          refreshToken: refresh_token
        });
        
        console.log('设置authState后，检查localStorage:', {
          accessToken: access_token,
          refreshToken: refresh_token,
          user: user
        });
        
        localStorage.setItem('accessToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);
        localStorage.setItem('currentUser', JSON.stringify(user));
        
        console.log('localStorage设置完成，现在获取:', {
          accessToken: localStorage.getItem('accessToken'),
          refreshToken: localStorage.getItem('refreshToken'),
          currentUser: localStorage.getItem('currentUser')
        });
        
        fetchAvailableModels();
      } else {
        setLoginError(response.data.message || '登录失败');
      }
    } catch (error) {
      console.error('登录失败:', error);
      setLoginError('登录失败，请稍后重试');
    } finally {
      setIsLoginLoading(false);
    }
  };

  // 处理登出
  const handleLogout = () => {
    setAuthState({
      isAuthenticated: false,
      user: null,
      accessToken: null,
      refreshToken: null
    });
    
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('currentUser');
    
    setMessages([
      {
        id: "1",
        role: "assistant",
        content: "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
        timestamp: Date.now(),
      },
    ]);
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
  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
  }, []);

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
        formData.append('use_coreml', useCoreML ? 'true' : 'false');
        formData.append('use_model', (selectedModel !== 'default') ? 'true' : 'false');
        formData.append('use_attributes', useAttributes ? 'true' : 'false');
        formData.append('model_name', selectedModel);
        formData.append('cache_bypass', Date.now().toString());

        console.log('发送请求参数:', {
          useCoreML,
          useModel: selectedModel !== 'default',
          useAttributes,
          modelName: selectedModel,
          isMacPlatform,
          multiRole: multiRole
        });
        
        // 确保use_model参数正确设置
        const useModelValue = selectedModel !== 'default' ? 'true' : 'false';
        console.log('useModelValue:', useModelValue);

        // 发送请求到后端API
        console.log('开始发送请求到API');
        console.log('authState:', authState);
        console.log('accessToken存在:', !!authState.accessToken);
        const endpoint = multiRole ? '/api/classify/multi-role' : '/api/classify';
        console.log(`使用端点: ${endpoint}`);
        const headers: any = {};
        if (authState.accessToken) {
          headers['Authorization'] = `Bearer ${authState.accessToken}`;
          console.log('添加Authorization头:', headers['Authorization']);
        } else {
          console.log('没有accessToken，无法添加Authorization头');
        }
        console.log('最终请求头:', headers);
        const response = await axios.post(endpoint, formData, {
          headers: headers,
        });

        console.log('API响应:', response.data);

        const data = response.data;

        // 构建助手消息
        let assistantMessage: Message;
        
        if (multiRole) {
          // 多角色检测结果
          const roles = data.data.roles || [];
          const count = data.data.count || 0;
          
          assistantMessage = {
            id: Date.now().toString(),
            role: "assistant",
            content: `多角色识别完成！检测到 ${count} 个角色`,
            multi_roles: roles.map((role: any, index: number) => ({
              id: role.id || index + 1,
              role: role.role || "未知角色",
              similarity: role.similarity || 0,
              confidence: role.confidence || 0,
              box: role.box || {},
              attributes: role.attributes || []
            })),
            nsfw: data.data.nsfw,
            thoughts: ["正在分析图片...", "正在检测多个角色...", "正在提取特征...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        } else {
          // 单角色检测结果
          assistantMessage = {
            id: Date.now().toString(),
            role: "assistant",
            content: `识别完成！${data.data.mode ? ` (使用 ${data.data.mode})` : ''}`,
            classification: {
              role: data.data.role || data.data.ai_predicted_role || data.data.predicted_role || "未知角色",
              similarity: data.data.similarity || 0,
              confidence: data.data.confidence || "medium",
            },
            attributes: data.data.attributes || [],
            tags: data.data.tags || [],
            ai_predicted_role: data.data.ai_predicted_role,
            nsfw: data.data.nsfw,
            possible_roles: data.data.possible_roles,
            thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        }

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
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage, useCoreML, selectedModel, useAttributes, authState]);

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
      {!authState.isAuthenticated ? (
        <Login 
          darkMode={darkMode} 
          onLogin={handleLogin} 
          isLoading={isLoginLoading} 
          error={loginError} 
        />
      ) : (
        <>
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
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">动漫角色识别</h1>
            
            {/* 用户信息 */}
            {authState.user && (
              <div className={`flex items-center space-x-4 px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <div className="flex items-center space-x-2">
                  <User className="h-5 w-5 text-blue-500" />
                  <span className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {authState.user.username}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    authState.user.role === 'admin' 
                      ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' 
                      : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                  }`}>
                    {authState.user.role}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                  title="退出登录"
                >
                  <LogOut className="h-5 w-5 text-red-500" />
                </button>
              </div>
            )}
          </div>
          
          {/* 模型选择和控制 */}
          <div className="flex items-center space-x-4">
              {/* 模型选择 */}
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">模型:</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className={`px-3 py-1.5 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                >
                  {availableModels.map(model => (
                    <option key={model} value={model}>
                      {model === 'default' ? '默认 (CLIP)' : model}
                    </option>
                  ))}
                </select>
              </div>
              
              {/* CoreML 开关 (仅 Mac 平台显示) */}
              {isMacPlatform && (
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium">CoreML:</label>
                  <button
                    onClick={() => setUseCoreML(!useCoreML)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useCoreML ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useCoreML ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
              )}
              
              {/* 属性预测开关 */}
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">属性:</label>
                <button
                  onClick={() => setUseAttributes(!useAttributes)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useAttributes ? 'bg-blue-600' : 'bg-gray-300'}`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useAttributes ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
              
              {/* 多角色检测开关 */}
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">多角色:</label>
                <button
                  onClick={() => setMultiRole(!multiRole)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${multiRole ? 'bg-blue-600' : 'bg-gray-300'}`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${multiRole ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
              
              {/* 暗黑模式开关 */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-100 text-gray-600'} transition-colors`}
                title={darkMode ? '切换到亮色模式' : '切换到暗黑模式'}
              >
                {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>
            </div>
          
          {/* 平台信息 */}
          {isMacPlatform && useCoreML && (
            <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
              🍎 检测到 Mac 平台，已启用 CoreML 加速
            </div>
          )}
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
        </>
      )}
    </div>
  );
}
