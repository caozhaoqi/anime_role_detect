"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Message, AuthState } from "./types";
import axios from 'axios';
import Login from './components/Login';
import HistoryPanel from './components/HistoryPanel';
import ConfigManager from './config/ConfigManager';
import ConfigPanel from './components/ConfigPanel';
import SearchPanel from './components/SearchPanel';
import VideoPanel from './components/VideoPanel';
import Header from './components/Header';
import TabSwitcher from './components/TabSwitcher';
import ChatPanel from './components/ChatPanel';

export default function AnimeRoleDetect() {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    accessToken: null,
    refreshToken: null
  });
  const [isLoginLoading, setIsLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [showSessionExpired, setShowSessionExpired] = useState(false);
  
  const [showHistory, setShowHistory] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<any>(null);
  
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
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [isBatchUpload, setIsBatchUpload] = useState(false);
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
  const [showConfig, setShowConfig] = useState(false);
  const [config, setConfig] = useState(ConfigManager.getConfig());
  const [activePanel, setActivePanel] = useState<'classify' | 'search' | 'video'>('classify');
  
  const isMountedRef = useRef(false);

  useEffect(() => {
    isMountedRef.current = true;

    // 设置Axios拦截器处理认证过期
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response && error.response.status === 401) {
          // 清除认证状态
          setAuthState({
            isAuthenticated: false,
            user: null,
            accessToken: null,
            refreshToken: null
          });
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          localStorage.removeItem('currentUser');
          
          // 显示会话过期提示
          setShowSessionExpired(true);
          setTimeout(() => setShowSessionExpired(false), 5000);
        }
        return Promise.reject(error);
      }
    );

    return () => {
      axios.interceptors.response.eject(interceptor);
    };
  }, []);

  useEffect(() => {
    isMountedRef.current = true;

    const appConfig = ConfigManager.getConfig();
    setConfig(appConfig);
    
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      setDarkMode(savedDarkMode === 'true');
    } else {
      setDarkMode(appConfig.ui.theme === 'dark');
    }

    const savedAccessToken = localStorage.getItem('accessToken');
    const savedRefreshToken = localStorage.getItem('refreshToken');
    const savedUser = localStorage.getItem('currentUser');
    
    if (savedAccessToken && savedRefreshToken && savedUser) {
      setAuthState({
        isAuthenticated: true,
        user: JSON.parse(savedUser),
        accessToken: savedAccessToken,
        refreshToken: savedRefreshToken
      });
    }

    const platform = navigator.platform.toLowerCase();
    const isMac = platform.includes('mac') || platform.includes('darwin');
    setIsMacPlatform(isMac);
    if (isMac && appConfig.features.enableCoremlSwitch) {
      setUseCoreML(true);
    }

    if (savedAccessToken && appConfig.features.enableModelSelection) {
      fetchAvailableModels();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowHistory(false);
        setShowConfig(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      isMountedRef.current = false;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const headers = authState.accessToken ? { Authorization: `Bearer ${authState.accessToken}` } : {};
      const response = await axios.get('/api/models', { headers });
      if (response.data.success) {
        const models = response.data.models || [];
        setAvailableModels(['default', ...models]);
      }
    } catch (error) {
      console.error('获取模型列表失败:', error);
    }
  };

  const handleLogin = async (username: string, password: string) => {
    setIsLoginLoading(true);
    setLoginError(null);
    
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post('/api/auth/login', formData);
      
      if (response.data.success) {
        const { access_token, refresh_token, username: userName, role } = response.data.data;
        
        const user = { username: userName, role };
        
        setAuthState({
          isAuthenticated: true,
          user,
          accessToken: access_token,
          refreshToken: refresh_token
        });
        
        localStorage.setItem('accessToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);
        localStorage.setItem('currentUser', JSON.stringify(user));
        
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

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (isBatchUpload) {
      const files = e.target.files;
      if (files && files.length > 0) {
        const newFiles = Array.from(files);
        setSelectedImages(newFiles);
        
        const previews: string[] = [];
        newFiles.forEach(file => {
          const reader = new FileReader();
          reader.onloadend = () => {
            if (isMountedRef.current) {
              previews.push(reader.result as string);
              if (previews.length === newFiles.length) {
                setImagePreviews(previews);
              }
            }
          };
          reader.readAsDataURL(file);
        });
      }
    } else {
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
    }
  };

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
  }, []);

  const removeBatchImage = useCallback((index: number) => {
    const newImages = [...selectedImages];
    const newPreviews = [...imagePreviews];
    newImages.splice(index, 1);
    newPreviews.splice(index, 1);
    setSelectedImages(newImages);
    setImagePreviews(newPreviews);
  }, [selectedImages, imagePreviews]);

  const clearBatchImages = useCallback(() => {
    setSelectedImages([]);
    setImagePreviews([]);
  }, []);

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

  const handleViewRecord = useCallback((record: any) => {
    setSelectedRecord(record);
    setShowHistory(false);
    
    const assistantMessage: Message = {
      id: `history-${record.id}`,
      role: 'assistant',
      content: `历史识别结果：${record.image_filename}`,
      image: null,
      tags: record.recognition_result.tags || [],
      attributes: record.recognition_result.attributes || [],
      text_detections: record.recognition_result.text_detections || [],
      nsfw: record.recognition_result.nsfw,
      role_info: record.recognition_result.role_info || null,
      model_name: record.model_used,
      timestamp: record.timestamp
    };
    
    setMessages([assistantMessage]);
  }, []);

  const handleSend = useCallback(async () => {
    if ((!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing) {
      return;
    }

    setInputText("");

    if (isBatchUpload && selectedImages.length > 0) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputText.trim(),
        image: imagePreviews[0] || undefined,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, userMessage]);

      setIsProcessing(true);

      const processingMessage: Message = {
        id: `processing_${Date.now()}`,
        role: "assistant",
        content: `正在识别 ${selectedImages.length} 张图片...`,
        isThinking: true,
        thoughts: ["正在分析图片...", "正在提取特征...", "正在批量处理..."],
        isThinkingFinished: false,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, processingMessage]);

      try {
        const formData = new FormData();
        selectedImages.forEach((file, index) => {
          formData.append('files', file, file.name);
        });
        formData.append('model_name', selectedModel);
        formData.append('use_attributes', useAttributes ? 'true' : 'false');
        formData.append('batch_size', '8');
        formData.append('multilabel', multiRole ? 'true' : 'false');
        formData.append('threshold', '0.4');

        const endpoint = '/api/model/batch-predict';
        const headers: any = {};
        if (authState.accessToken) {
          headers['Authorization'] = `Bearer ${authState.accessToken}`;
        }
        const response = await axios.post(endpoint, formData, { headers });

        const data = response.data;
        const results = data.results || [];

        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: `批量识别完成！处理了 ${results.length} 张图片`,
          batch_results: results.map((result: any, index: number) => ({
            id: index + 1,
            filename: result.filename || `图片 ${index + 1}`,
            role: result.role || "未知角色",
            similarity: result.similarity || 0,
            attributes: result.attributes || [],
            roles: result.roles || []
          })),
          thoughts: ["正在分析图片...", "正在提取特征...", "正在批量处理...", "识别完成！"],
          isThinkingFinished: true,
          timestamp: Date.now(),
        };

        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, assistantMessage];
        });

      } catch (error: any) {
        let errorContent = "批量识别过程中出现错误，请重试。";
        let errorTitle = "批量识别失败";

        if (error.response && error.response.status === 401) {
          errorContent = "认证已过期，请重新登录。";
          errorTitle = "认证失败";
          setAuthState({
            isAuthenticated: false,
            user: null,
            accessToken: null,
            refreshToken: null
          });
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          localStorage.removeItem('currentUser');
        } else if (error.response) {
          errorContent = error.response.data?.detail || error.response.data?.error || errorContent;
        } else if (error.message) {
          errorContent = error.message;
        }

        const errorMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: errorTitle,
          error: errorContent,
          timestamp: Date.now(),
        };

        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, errorMessage];
        });
      } finally {
        setIsProcessing(false);
        clearBatchImages();
      }
    } else if (selectedImage) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputText.trim(),
        image: selectedImage ? imagePreview || undefined : undefined,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, userMessage]);

      setIsProcessing(true);

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
        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('use_coreml', useCoreML ? 'true' : 'false');
        formData.append('use_model', (selectedModel !== 'default') ? 'true' : 'false');
        formData.append('use_attributes', useAttributes ? 'true' : 'false');
        formData.append('model_name', selectedModel);
        formData.append('multilabel', multiRole ? 'true' : 'false');
        formData.append('threshold', '0.4');
        formData.append('cache_bypass', Date.now().toString());

        const endpoint = multiRole ? '/api/classify/multi-role' : '/api/classify';
        const headers: any = {};
        if (authState.accessToken) {
          headers['Authorization'] = `Bearer ${authState.accessToken}`;
        }
        const response = await axios.post(endpoint, formData, { headers });

        const data = response.data;

        let assistantMessage: Message;
        
        if (multiRole) {
          const roles = data.data.roles || [];
          const count = data.data.count || 0;
          
          assistantMessage = {
            id: Date.now().toString(),
            role: "assistant",
            content: data.data.summary || `多角色识别完成！检测到 ${count} 个角色`,
            multi_roles: roles.map((role: any, index: number) => ({
              id: role.id || index + 1,
              role: role.role || "未知角色",
              similarity: role.similarity || 0,
              confidence: role.confidence || 0,
              box: role.box || {},
              attributes: role.attributes || []
            })),
            tags: data.data.tags || [],
            text_detections: data.data.text_detections || [],
            nsfw: data.data.nsfw,
            summary: data.data.summary,
            thoughts: ["正在分析图片...", "正在检测多个角色...", "正在提取特征...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        } else {
          assistantMessage = {
            id: Date.now().toString(),
            role: "assistant",
            content: data.data.summary || `识别完成！${data.data.mode ? ` (使用 ${data.data.mode})` : ''}`,
            classification: {
              role: data.data.role || data.data.ai_predicted_role || data.data.predicted_role || "未知角色",
              similarity: data.data.similarity || 0,
              confidence: data.data.confidence || "medium",
            },
            attributes: data.data.attributes || [],
            tags: data.data.tags || [],
            text_detections: data.data.text_detections || [],
            ai_predicted_role: data.data.ai_predicted_role,
            nsfw: data.data.nsfw,
            possible_roles: data.data.possible_roles,
            summary: data.data.summary,
            thoughts: ["正在分析图片...", "正在提取特征...", "正在匹配角色...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        }

        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, assistantMessage];
        });

      } catch (error: any) {
        let errorContent = "识别过程中出现错误，请重试。";
        let errorTitle = "识别失败";

        if (error.response && error.response.status === 401) {
          errorContent = "认证已过期，请重新登录。";
          errorTitle = "认证失败";
          setAuthState({
            isAuthenticated: false,
            user: null,
            accessToken: null,
            refreshToken: null
          });
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          localStorage.removeItem('currentUser');
        } else if (error.response) {
          errorContent = error.response.data?.detail || error.response.data?.error || errorContent;
        } else if (error.message) {
          errorContent = error.message;
        }

        const errorMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: errorTitle,
          error: errorContent,
          timestamp: Date.now(),
        };

        setMessages(prev => {
          const newMessages = prev.filter(msg => !msg.isThinking);
          return [...newMessages, errorMessage];
        });
      } finally {
        setIsProcessing(false);
        removeImage();
      }
    }
  }, [inputText, selectedImage, imagePreview, selectedImages, imagePreviews, isBatchUpload, isProcessing, removeImage, clearBatchImages, useCoreML, selectedModel, useAttributes, multiRole, authState]);

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

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

  const handleConfigUpdate = useCallback((newConfig: any) => {
    setConfig(newConfig);
    ConfigManager.updateConfig(newConfig);
  }, []);

  const handleDarkModeToggle = useCallback(() => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode.toString());
  }, [darkMode]);

  const handleBatchUploadChange = useCallback((value: boolean) => {
    setIsBatchUpload(value);
    if (!value) {
      setSelectedImage(null);
      setImagePreview(null);
    } else {
      setSelectedImages([]);
      setImagePreviews([]);
    }
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
          {showSessionExpired && (
            <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-[10000] animate-bounce">
              <div className="bg-red-500 text-white px-6 py-3 rounded-lg shadow-xl flex items-center space-x-2">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-medium">会话已过期，请重新登录</span>
              </div>
            </div>
          )}
          
          {isDragging && (
            <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-[9999] border-2 border-dashed border-blue-500 rounded-lg animate-pulse">
              <div className="text-center p-8 bg-white dark:bg-gray-800 rounded-xl shadow-2xl transform transition-transform duration-300 hover:scale-105">
                <div className="h-16 w-16 mx-auto mb-4 text-blue-500 animate-bounce flex items-center justify-center">
                  <svg className="h-16 w-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
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
          
          <Header
            darkMode={darkMode}
            authState={authState}
            config={config}
            showHistory={showHistory}
            showConfig={showConfig}
            availableModels={availableModels}
            selectedModel={selectedModel}
            isMacPlatform={isMacPlatform}
            useCoreML={useCoreML}
            useAttributes={useAttributes}
            multiRole={multiRole}
            isBatchUpload={isBatchUpload}
            onShowHistoryChange={setShowHistory}
            onShowConfigChange={setShowConfig}
            onDarkModeToggle={handleDarkModeToggle}
            onLogout={handleLogout}
            onModelChange={setSelectedModel}
            onCoreMLChange={setUseCoreML}
            onAttributesChange={setUseAttributes}
            onMultiRoleChange={setMultiRole}
            onBatchUploadChange={handleBatchUploadChange}
          />

          <TabSwitcher 
            darkMode={darkMode} 
            activePanel={activePanel} 
            onPanelChange={setActivePanel} 
          />
      
          <div className="flex-1 flex overflow-hidden">
            <main className={`flex-1 overflow-y-auto transition-all duration-300 ${showHistory ? 'md:ml-96' : ''}`}>
              <div className="flex-1 overflow-y-auto">
                <div className="container mx-auto px-4 md:px-6 py-6 md:py-8">
                  {activePanel === 'classify' ? (
                    <ChatPanel
                      darkMode={darkMode}
                      messages={messages}
                      inputText={inputText}
                      isBatchUpload={isBatchUpload}
                      isProcessing={isProcessing}
                      selectedImage={selectedImage}
                      imagePreview={imagePreview}
                      selectedImages={selectedImages}
                      imagePreviews={imagePreviews}
                      onInputChange={setInputText}
                      onKeyPress={handleKeyPress}
                      onImageSelect={handleImageSelect}
                      onSend={handleSend}
                      onRemoveImage={removeImage}
                      onRemoveBatchImage={removeBatchImage}
                      onClearBatchImages={clearBatchImages}
                      onCopyMessage={handleCopyMessage}
                      onDownloadMessage={handleDownloadMessage}
                    />
                  ) : activePanel === 'search' ? (
                    <SearchPanel darkMode={darkMode} accessToken={authState.accessToken ?? undefined} />
                  ) : (
                    <VideoPanel darkMode={darkMode} accessToken={authState.accessToken ?? undefined} />
                  )}
                </div>
              </div>
            </main>
            
            {showHistory && (
              <div className="fixed right-0 top-[6rem] bottom-0 w-full md:w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg z-40 transform transition-transform duration-300 ease-in-out">
                <div className="h-full overflow-y-auto">
                  <HistoryPanel 
                    darkMode={darkMode}
                    onViewRecord={handleViewRecord}
                    onDeleteRecord={() => {}}
                  />
                </div>
              </div>
            )}
            
            {showConfig && (
              <div className="fixed right-0 top-[6rem] bottom-0 w-full md:w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg z-40 transform transition-transform duration-300 ease-in-out">
                <div className="h-full overflow-y-auto">
                  <ConfigPanel 
                    darkMode={darkMode}
                    config={config}
                    onConfigUpdate={handleConfigUpdate}
                  />
                </div>
              </div>
            )}
          </div>
      
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