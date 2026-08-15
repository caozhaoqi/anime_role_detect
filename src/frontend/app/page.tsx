"use client";

import { useState, useEffect, useCallback } from 'react';
import Login from './components/Login';
import Header from './components/Header';
import TabSwitcher from './components/TabSwitcher';
import ChatPanel from './components/ChatPanel';
import ConfigManager from './config/ConfigManager';
import { initApiClient } from './api/client';
import { RecognitionService } from './api/services/RecognitionService';
import { useAuth } from './hooks/useAuth';
import { useImageUpload } from './hooks/useImageUpload';
import { useRecognition } from './hooks/useRecognition';
import { useChat } from './hooks/useChat';
import { useAppStore } from './store/useAppStore';
import { Message } from './types';
import ErrorBoundary, { useGlobalErrorHandler } from './components/ErrorBoundary';
import ToastContainer from './components/ToastContainer';
import dynamic from 'next/dynamic';

const HistoryPanel = dynamic(() => import('./components/HistoryPanel'), {
  loading: () => <div className="animate-pulse h-96 bg-gray-100 dark:bg-gray-700 rounded-lg" />,
  ssr: false,
});

const ConfigPanel = dynamic(() => import('./components/ConfigPanel'), {
  loading: () => <div className="animate-pulse h-96 bg-gray-100 dark:bg-gray-700 rounded-lg" />,
  ssr: false,
});

const SearchPanel = dynamic(() => import('./components/SearchPanel'), {
  loading: () => <div className="animate-pulse h-96 bg-gray-100 dark:bg-gray-700 rounded-lg" />,
  ssr: false,
});

const VideoPanel = dynamic(() => import('./components/VideoPanel'), {
  loading: () => <div className="animate-pulse h-96 bg-gray-100 dark:bg-gray-700 rounded-lg" />,
  ssr: false,
});

const GeneratePanel = dynamic(() => import('./components/GeneratePanel'), {
  loading: () => <div className="animate-pulse h-96 bg-gray-100 dark:bg-gray-700 rounded-lg" />,
  ssr: false,
});

// 对话生成意图关键词：命中且未上传图片时，把输入当作 t2i 聊天请求
const T2I_INTENT_KW = [
  '生成', '画', '出图', '图片', '角色图', '绘', '做一张', '来一张',
  'generate', 'make', 'draw', 'create', 'image of', 'picture',
];

export default function AnimeRoleDetect() {
  useGlobalErrorHandler();

  const {
    authState,
    loginError,
    isLoginLoading,
    showSessionExpired,
    handleLogin,
    handleRegister,
    handleLogout,
    handleUnauthorized,
  } = useAuth();

  const { messages, inputText, copySuccess, isGenerating, setInputText, addMessage, replaceThinkingWithMessages, handleViewHistoryRecord, handleCopyMessage, handleDownloadMessage, resetMessages, generateFromChat } = useChat();

  const addToast = useAppStore((s) => s.addToast);

  const { selectedImage, imagePreview, selectedImages, imagePreviews, isDragging, handleImageSelect, handleDrop, handleDragEnter, handleDragOver, handleDragLeave, removeImage, removeBatchImage, clearBatchImages, reset } = useImageUpload({
    onMessageAdd: addMessage,
  });

  const { isProcessing, classify, batchClassify } = useRecognition();

  const [showHistory, setShowHistory] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [isMacPlatform, setIsMacPlatform] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('default');
  const [useCoreML, setUseCoreML] = useState(false);
  const [useAttributes, setUseAttributes] = useState(true);
  const [multiRole, setMultiRole] = useState(false);
  const [useYolo, setUseYolo] = useState(false);
  const [isBatchUpload, setIsBatchUpload] = useState(false);
  const [config, setConfig] = useState(ConfigManager.getConfig());
  const [activePanel, setActivePanel] = useState<'classify' | 'search' | 'video' | 'generate'>('classify');

  useEffect(() => {
    initApiClient({
      onUnauthorized: handleUnauthorized,
    });
  }, [handleUnauthorized]);

  useEffect(() => {
    const appConfig = ConfigManager.getConfig();
    setConfig(appConfig);

    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      setDarkMode(savedDarkMode === 'true');
    } else {
      setDarkMode(appConfig.ui.theme === 'dark');
    }

    const platform = navigator.platform.toLowerCase();
    const isMac = platform.includes('mac') || platform.includes('darwin');
    setIsMacPlatform(isMac);
    if (isMac && appConfig.features.enableCoremlSwitch) {
      setUseCoreML(true);
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowHistory(false);
        setShowConfig(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (authState.isAuthenticated && authState.accessToken) {
      const appConfig = ConfigManager.getConfig();
      if (appConfig.features.enableModelSelection) {
        fetchAvailableModels();
      }
    }
  }, [authState.isAuthenticated, authState.accessToken]);

  // 暗色模式：把 .dark 同步到 <html>，激活 dark: 类 + .dark 主题变量
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  useEffect(() => {
    const handler = () => {
      handleLogout();
    };
    window.addEventListener('open-login', handler);
    return () => window.removeEventListener('open-login', handler);
  }, [handleLogout]);

  const fetchAvailableModels = useCallback(async () => {
    try {
      const models = await RecognitionService.getModels();
      setAvailableModels(models);
    } catch {
      setAvailableModels(['default']);
    }
  }, []);

  const handleSend = useCallback(async () => {
    if ((!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing || isGenerating) {
      return;
    }

    // 对话生成意图：未上传图片且输入含"生成/画/…"关键词时，走 t2i 聊天生成
    const hasT2iIntent = !isBatchUpload && !selectedImage && selectedImages.length === 0 && inputText.trim().length > 0 &&
      T2I_INTENT_KW.some((k) => inputText.toLowerCase().includes(k.toLowerCase()));
    if (hasT2iIntent) {
      const { shouldLogout } = await generateFromChat(inputText, { onUnauthorized: handleLogout });
      if (shouldLogout) {
        return;
      }
      setInputText('');
      return;
    }

    if (isBatchUpload && selectedImages.length > 0) {
      const { messages: newMessages, shouldLogout } = await batchClassify(
        selectedImages,
        imagePreviews,
        inputText
      );

      if (shouldLogout) {
        handleLogout();
        return;
      }

      replaceThinkingWithMessages(newMessages);
      clearBatchImages();
    } else if (selectedImage) {
      const { messages: newMessages, shouldLogout } = await classify(
        selectedImage,
        imagePreview,
        inputText,
        {
          useCoreML,
          useModel: selectedModel !== 'default',
          useAttributes,
          modelName: selectedModel,
          multiRole,
          threshold: 0.4,
          useYolo,
          debug: useAppStore.getState().debugEnabled,
        }
      );

      if (shouldLogout) {
        handleLogout();
        return;
      }

      replaceThinkingWithMessages(newMessages);
      removeImage();
    }

    setInputText('');
  }, [inputText, selectedImage, imagePreview, selectedImages, imagePreviews, isBatchUpload, isProcessing, isGenerating, removeImage, clearBatchImages, useCoreML, selectedModel, useAttributes, multiRole, useYolo, classify, batchClassify, replaceThinkingWithMessages, generateFromChat, handleLogout, setInputText]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleSendToChat = useCallback((msg: Message) => {
    addMessage(msg);
    setActivePanel('classify');
    addToast('已发送到对话', 'success');
  }, [addMessage, setActivePanel, addToast]);

  const handleConfigUpdate = useCallback((newConfig: any) => {
    setConfig(newConfig);
    ConfigManager.updateConfig(newConfig);
  }, []);

  const handleDarkModeToggle = useCallback(() => {
    const newDarkMode = !darkMode;
    // 主题切换过渡：临时挂载全局 transition 类，350ms 后移除
    const html = document.documentElement;
    html.classList.add('theme-transition');
    setTimeout(() => html.classList.remove('theme-transition'), 350);
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode.toString());
  }, [darkMode]);

  const handleBatchUploadChange = useCallback((value: boolean) => {
    setIsBatchUpload(value);
    if (!value) {
      reset();
    }
  }, [reset]);

  const handleModelChange = useCallback((model: string) => {
    setSelectedModel(model);
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
          onRegister={handleRegister}
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
            useYolo={useYolo}
            isBatchUpload={isBatchUpload}
            onShowHistoryChange={setShowHistory}
            onShowConfigChange={setShowConfig}
            onDarkModeToggle={handleDarkModeToggle}
            onLogout={handleLogout}
            onModelChange={handleModelChange}
            onCoreMLChange={setUseCoreML}
            onAttributesChange={setUseAttributes}
            onMultiRoleChange={setMultiRole}
            onYoloChange={setUseYolo}
            onBatchUploadChange={handleBatchUploadChange}
          />

          <TabSwitcher
            darkMode={darkMode}
            activePanel={activePanel}
            onPanelChange={setActivePanel}
          />

          <div className="flex-1 flex overflow-hidden">
            <main className={`flex-1 transition-all duration-300 ${showHistory ? 'md:mr-96' : ''} ${activePanel === 'classify' ? 'flex flex-col overflow-hidden' : 'overflow-y-auto'}`}>
              <div key={activePanel} className={`${activePanel === 'classify' ? 'flex-1 flex flex-col min-h-0 overflow-hidden' : 'flex-1 overflow-y-auto'} animate-panel-in`}>
                <div className={`container mx-auto px-4 md:px-6 py-2 md:py-3 ${activePanel === 'classify' ? 'flex-1 flex flex-col min-h-0' : ''}`}>
                  {activePanel === 'classify' ? (
                    <ChatPanel
                      darkMode={darkMode}
                      messages={messages}
                      inputText={inputText}
                      isBatchUpload={isBatchUpload}
                      isProcessing={isProcessing}
                      isGenerating={isGenerating}
                      selectedImage={selectedImage}
                      imagePreview={imagePreview}
                      selectedImages={selectedImages}
                      imagePreviews={imagePreviews}
                      onInputChange={setInputText}
                      onKeyPress={handleKeyPress}
                      onImageSelect={(e) => handleImageSelect(e, isBatchUpload)}
                      onSend={handleSend}
                      onRemoveImage={removeImage}
                      onRemoveBatchImage={removeBatchImage}
                      onClearBatchImages={clearBatchImages}
                      onCopyMessage={handleCopyMessage}
                      onDownloadMessage={handleDownloadMessage}
                    />
                      ) : activePanel === 'search' ? (
                        <SearchPanel darkMode={darkMode} accessToken={authState.accessToken ?? undefined} onSendToChat={handleSendToChat} />
                      ) : activePanel === 'generate' ? (
                        <GeneratePanel darkMode={darkMode} />
                      ) : (
                        <VideoPanel darkMode={darkMode} accessToken={authState.accessToken ?? undefined} onSendToChat={handleSendToChat} />
                      )}
                </div>
              </div>
            </main>

            {showHistory && (
              <div className="fixed right-0 top-[6rem] bottom-0 w-full md:w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg z-40 transform transition-transform duration-300 ease-in-out">
                <div className="h-full overflow-y-auto">
                  <HistoryPanel
                    darkMode={darkMode}
                    onViewRecord={handleViewHistoryRecord}
                    onDeleteRecord={() => {}}
                    onClose={() => setShowHistory(false)}
                    onAuthError={handleUnauthorized}
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
                    onClose={() => setShowConfig(false)}
                  />
                </div>
              </div>
            )}
          </div>

          <footer className={`py-2 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'} transition-all duration-300`}>
            <div className="container mx-auto px-4 text-center text-xs text-gray-500 dark:text-gray-400">
              <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent font-medium">Anime Role detect ©Arona 2026</span>
              <span className="mx-2">·</span>
              <span>A role detection system based on deep learning</span>
            </div>
          </footer>

          <ToastContainer />
        </>
      )}
    </div>
  );
}