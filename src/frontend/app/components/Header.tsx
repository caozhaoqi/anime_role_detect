"use client";

import { useState } from "react";
import { User, LogOut, History, Settings, Moon, Sun } from "lucide-react";
import { AuthState } from "../types";

interface HeaderProps {
  darkMode: boolean;
  authState: AuthState;
  config: any;
  showHistory: boolean;
  showConfig: boolean;
  availableModels: string[];
  selectedModel: string;
  isMacPlatform: boolean;
  useCoreML: boolean;
  useAttributes: boolean;
  multiRole: boolean;
  isBatchUpload: boolean;
  onShowHistoryChange: (value: boolean) => void;
  onShowConfigChange: (value: boolean) => void;
  onDarkModeToggle: () => void;
  onLogout: () => void;
  onModelChange: (model: string) => void;
  onCoreMLChange: (value: boolean) => void;
  onAttributesChange: (value: boolean) => void;
  onMultiRoleChange: (value: boolean) => void;
  onBatchUploadChange: (value: boolean) => void;
}

export default function Header({
  darkMode,
  authState,
  config,
  showHistory,
  showConfig,
  availableModels,
  selectedModel,
  isMacPlatform,
  useCoreML,
  useAttributes,
  multiRole,
  isBatchUpload,
  onShowHistoryChange,
  onShowConfigChange,
  onDarkModeToggle,
  onLogout,
  onModelChange,
  onCoreMLChange,
  onAttributesChange,
  onMultiRoleChange,
  onBatchUploadChange,
}: HeaderProps) {
  const [internalBatchUpload, setInternalBatchUpload] = useState(isBatchUpload);

  const handleBatchUploadToggle = () => {
    const newValue = !internalBatchUpload;
    setInternalBatchUpload(newValue);
    onBatchUploadChange(newValue);
  };

  return (
    <header className={`sticky top-0 z-50 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b transition-all duration-300`}>
      <div className="container mx-auto px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* 左侧：标题 */}
          <div className="flex items-center">
            <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">动漫角色识别</h1>
          </div>
          
          {/* 右侧：配置和个人信息 */}
          <div className="flex flex-wrap items-center gap-4">
            {config.features.enableHistoryPanel && (
              <button
                onClick={() => onShowHistoryChange(!showHistory)}
                className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors ${showHistory ? 'text-blue-500' : ''}`}
                title="查看历史记录"
              >
                <History className="h-5 w-5" />
              </button>
            )}
            
            <button
              onClick={() => onShowConfigChange(!showConfig)}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors ${showConfig ? 'text-blue-500' : ''}`}
              title="配置"
            >
              <Settings className="h-5 w-5" />
            </button>
            
            <button
              onClick={onDarkModeToggle}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-100 text-gray-600'} transition-colors`}
              title={darkMode ? '切换到亮色模式' : '切换到暗黑模式'}
            >
              {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
            
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
                  onClick={onLogout}
                  className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                  title="退出登录"
                >
                  <LogOut className="h-5 w-5 text-red-500" />
                </button>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex flex-wrap items-center gap-4 mt-4">
          {config.features.enableModelSelection && (
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">模型:</label>
              <select
                value={selectedModel}
                onChange={(e) => onModelChange(e.target.value)}
                className={`px-3 py-1.5 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>
                    {model === 'default' ? '默认 (CLIP)' : model}
                  </option>
                ))}
              </select>
            </div>
          )}
          
          {isMacPlatform && config.features.enableCoremlSwitch && (
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">CoreML:</label>
              <button
                onClick={() => onCoreMLChange(!useCoreML)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useCoreML ? 'bg-blue-600' : 'bg-gray-300'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useCoreML ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
          )}
          
          {config.features.enableAttributesSwitch && (
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">属性:</label>
              <button
                onClick={() => onAttributesChange(!useAttributes)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useAttributes ? 'bg-blue-600' : 'bg-gray-300'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useAttributes ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
          )}
          
          {config.features.enableMultiRoleSwitch && (
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">多角色:</label>
              <button
                onClick={() => onMultiRoleChange(!multiRole)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${multiRole ? 'bg-blue-600' : 'bg-gray-300'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${multiRole ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
          )}
          
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium">批量上传:</label>
            <button
              onClick={handleBatchUploadToggle}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${internalBatchUpload ? 'bg-blue-600' : 'bg-gray-300'}`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${internalBatchUpload ? 'translate-x-6' : 'translate-x-1'}`} />
            </button>
          </div>
        </div>
        
        {isMacPlatform && useCoreML && (
          <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
            🍎 检测到 Mac 平台，已启用 CoreML 加速
          </div>
        )}
      </div>
    </header>
  );
}