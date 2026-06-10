import React, { useState } from 'react';
import { X, Save, RefreshCw, Check, Sun, Moon, Eye, EyeOff, Sparkles, Download, Upload, History, Settings, Code } from 'lucide-react';

interface ConfigPanelProps {
  darkMode: boolean;
  config: any;
  onConfigUpdate: (config: any) => void;
  onClose: () => void;
}

const ConfigPanel: React.FC<ConfigPanelProps> = ({ darkMode, config, onConfigUpdate, onClose }) => {
  const [localConfig, setLocalConfig] = useState({ ...config });
  const [activeTab, setActiveTab] = useState('ui');
  const [isSaving, setIsSaving] = useState(false);

  const handleChange = (path: string, value: any) => {
    const keys = path.split('.');
    const newConfig = { ...localConfig };
    let current = newConfig;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;
    setLocalConfig(newConfig);
  };

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      onConfigUpdate(localConfig);
      setIsSaving(false);
    }, 500);
  };

  const handleReset = () => {
    setLocalConfig({ ...config });
  };

  const tabs = [
    { id: 'ui', label: '界面', icon: <Eye className="h-4 w-4" /> },
    { id: 'features', label: '功能', icon: <Sparkles className="h-4 w-4" /> },
    { id: 'api', label: 'API', icon: <Code className="h-4 w-4" /> },
    { id: 'appearance', label: '外观', icon: <Settings className="h-4 w-4" /> }
  ];

  return (
    <div className={`h-full flex flex-col ${darkMode ? 'bg-gray-800 text-gray-100' : 'bg-white text-gray-900'}`}>
      {/* 面板头部 */}
      <div className={`p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} flex items-center justify-between`}>
        <h3 className="text-lg font-semibold flex items-center space-x-2">
          <Settings className="h-5 w-5 text-blue-500" />
          <span>配置</span>
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleSave}
            disabled={isSaving}
            className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${isSaving ? 'opacity-50 cursor-not-allowed' : darkMode ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white'}`}
          >
            {isSaving ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>{isSaving ? '保存中...' : '保存'}</span>
          </button>
          <button
            onClick={onClose}
            className={`p-1.5 rounded-lg transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'}`}
            title="关闭"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* 标签页 */}
      <div className={`flex border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-1 px-4 py-3 text-sm font-medium transition-colors ${activeTab === tab.id ? (darkMode ? 'border-b-2 border-blue-500 text-blue-400' : 'border-b-2 border-blue-500 text-blue-600') : (darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100')}`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* 配置内容 */}
      <div className="flex-1 p-4 overflow-y-auto">
        {/* UI 配置 */}
        {activeTab === 'ui' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-md font-medium mb-3">界面设置</h4>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm">默认主题</label>
                  <select
                    value={localConfig.ui.theme}
                    onChange={(e) => handleChange('ui.theme', e.target.value)}
                    className={`px-3 py-1.5 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  >
                    <option value="light">亮色</option>
                    <option value="dark">暗色</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用暗黑模式切换</label>
                  <button
                    onClick={() => handleChange('ui.enableDarkMode', !localConfig.ui.enableDarkMode)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.ui.enableDarkMode ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.ui.enableDarkMode ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用过渡动画</label>
                  <button
                    onClick={() => handleChange('ui.animateTransitions', !localConfig.ui.animateTransitions)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.ui.animateTransitions ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.ui.animateTransitions ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">显示平台信息</label>
                  <button
                    onClick={() => handleChange('ui.showPlatformInfo', !localConfig.ui.showPlatformInfo)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.ui.showPlatformInfo ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.ui.showPlatformInfo ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用通知</label>
                  <button
                    onClick={() => handleChange('ui.enableNotifications', !localConfig.ui.enableNotifications)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.ui.enableNotifications ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.ui.enableNotifications ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-md font-medium mb-3">消息设置</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">欢迎消息</label>
                  <input
                    type="text"
                    value={localConfig.messages.welcomeMessage}
                    onChange={(e) => handleChange('messages.welcomeMessage', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">处理中消息</label>
                  <input
                    type="text"
                    value={localConfig.messages.processingMessage}
                    onChange={(e) => handleChange('messages.processingMessage', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 功能配置 */}
        {activeTab === 'features' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-md font-medium mb-3">功能开关</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm">启用模型选择</label>
                  <button
                    onClick={() => handleChange('features.enableModelSelection', !localConfig.features.enableModelSelection)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableModelSelection ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableModelSelection ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用 CoreML 开关</label>
                  <button
                    onClick={() => handleChange('features.enableCoremlSwitch', !localConfig.features.enableCoremlSwitch)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableCoremlSwitch ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableCoremlSwitch ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用属性预测</label>
                  <button
                    onClick={() => handleChange('features.enableAttributesSwitch', !localConfig.features.enableAttributesSwitch)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableAttributesSwitch ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableAttributesSwitch ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用多角色检测</label>
                  <button
                    onClick={() => handleChange('features.enableMultiRoleSwitch', !localConfig.features.enableMultiRoleSwitch)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableMultiRoleSwitch ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableMultiRoleSwitch ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用历史记录面板</label>
                  <button
                    onClick={() => handleChange('features.enableHistoryPanel', !localConfig.features.enableHistoryPanel)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableHistoryPanel ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableHistoryPanel ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用拖拽上传</label>
                  <button
                    onClick={() => handleChange('features.enableDragDrop', !localConfig.features.enableDragDrop)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableDragDrop ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableDragDrop ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用复制/下载</label>
                  <button
                    onClick={() => handleChange('features.enableCopyDownload', !localConfig.features.enableCopyDownload)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableCopyDownload ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableCopyDownload ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm">启用图片预览</label>
                  <button
                    onClick={() => handleChange('features.enableImagePreview', !localConfig.features.enableImagePreview)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localConfig.features.enableImagePreview ? 'bg-blue-600' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localConfig.features.enableImagePreview ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* API 配置 */}
        {activeTab === 'api' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-md font-medium mb-3">API 设置</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">基础 URL</label>
                  <input
                    type="text"
                    value={localConfig.api.baseUrl}
                    onChange={(e) => handleChange('api.baseUrl', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">超时时间 (毫秒)</label>
                  <input
                    type="number"
                    value={localConfig.api.timeout}
                    onChange={(e) => handleChange('api.timeout', parseInt(e.target.value))}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">重试次数</label>
                  <input
                    type="number"
                    value={localConfig.api.retryCount}
                    onChange={(e) => handleChange('api.retryCount', parseInt(e.target.value))}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">重试延迟 (毫秒)</label>
                  <input
                    type="number"
                    value={localConfig.api.retryDelay}
                    onChange={(e) => handleChange('api.retryDelay', parseInt(e.target.value))}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-md font-medium mb-3">验证设置</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">最大图片大小 (字节)</label>
                  <input
                    type="number"
                    value={localConfig.validation.maxImageSize}
                    onChange={(e) => handleChange('validation.maxImageSize', parseInt(e.target.value))}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">最小图像尺寸</label>
                  <input
                    type="number"
                    value={localConfig.validation.minImageDimension}
                    onChange={(e) => handleChange('validation.minImageDimension', parseInt(e.target.value))}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 外观配置 */}
        {activeTab === 'appearance' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-md font-medium mb-3">颜色设置</h4>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm mb-1">主色调</label>
                  <input
                    type="color"
                    value={localConfig.appearance.primaryColor}
                    onChange={(e) => handleChange('appearance.primaryColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">次要色调</label>
                  <input
                    type="color"
                    value={localConfig.appearance.secondaryColor}
                    onChange={(e) => handleChange('appearance.secondaryColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">强调色</label>
                  <input
                    type="color"
                    value={localConfig.appearance.accentColor}
                    onChange={(e) => handleChange('appearance.accentColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">成功色</label>
                  <input
                    type="color"
                    value={localConfig.appearance.successColor}
                    onChange={(e) => handleChange('appearance.successColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">警告色</label>
                  <input
                    type="color"
                    value={localConfig.appearance.warningColor}
                    onChange={(e) => handleChange('appearance.warningColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">错误色</label>
                  <input
                    type="color"
                    value={localConfig.appearance.errorColor}
                    onChange={(e) => handleChange('appearance.errorColor', e.target.value)}
                    className="w-full h-8 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-md font-medium mb-3">布局设置</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">侧边栏宽度</label>
                  <input
                    type="text"
                    value={localConfig.layout.sidebarWidth}
                    onChange={(e) => handleChange('layout.sidebarWidth', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">头部高度</label>
                  <input
                    type="text"
                    value={localConfig.layout.headerHeight}
                    onChange={(e) => handleChange('layout.headerHeight', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">页脚高度</label>
                  <input
                    type="text"
                    value={localConfig.layout.footerHeight}
                    onChange={(e) => handleChange('layout.footerHeight', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">内容边距</label>
                  <input
                    type="text"
                    value={localConfig.layout.contentPadding}
                    onChange={(e) => handleChange('layout.contentPadding', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 底部操作 */}
      <div className={`p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'} flex justify-between`}>
        <button
          onClick={handleReset}
          className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${darkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
        >
          <RefreshCw className="h-4 w-4" />
          <span>重置</span>
        </button>
        <div className="flex space-x-2">
          <button
            onClick={handleSave}
            disabled={isSaving}
            className={`flex items-center space-x-1 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${isSaving ? 'opacity-50 cursor-not-allowed' : darkMode ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white'}`}
          >
            {isSaving ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>{isSaving ? '保存中...' : '保存'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfigPanel;
