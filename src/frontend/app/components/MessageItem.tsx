import React from 'react';
import { Bot, User, Copy, Download, X, Sparkles, CheckCircle } from 'lucide-react';
import { Message } from '../types';

interface MessageItemProps {
  message: Message;
  darkMode: boolean;
  handleCopyMessage: (content: string) => void;
  handleDownloadMessage: (content: string, role: string) => void;
}

const MessageItem: React.FC<MessageItemProps> = ({ message, darkMode, handleCopyMessage, handleDownloadMessage }) => {
  const getCategoryInfo = (key: string) => {
    const categoryMap: Record<string, { label: string; color: string; bgColor: string }> = {
      'drawings': { label: '绘画', color: 'text-purple-500', bgColor: 'bg-purple-500' },
      'hentai': { label: '色情动漫', color: 'text-pink-500', bgColor: 'bg-pink-500' },
      'neutral': { label: '正常', color: 'text-gray-500', bgColor: 'bg-gray-500' },
      'porn': { label: '色情', color: 'text-red-500', bgColor: 'bg-red-500' },
      'sexy': { label: '性感', color: 'text-orange-500', bgColor: 'bg-orange-500' }
    };
    return categoryMap[key] || { label: key, color: 'text-blue-500', bgColor: 'bg-blue-500' };
  };

  const getHighestCategory = (details?: Record<string, number>): string => {
    if (!details) return 'unknown';
    const entries = Object.entries(details);
    if (entries.length === 0) return 'unknown';
    return entries.reduce((a, b) => Number(a[1]) > Number(b[1]) ? a : b)[0];
  };

  return (
    <div
      key={message.id}
      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
    >
      <div
        className={`flex-shrink-0 mr-2 ml-2 ${message.role === "user" ? "order-2" : "order-1"}`}
      >
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${message.role === "user" ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')} transition-transform hover:scale-110`}>
          {message.role === "user" ? (
            <User className="h-5 w-5" />
          ) : (
            <Bot className="h-5 w-5" />
          )}
        </div>
      </div>
      <div
        className={`max-w-full ${message.role === "user" ? "order-1" : "order-2"}`}
      >
        <div
          className={`rounded-xl p-4 ${message.role === "user" ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' : (darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900')} shadow-sm transition-all hover:shadow-md`}
        >
          {message.image && (
            <div className="mb-3 rounded-lg overflow-hidden shadow-md transform hover:scale-[1.02] transition-transform">
              <img
                src={message.image}
                alt="User uploaded image"
                className="w-full h-auto object-cover"
              />
            </div>
          )}
          <p className="whitespace-pre-wrap break-words">{message.content}</p>

          {message.classification && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">识别结果</h4>
              </div>
              <div className={`grid grid-cols-2 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">角色</p>
                  <p className="text-sm font-medium">{message.classification.role}</p>
                  {message.classification.role_cn && message.classification.role_cn !== message.classification.role && (
                    <p className="text-xs text-blue-500 mt-1">{message.classification.role_cn}</p>
                  )}
                  {message.classification.role_jp && (
                    <p className="text-xs text-pink-500 mt-1">{message.classification.role_jp}</p>
                  )}
                  {message.classification.role_anime && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{message.classification.role_anime}</p>
                  )}
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">相似度</p>
                  <p className="text-sm font-medium">{(message.classification.similarity * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}

          {message.multi_roles && message.multi_roles.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">多角色识别结果</h4>
              </div>
              <div className="space-y-2">
                {message.multi_roles.map((role, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="text-sm font-medium">{role.role}</p>
                        {role.role_cn && role.role_cn !== role.role && (
                          <p className="text-xs text-blue-500 mt-1">{role.role_cn}</p>
                        )}
                        {role.role_jp && (
                          <p className="text-xs text-pink-500 mt-1">{role.role_jp}</p>
                        )}
                        {role.role_anime && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{role.role_anime}</p>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm">{(role.similarity * 100).toFixed(1)}%</p>
                        <div
                          className={`w-2 h-2 rounded-full ${role.similarity >= 0.8 ? "bg-green-500" : role.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {message.attributes && message.attributes.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">角色属性</h4>
              </div>
              <div className="flex flex-wrap gap-2">
                {message.attributes.map((attr, index) => (
                  <span
                    key={index}
                    className={`px-4 py-2 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-sm font-medium transform hover:scale-105 transition-transform`}
                  >
                    {attr.tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* 只有当消息包含图片或识别结果时才显示文本检测 */}
          {(message.image || message.classification || message.multi_roles || message.attributes) && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">文本检测</h4>
              </div>
              {message.text_detections && message.text_detections.length > 0 ? (
                <div className="space-y-2">
                  {message.text_detections.map((text, index) => (
                    <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                      <p className="text-sm font-medium">{text.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg`}>
                  <p className="text-sm font-medium">图片中无文字</p>
                </div>
              )}
            </div>
          )}

          {message.ai_predicted_role && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <h4 className="font-semibold text-sm">AI预测角色</h4>
              </div>
              <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                <p className="text-sm font-medium">{message.ai_predicted_role}</p>
              </div>
            </div>
          )}

          {message.thoughts && !message.isThinkingFinished && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">识别过程</h4>
              </div>
              <div className="space-y-2">
                {message.thoughts.map((thought, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                    <p className="text-sm">{thought}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {message.nsfw && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${message.nsfw.is_nsfw ? "bg-red-500 animate-pulse" : "bg-green-500"}`} />
                <h4 className="font-semibold text-sm">NSFW 内容检测</h4>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  message.nsfw.is_nsfw
                    ? 'bg-red-100 text-red-600 dark:bg-red-900/50 dark:text-red-400'
                    : 'bg-green-100 text-green-600 dark:bg-green-900/50 dark:text-green-400'
                }`}>
                  {message.nsfw.is_nsfw ? "⚠️ 包含敏感内容" : "✅ 安全内容"}
                </span>
              </div>
              <div className={`grid grid-cols-3 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-3 ${message.nsfw.is_nsfw ? 'bg-red-900/20 border border-red-800' : darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">检测结果</p>
                  <div className="flex items-center space-x-2">
                    <p className="text-sm font-medium">
                      {message.nsfw.is_nsfw ? "NSFW" : "安全"}
                    </p>
                    <div
                      className={`w-2 h-2 rounded-full ${message.nsfw.is_nsfw ? "bg-red-500 animate-pulse" : "bg-green-500"}`}
                    />
                  </div>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">皮肤比例</p>
                  <p className="text-sm font-medium">{(message.nsfw.skin_ratio * 100).toFixed(1)}%</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">预测类别</p>
                  <p className="text-sm font-medium">
                    {getCategoryInfo(getHighestCategory(message.nsfw.details)).label}
                  </p>
                </div>
              </div>
              {message.nsfw.details && (
                <div className="mt-3 space-y-2">
                  <h5 className="text-xs text-gray-500 dark:text-gray-400">各类别概率分布</h5>
                  <div className="space-y-2">
                    {Object.entries(message.nsfw.details)
                      .sort(([, a], [, b]) => Number(b) - Number(a))
                      .map(([key, value]) => {
                        const percentage = (Number(value) * 100).toFixed(1);
                        const categoryInfo = getCategoryInfo(key);
                        const details = message.nsfw?.details;
                        const isHighest = details ? key === getHighestCategory(details) : false;
                        const isUnsafe = key === 'porn' || key === 'sexy' || key === 'hentai';

                        return (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between items-center">
                              <span className={`text-xs font-medium ${
                                isUnsafe ? categoryInfo.color : (darkMode ? 'text-gray-300' : 'text-gray-700')
                              }`}>
                                {categoryInfo.label}
                              </span>
                              <span className={`text-xs font-medium ${
                                isHighest ? 'text-blue-500 dark:text-blue-400' : ''
                              }`}>
                                {percentage}%
                              </span>
                            </div>
                            <div className={`h-2 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded-full overflow-hidden`}>
                              <div
                                className={`h-full ${categoryInfo.bgColor} rounded-full transition-all duration-500`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          )}

          {message.tags && message.tags.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">标签</h4>
              </div>
              <div className="flex flex-wrap gap-2">
                {message.tags.map((tag, index) => (
                  <span
                    key={index}
                    className={`px-4 py-2 ${darkMode ? 'bg-purple-900/50 text-purple-400' : 'bg-purple-100 text-purple-600'} rounded-full text-sm font-medium transform hover:scale-105 transition-transform`}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {message.possible_roles && message.possible_roles.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">其他模型检测结果</h4>
              </div>
              <div className="space-y-2">
                {message.possible_roles.map((role, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center">
                      <p className="text-sm font-medium">{role.role}</p>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm">{(role.probability * 100).toFixed(1)}%</p>
                        <div
                          className={`w-2 h-2 rounded-full ${role.probability >= 0.8 ? "bg-green-500" : role.probability >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {message.batch_results && message.batch_results.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">批量识别结果</h4>
              </div>
              <div className="space-y-3">
                {message.batch_results.map((result, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-sm font-medium">{result.filename}</p>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm">{(result.similarity * 100).toFixed(1)}%</p>
                        <div
                          className={`w-2 h-2 rounded-full ${result.similarity >= 0.8 ? "bg-green-500" : result.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                        />
                      </div>
                    </div>
                    <p className="text-sm">角色: {result.role}</p>
                    {result.roles && result.roles.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">多角色识别:</p>
                        <div className="flex flex-wrap gap-2">
                          {result.roles.map((role, roleIndex) => (
                            <span
                              key={roleIndex}
                              className={`px-3 py-1 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-xs font-medium`}
                            >
                              {role.role} ({(role.similarity * 100).toFixed(0)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
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
                className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="复制内容"
              >
                <Copy className="h-3 w-3" />
              </button>
              <button
                onClick={() => handleDownloadMessage(message.content, message.role)}
                className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="下载内容"
              >
                <Download className="h-3 w-3" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageItem;