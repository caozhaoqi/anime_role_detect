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
        className={`max-w-[80%] ${message.role === "user" ? "order-1" : "order-2"}`}
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
              <div className={`grid grid-cols-3 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">角色</p>
                  <p className="text-sm font-medium">{message.classification.role}</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">相似度</p>
                  <p className="text-sm font-medium">{(message.classification.similarity * 100).toFixed(1)}%</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">置信度</p>
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
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">多角色识别结果</h4>
              </div>
              <div className="space-y-2">
                {message.multi_roles.map((role, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center">
                      <p className="text-sm font-medium">{role.role}</p>
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

          {message.text_detections && message.text_detections.length > 0 && (
            <div className="mt-4 space-y-3 animate-fade-in">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <h4 className="font-semibold text-sm">文本检测</h4>
              </div>
              <div className="space-y-2">
                {message.text_detections.map((text, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <p className="text-sm font-medium">{text.text}</p>
                  </div>
                ))}
              </div>
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