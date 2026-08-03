"use client";

import { useState, useRef, useEffect } from "react";
import { X, Sparkles } from "lucide-react";
import { Message } from "../types";
import MessageItem from "./MessageItem";

interface ChatPanelProps {
  darkMode: boolean;
  messages: Message[];
  inputText: string;
  isBatchUpload: boolean;
  isProcessing: boolean;
  selectedImage: File | null;
  imagePreview: string | null;
  selectedImages: File[];
  imagePreviews: string[];
  onInputChange: (text: string) => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onImageSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSend: () => void;
  onRemoveImage: () => void;
  onRemoveBatchImage: (index: number) => void;
  onClearBatchImages: () => void;
  onCopyMessage: (content: string) => void;
  onDownloadMessage: (content: string, role: string) => void;
}

export default function ChatPanel({
  darkMode,
  messages,
  inputText,
  isBatchUpload,
  isProcessing,
  selectedImage,
  imagePreview,
  selectedImages,
  imagePreviews,
  onInputChange,
  onKeyPress,
  onImageSelect,
  onSend,
  onRemoveImage,
  onRemoveBatchImage,
  onClearBatchImages,
  onCopyMessage,
  onDownloadMessage,
}: ChatPanelProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className={`w-full flex-1 flex flex-col min-h-0 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-lg border ${darkMode ? 'border-gray-700' : 'border-gray-200'} transform transition-all duration-300 hover:shadow-xl`}>
      <div className="flex-1 overflow-y-auto min-h-0 px-4 md:px-6 py-3 md:py-4">
        {messages.map((message, index) => (
          <div key={message.id} className="space-y-3 mb-4">
            <MessageItem
              message={message}
              darkMode={darkMode}
              handleCopyMessage={onCopyMessage}
              handleDownloadMessage={onDownloadMessage}
            />
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className={`shrink-0 p-3 md:p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="flex flex-col md:flex-row items-stretch md:items-center space-y-3 md:space-y-0 md:space-x-4">
          <input
            type="file"
            accept="image/*"
            multiple={isBatchUpload}
            onChange={onImageSelect}
            className={`w-full md:w-1/4 px-3 py-2 md:px-4 md:py-3 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
          />
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputText}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyPress={onKeyPress}
              placeholder="输入消息或上传图片..."
              className={`w-full px-4 py-2 md:px-5 md:py-3 pr-12 md:pr-16 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm transition-all hover:border-blue-300`}
              disabled={isProcessing}
            />
            <button
              onClick={() => onInputChange("")}
              className={`absolute right-8 top-1/2 transform -translate-y-1/2 p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
              title="清空输入"
              disabled={!inputText.trim() || isProcessing}
            >
              <X className={`h-4 w-4 ${inputText.trim() && !isProcessing ? '' : 'opacity-50 cursor-not-allowed'}`} />
            </button>
          </div>
          <button
            onClick={onSend}
            disabled={(!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing}
            className={`w-full md:w-auto min-w-[120px] bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 md:px-6 md:py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-1 md:space-x-2 ${(!inputText.trim() && !selectedImage && selectedImages.length === 0) || isProcessing ? 'opacity-50 cursor-not-allowed' : 'transform hover:scale-105 hover:shadow-lg'}`}
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

        {!isBatchUpload && selectedImage && imagePreview && (
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
              onClick={onRemoveImage}
              className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors transform hover:scale-110`}
              title="移除图片"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {isBatchUpload && selectedImages.length > 0 && (
          <div className={`mt-3 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-3 animate-fade-in`}>
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-medium">已选择 {selectedImages.length} 张图片</h3>
              <button
                onClick={onClearBatchImages}
                className={`text-xs px-2 py-1 rounded ${darkMode ? 'bg-red-900/20 text-red-400' : 'bg-red-50 text-red-600'} hover:opacity-80 transition-opacity`}
                title="清空所有图片"
              >
                清空
              </button>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {selectedImages.map((file, index) => (
                <div key={index} className={`relative ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg overflow-hidden shadow-md group`}>
                  <div className="aspect-square">
                    <img
                      src={imagePreviews[index]}
                      alt={`Selected image ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button
                      onClick={() => onRemoveBatchImage(index)}
                      className={`p-1.5 rounded-full bg-red-500 text-white hover:bg-red-600 transition-colors transform hover:scale-110`}
                      title="移除图片"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="p-2">
                    <p className="text-xs truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {Math.round(file.size / 1024)} KB
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}