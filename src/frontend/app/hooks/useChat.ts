import { useState, useCallback } from 'react';
import { Message } from '../types';
import { HistoryRecord } from '../api/services/HistoryService';
import { useAppStore } from '../store/useAppStore';

export const useChat = () => {
  const addToast = useAppStore((s) => s.addToast);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: '你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。',
      timestamp: Date.now(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [copySuccess, setCopySuccess] = useState<string | null>(null);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const addMessages = useCallback((newMessages: Message[]) => {
    setMessages((prev) => [...prev, ...newMessages]);
  }, []);

  const removeMessage = useCallback((messageId: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
  }, []);

  const removeThinkingMessages = useCallback(() => {
    setMessages((prev) => prev.filter((msg) => !msg.isThinking));
  }, []);

  const replaceThinkingWithMessages = useCallback((newMessages: Message[]) => {
    setMessages((prev) => {
      const filtered = prev.filter((msg) => !msg.isThinking);
      return [...filtered, ...newMessages];
    });
  }, []);

  const handleViewHistoryRecord = useCallback((record: HistoryRecord) => {
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
      timestamp: record.timestamp,
    };

    setMessages([assistantMessage]);
  }, []);

  const handleCopyMessage = useCallback((content: string) => {
    if (!content) return;

    navigator.clipboard
      .writeText(content)
      .then(() => {
        setCopySuccess('复制成功！');
        addToast('已复制到剪贴板', 'success');
        setTimeout(() => setCopySuccess(null), 3000);
      })
      .catch((err) => {
        console.error('复制失败:', err);
        addToast('复制失败，请重试', 'error');
      });
  }, [addToast]);

  const handleDownloadMessage = useCallback((content: string, role: string) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${role === 'user' ? '用户' : '助手'}_消息_${new Date().toISOString().slice(0, 19).replace(/[-:]/g, '')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const resetMessages = useCallback(() => {
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: '你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。',
        timestamp: Date.now(),
      },
    ]);
    setInputText('');
  }, []);

  return {
    messages,
    inputText,
    copySuccess,
    setInputText,
    addMessage,
    addMessages,
    removeMessage,
    removeThinkingMessages,
    replaceThinkingWithMessages,
    handleViewHistoryRecord,
    handleCopyMessage,
    handleDownloadMessage,
    resetMessages,
  };
};