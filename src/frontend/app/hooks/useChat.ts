import { useState, useCallback } from 'react';
import { Message } from '../types';
import { HistoryRecord } from '../api/services/HistoryService';
import { GenerationService } from '../api/services/GenerationService';
import { useAppStore } from '../store/useAppStore';

export const useChat = () => {
  const addToast = useAppStore((s) => s.addToast);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: '你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。也可以说"生成 <角色名>"来画出对应角色的图像。',
      timestamp: Date.now(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

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
        content: '你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。也可以说"生成 <角色名>"来画出对应角色的图像。',
        timestamp: Date.now(),
      },
    ]);
    setInputText('');
  }, []);

  /**
   * 对话生成：把用户输入当作 t2i 聊天意图，调用后端 /api/t2i/chat。
   * 后端负责角色名匹配 + 意图识别；命中的话直接出图，未命中则给出提示。
   * 返回 { shouldLogout } 以便上层在鉴权失败时跳转登录。
   */
  const generateFromChat = useCallback(
    async (
      text: string,
      opts?: { method?: 'ip_adapter' | 'lora'; num?: number; onUnauthorized?: () => void }
    ): Promise<{ shouldLogout: boolean }> => {
      const userMsg: Message = {
        id: `u-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };
      addMessage(userMsg);

      const thinkingId = `t2i-think-${Date.now()}`;
      addMessage({
        id: thinkingId,
        role: 'assistant',
        content: '正在生成角色图像…',
        isThinking: true,
        isThinkingFinished: false,
        timestamp: Date.now(),
      });

      setIsGenerating(true);
      try {
        const data = await GenerationService.chat(
          text,
          opts?.method ?? 'ip_adapter',
          opts?.num ?? 1
        );
        // 未命中角色 / 无生成意图：直接把后端提示语作为助手消息
        if (!data.job_id) {
          replaceThinkingWithMessages([
            {
              id: `t2i-${Date.now()}`,
              role: 'assistant',
              content: data.reply || '未识别到角色。',
              timestamp: Date.now(),
            },
          ]);
          return { shouldLogout: false };
        }
        // 命中角色：轮询生成作业直到出图（后台推理，UI 不卡）
        const job = await GenerationService.pollJob(data.job_id);
        const result = job.result!;
        const assistantMsg: Message = {
          id: `t2i-${Date.now()}`,
          role: 'assistant',
          content: `已为角色「${data.matched_role}」生成 ${result.images.length} 张图。`,
          generated_role: data.matched_role ?? null,
          generated_images: result.images,
          generated_method: result.method,
          generated_fell_back: result.fell_back,
          timestamp: Date.now(),
        };
        replaceThinkingWithMessages([assistantMsg]);
        return { shouldLogout: false };
      } catch (e: any) {
        const status = e?.response?.status;
        let errMsg = '生成失败，请稍后重试';
        if (status === 401) {
          opts?.onUnauthorized?.();
          return { shouldLogout: true };
        }
        if (status === 404) {
          errMsg = e?.response?.data?.detail || '未找到该角色的参考图';
        } else if (e?.response?.data?.detail) {
          errMsg = String(e.response.data.detail);
        } else if (e?.message) {
          errMsg = e.message;
        }
        replaceThinkingWithMessages([
          {
            id: `t2i-err-${Date.now()}`,
            role: 'assistant',
            content: `生成失败：${errMsg}`,
            error: errMsg,
            timestamp: Date.now(),
          },
        ]);
        return { shouldLogout: false };
      } finally {
        setIsGenerating(false);
      }
    },
    [addMessage, replaceThinkingWithMessages]
  );

  return {
    messages,
    inputText,
    copySuccess,
    isGenerating,
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
    generateFromChat,
  };
};