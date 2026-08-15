import { useState, useCallback, useRef } from 'react';
import { Message } from '../types';
import { RecognitionService } from '../api/services/RecognitionService';

interface RecognitionOptions {
  useCoreML?: boolean;
  useModel?: boolean;
  useAttributes?: boolean;
  modelName?: string;
  multiRole?: boolean;
  threshold?: number;
  useYolo?: boolean;
  debug?: boolean;
}

export const useRecognition = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const lastRequestTimeRef = useRef<number>(0);
  const requestLockRef = useRef<boolean>(false);
  const REQUEST_DEBOUNCE_MS = 1000;

  const createProcessingMessage = useCallback((content: string, thoughts: string[]): Message => ({
    id: `processing_${Date.now()}`,
    role: 'assistant',
    content,
    isThinking: true,
    thoughts,
    isThinkingFinished: false,
    timestamp: Date.now(),
  }), []);

  const createUserMessage = useCallback((inputText: string, image?: string): Message => ({
    id: Date.now().toString(),
    role: 'user',
    content: inputText,
    image: image || undefined,
    timestamp: Date.now(),
  }), []);

  const createErrorResponse = useCallback((error: any): Message => {
    let errorContent = '识别过程中出现错误，请重试。';
    let errorTitle = '识别失败';

    if (error.response?.status === 401) {
      errorContent = '认证已过期，请重新登录。';
      errorTitle = '认证失败';
    } else if (error.response) {
      errorContent = error.response.data?.detail || error.response.data?.error || errorContent;
    } else if (error.message) {
      errorContent = error.message;
    }

    return {
      id: Date.now().toString(),
      role: 'assistant',
      content: errorTitle,
      error: errorContent,
      timestamp: Date.now(),
    };
  }, []);

  const createRecognitionResponse = useCallback((data: any, options: RecognitionOptions): Message => {
    if (options.useYolo) {
      const roles = data.roles || [];
      const count = data.count || 0;
      const detector = data.detector || 'YOLOv8';

      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: `YOLOv8 多目标检测完成！检测到 ${count} 个角色 (${detector})`,
        multi_roles: roles.map((role: any, index: number) => ({
          id: role.id || index + 1,
          role: role.role || '未知角色',
          role_cn: role.role_cn || '',
          role_jp: role.role_jp || '',
          role_anime: role.role_anime || '',
          similarity: role.confidence || 0,
          confidence: role.confidence || 0,
          box: role.bbox ? { x1: role.bbox[0], y1: role.bbox[1], x2: role.bbox[2], y2: role.bbox[3] } : {},
          attributes: [],
          decision: '',
          is_unknown: false,
          is_fuzzy: false,
          used_model: true,
        })),
        tags: [],
        text_detections: [],
        summary: `YOLOv8 检测到 ${count} 个角色`,
        thoughts: ['正在分析图片...', 'YOLOv8 人体检测...', '角色分类...', '检测完成！'],
        isThinkingFinished: true,
        timestamp: Date.now(),
        debug: data.debug || undefined,
      };
    }

    if (options.multiRole) {
      const roles = data.roles || [];
      const count = data.count || 0;
      const fallback = data.fallback || false;

      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.summary || `多角色识别完成！检测到 ${count} 个角色`,
        multi_roles: roles.map((role: any, index: number) => ({
          id: role.id || index + 1,
          role: role.role || '未知角色',
          role_cn: role.role_cn || '',
          role_jp: role.role_jp || '',
          role_anime: role.role_anime || '',
          similarity: role.similarity || 0,
          confidence: role.confidence || 0,
          box: role.box || {},
          attributes: role.attributes || [],
          decision: role.decision || '',
          is_unknown: role.is_unknown || false,
          is_fuzzy: role.is_fuzzy || false,
          used_model: role.used_model || false,
        })),
        tags: data.tags || [],
        text_detections: data.text_detections || [],
        nsfw: data.nsfw,
        summary: data.summary,
        fallback,
        model_coverage: data.model_coverage,
        thoughts: ['正在分析图片...', '正在检测多个角色...', '正在提取特征...', '识别完成！'],
        isThinkingFinished: true,
        timestamp: Date.now(),
        debug: data.debug || undefined,
      };
    }

    return {
      id: Date.now().toString(),
      role: 'assistant',
      content: data.summary || `识别完成！${data.mode ? ` (使用 ${data.mode})` : ''}`,
      classification: {
        role: data.role || data.ai_predicted_role || data.predicted_role || '未知角色',
        role_cn: data.role_cn || '',
        role_jp: data.role_jp || '',
        role_anime: data.role_anime || '',
        similarity: data.similarity || 0,
        confidence: (data.confidence as 'high' | 'medium' | 'low') || 'medium',
        used_model: true,
      },
      attributes: data.attributes || [],
      tags: data.tags || [],
      text_detections: data.text_detections || [],
      ai_predicted_role: data.ai_predicted_role,
      model_coverage: data.model_coverage,
      nsfw: data.nsfw,
      possible_roles: data.possible_roles,
      summary: data.summary,
      thoughts: ['正在分析图片...', '正在提取特征...', '正在匹配角色...', '识别完成！'],
      isThinkingFinished: true,
      timestamp: Date.now(),
    };
  }, []);

  const createBatchResponse = useCallback((results: any[]): Message => ({
    id: Date.now().toString(),
    role: 'assistant',
    content: `批量识别完成！处理了 ${results.length} 张图片`,
    batch_results: results.map((result: any, index: number) => ({
      id: index + 1,
      filename: result.filename || `图片 ${index + 1}`,
      role: result.role || '未知角色',
      similarity: result.similarity || 0,
      attributes: result.attributes || [],
      roles: result.roles || [],
    })),
    thoughts: ['正在分析图片...', '正在提取特征...', '正在批量处理...', '识别完成！'],
    isThinkingFinished: true,
    timestamp: Date.now(),
  }), []);

  const isRequestAllowed = useCallback((): boolean => {
    const now = Date.now();
    if (now - lastRequestTimeRef.current < REQUEST_DEBOUNCE_MS) {
      return false;
    }
    if (requestLockRef.current) {
      return false;
    }
    return true;
  }, []);

  const setRequestLock = useCallback((locked: boolean) => {
    requestLockRef.current = locked;
    if (locked) {
      lastRequestTimeRef.current = Date.now();
    }
  }, []);

  const classify = useCallback(
    async (
      file: File,
      imagePreview: string | null,
      inputText: string,
      options: RecognitionOptions = {}
    ): Promise<{ messages: Message[]; shouldLogout: boolean }> => {
      if (!isRequestAllowed()) {
        return { messages: [], shouldLogout: false };
      }

      setRequestLock(true);
      setIsProcessing(true);

      const messages: Message[] = [];
      let shouldLogout = false;

      const userMessage = createUserMessage(inputText, imagePreview || undefined);
      messages.push(userMessage);

      const processingMessage = createProcessingMessage('正在识别...', [
        '正在分析图片...',
        '正在提取特征...',
        '正在匹配角色...',
      ]);
      messages.push(processingMessage);

      try {
        const response = await RecognitionService.classify(file, options);

        // 防御性兼容：无论路由返回的是标准信封（{success, data}）
        // 还是被多包一层（{data:{success, data}}），都能正确取值。
        // response 的静态类型为 RecognitionResponse，但运行时可能为被多包一层的结构，
        // 故以 any 访问，兼容两种形态。
        const resp: any = response;
        const success = resp.success ?? resp.data?.success;
        const payload = resp.data?.data ?? resp.data;

        if (success && payload) {
          const resultMessage = createRecognitionResponse(payload, options);
          messages.push(resultMessage);
        } else {
          const errorMessage = createErrorResponse(new Error(resp.message || resp.data?.message || '识别失败'));
          messages.push(errorMessage);
        }
      } catch (error: any) {
        if (error.response?.status === 401) {
          shouldLogout = true;
        }
        const errorMessage = createErrorResponse(error);
        messages.push(errorMessage);
      } finally {
        setIsProcessing(false);
        setRequestLock(false);
      }

      return { messages, shouldLogout };
    },
    [isRequestAllowed, setRequestLock, createUserMessage, createProcessingMessage, createRecognitionResponse, createErrorResponse]
  );

  const batchClassify = useCallback(
    async (
      files: File[],
      imagePreviews: string[],
      inputText: string
    ): Promise<{ messages: Message[]; shouldLogout: boolean }> => {
      if (!isRequestAllowed()) {
        return { messages: [], shouldLogout: false };
      }

      setRequestLock(true);
      setIsProcessing(true);

      const messages: Message[] = [];
      let shouldLogout = false;

      const userMessage = createUserMessage(inputText, imagePreviews[0] || undefined);
      messages.push(userMessage);

      const processingMessage = createProcessingMessage(`正在识别 ${files.length} 张图片...`, [
        '正在分析图片...',
        '正在提取特征...',
        '正在批量处理...',
      ]);
      messages.push(processingMessage);

      try {
        const response = await RecognitionService.batchClassify(files);

        if (response.success && response.results) {
          const resultMessage = createBatchResponse(response.results);
          messages.push(resultMessage);
        } else {
          const errorMessage = createErrorResponse(new Error(response.message || '批量识别失败'));
          messages.push(errorMessage);
        }
      } catch (error: any) {
        if (error.response?.status === 401) {
          shouldLogout = true;
        }
        const errorMessage = createErrorResponse(error);
        messages.push(errorMessage);
      } finally {
        setIsProcessing(false);
        setRequestLock(false);
      }

      return { messages, shouldLogout };
    },
    [isRequestAllowed, setRequestLock, createUserMessage, createProcessingMessage, createBatchResponse, createErrorResponse]
  );

  return {
    isProcessing,
    classify,
    batchClassify,
  };
};