"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, Image as ImageIcon, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Search, Settings, HelpCircle, Moon, Sun, Zap, Layers, Trash2, Clock } from "lucide-react";
import { Message, Model } from "./types";
import { useHistory } from "./hooks/useHistory";
import axios from 'axios';

export default function AnimeRoleDetect() {
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
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("default");
  const [useMultiRole, setUseMultiRole] = useState<boolean>(false);
  const [models, setModels] = useState<Model[]>([
    { name: "default", path: "", files: [], available: true, description: "默认分类模型" },
    { name: "mobilenet_v2", path: "models/incremental", files: [], available: true, description: "MobileNetV2模型 (准确率: 81.13%)" },
    { name: "efficientnet_b0", path: "models/incremental_efficientnet_b0", files: [], available: true, description: "EfficientNet-B0模型 (准确率: 64.15%)" },
    { name: "resnet50", path: "models/incremental_resnet50", files: [], available: true, description: "ResNet50模型 (准确率: 52.83%)" },
    // { name: "augmented_training", path: "models/augmented_training", files: [], available: false, description: "增强训练模型" },
    // { name: "arona_plana", path: "models/arona_plana", files: [], available: false, description: "阿罗娜普拉娜模型" },
    // { name: "arona_plana_efficientnet", path: "models/arona_plana_efficientnet", files: [], available: false, description: "EfficientNet模型" },
    // { name: "arona_plana_resnet18", path: "models/arona_plana_resnet18", files: [], available: false, description: "ResNet18模型" },
    // { name: "optimized", path: "models/optimized", files: [], available: false, description: "优化模型" }
  ]);
  const [inputText, setInputText] = useState<string>("");
  const [showUploadOptions, setShowUploadOptions] = useState(false);
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false); // 默认隐藏侧边栏
  const [showModelSelect, setShowModelSelect] = useState(false); // 移动端模型选择
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMountedRef = useRef(false);

  // 使用历史记录Hook
  const { history, loadHistory, addToHistory, clearHistory } = useHistory();

  // 组件挂载时执行
  useEffect(() => {
    isMountedRef.current = true;
    // 立即执行loadModels函数
    (async () => {
      await loadModels();
    })();
    loadHistory();
  }, [loadHistory]);

  // 主题切换效果
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      document.body.style.setProperty('--background', '#1a1a2e');
      document.body.style.setProperty('--foreground', '#f5f5f5');
      document.body.style.setProperty('--primary', '#ff6b9d');
      document.body.style.setProperty('--primary-hover', '#ff4785');
      document.body.style.setProperty('--secondary', '#4ecdc4');
      document.body.style.setProperty('--accent', '#45b7d1');
      document.body.style.setProperty('--danger', '#ff5252');
      document.body.style.setProperty('--warning', '#ffb74d');
      document.body.style.setProperty('--info', '#64b5f6');
      document.body.style.setProperty('--success', '#66bb6a');
      document.body.style.setProperty('--border', '#2c3e50');
      document.body.style.setProperty('--border-light', '#34495e');
      document.body.style.setProperty('--border-dark', '#1e293b');
      document.body.style.setProperty('--text-primary', '#f5f5f5');
      document.body.style.setProperty('--text-secondary', '#e0e0e0');
      document.body.style.setProperty('--text-light', '#bdbdbd');
      document.body.style.setProperty('--text-placeholder', '#9e9e9e');
      document.body.style.setProperty('--card-bg', '#2c3e50');
      document.body.style.setProperty('--card-hover', '#34495e');
      document.body.style.setProperty('--gradient-start', '#ff6b9d');
      document.body.style.setProperty('--gradient-end', '#4ecdc4');
      document.body.style.setProperty('--glass-bg', 'rgba(44, 62, 80, 0.85)');
      document.body.style.setProperty('--glass-border', 'rgba(255, 107, 157, 0.2)');
      document.body.style.backgroundImage = 'radial-gradient(circle at 15% 50%, rgba(255, 107, 157, 0.08) 0%, transparent 25%), radial-gradient(circle at 85% 30%, rgba(78, 205, 196, 0.08) 0%, transparent 25%)';
    } else {
      document.documentElement.classList.remove('dark');
      document.body.style.setProperty('--background', '#fef7ff');
      document.body.style.setProperty('--foreground', '#1a1a2e');
      document.body.style.setProperty('--primary', '#ff6b9d');
      document.body.style.setProperty('--primary-hover', '#ff4785');
      document.body.style.setProperty('--secondary', '#4ecdc4');
      document.body.style.setProperty('--accent', '#45b7d1');
      document.body.style.setProperty('--danger', '#ff5252');
      document.body.style.setProperty('--warning', '#ffb74d');
      document.body.style.setProperty('--info', '#64b5f6');
      document.body.style.setProperty('--success', '#66bb6a');
      document.body.style.setProperty('--border', '#e8eaf6');
      document.body.style.setProperty('--border-light', '#c5cae9');
      document.body.style.setProperty('--border-dark', '#9fa8da');
      document.body.style.setProperty('--text-primary', '#1a1a2e');
      document.body.style.setProperty('--text-secondary', '#2c3e50');
      document.body.style.setProperty('--text-light', '#4a5568');
      document.body.style.setProperty('--text-placeholder', '#78909c');
      document.body.style.setProperty('--card-bg', '#ffffff');
      document.body.style.setProperty('--card-hover', '#f5f5f5');
      document.body.style.setProperty('--gradient-start', '#ff6b9d');
      document.body.style.setProperty('--gradient-end', '#4ecdc4');
      document.body.style.setProperty('--glass-bg', 'rgba(255, 255, 255, 0.85)');
      document.body.style.setProperty('--glass-border', 'rgba(255, 107, 157, 0.2)');
      document.body.style.backgroundImage = 'radial-gradient(circle at 15% 50%, rgba(255, 107, 157, 0.08) 0%, transparent 25%), radial-gradient(circle at 85% 30%, rgba(78, 205, 196, 0.08) 0%, transparent 25%)';
    }
  }, [darkMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 快捷键功能
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+H 打开/关闭历史记录
      if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
        e.preventDefault();
        setShowHistory(!showHistory);
      }
      
      // Esc 关闭上传选项
      if (e.key === 'Escape') {
        setShowUploadOptions(false);
        setIsDragging(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showHistory]);

  const loadModels = async () => {
    // 直接使用默认模型列表，确保所有模型都显示
    const defaultModels = [
      { name: "efficientnet_b0", path: "models/incremental_efficientnet_b0", description: "EfficientNet-B0模型 (准确率: 88.68%)", available: true, files: [] },
      { name: "efficientnet_b3", path: "models/incremental_efficientnet_b3", description: "EfficientNet-B3模型 (准确率: 88.68%)", available: true, files: [] },
      { name: "mobilenet_v2", path: "models/incremental", description: "MobileNetV2模型 (准确率: 81.13%)", available: true, files: [] },
      { name: "resnet50", path: "models/incremental_resnet50", description: "ResNet50模型 (准确率: 47.17%)", available: true, files: [] },
      { name: "default", path: "", description: "默认分类模型", available: true, files: [] }
    ];
    console.log('使用默认模型列表:', defaultModels);
    // 确保组件仍然挂载
    if (isMountedRef.current) {
      setModels(defaultModels);
      if (defaultModels.length > 0) {
        setSelectedModel(defaultModels[0].name);
      }
    }
  };

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // 压缩图片
      const reader = new FileReader();
      reader.onloadend = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (ctx) {
            // 计算压缩后的尺寸
            const maxWidth = 800;
            const maxHeight = 800;
            let width = img.width;
            let height = img.height;
            
            if (width > maxWidth) {
              height = (height * maxWidth) / width;
              width = maxWidth;
            }
            
            if (height > maxHeight) {
              width = (width * maxHeight) / height;
              height = maxHeight;
            }
            
            canvas.width = width;
            canvas.height = height;
            
            // 绘制压缩后的图片
            ctx.drawImage(img, 0, 0, width, height);
            
            // 获取压缩后的图片数据
            const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.8);
            
            // 创建压缩后的文件
            const byteString = atob(compressedDataUrl.split(',')[1]);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
              ia[i] = byteString.charCodeAt(i);
            }
            const compressedFile = new File([ab], file.name, { type: 'image/jpeg' });
            
            setSelectedImage(compressedFile);
            setImagePreview(compressedDataUrl);
          }
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        // 压缩图片
        const reader = new FileReader();
        reader.onloadend = (e) => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (ctx) {
              // 计算压缩后的尺寸
              const maxWidth = 800;
              const maxHeight = 800;
              let width = img.width;
              let height = img.height;
              
              if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
              }
              
              if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
              }
              
              canvas.width = width;
              canvas.height = height;
              
              // 绘制压缩后的图片
              ctx.drawImage(img, 0, 0, width, height);
              
              // 获取压缩后的图片数据
              const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.8);
              
              // 创建压缩后的文件
              const byteString = atob(compressedDataUrl.split(',')[1]);
              const ab = new ArrayBuffer(byteString.length);
              const ia = new Uint8Array(ab);
              for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
              }
              const compressedFile = new File([ab], file.name, { type: 'image/jpeg' });
              
              setSelectedImage(compressedFile);
              setImagePreview(compressedDataUrl);
            }
          };
          img.src = e.target?.result as string;
        };
        reader.readAsDataURL(file);
      }
    }
  }, []);

  const classifyImage = async (imageData: string, multiRole: boolean = false): Promise<any> => {
    try {
      console.log("开始classifyImage函数");
      console.log("imageData长度:", imageData.length);
      console.log("imageData前100个字符:", imageData.substring(0, 100));

      // 直接从base64字符串创建Blob对象
      const base64Data = imageData.split(',')[1];
      console.log("base64Data长度:", base64Data.length);
      console.log("base64Data前100个字符:", base64Data.substring(0, 100));

      const byteCharacters = atob(base64Data);
      console.log("byteCharacters长度:", byteCharacters.length);
      console.log("byteCharacters前100个字符:", byteCharacters.substring(0, 100));

      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      console.log("byteNumbers长度:", byteNumbers.length);
      console.log("byteNumbers前10个元素:", byteNumbers.slice(0, 10));

      const byteArray = new Uint8Array(byteNumbers);
      console.log("byteArray长度:", byteArray.length);
      console.log("byteArray前10个元素:", byteArray.slice(0, 10));

      const blob = new Blob([byteArray], { type: 'image/jpeg' });
      console.log("blob大小:", blob.size);
      console.log("blob类型:", blob.type);

      const file = new File([blob], "uploaded_image.jpg", { type: "image/jpeg" });
      console.log("file名称:", file.name);
      console.log("file大小:", file.size);
      console.log("file类型:", file.type);

      const formData = new FormData();
      formData.append("file", file);
      
      if (!multiRole) {
        formData.append("use_model", "true");
        formData.append("use_attributes", "true");
        formData.append("model_name", selectedModel);
        formData.append("cache_bypass", "false");
      } else {
        formData.append("model_name", selectedModel);
        formData.append("cache_bypass", "false");
      }

      console.log("FormData创建完成");
      console.log("selectedModel:", selectedModel);

      const apiUrl = multiRole 
        ? "http://localhost:8000/api/classify/multi-role" 
        : "http://localhost:8000/api/classify";
      console.log(`准备发送API请求到 ${apiUrl}`);

      try {
        console.log("开始发送API请求");
        
        // 使用axios发送请求
        const response = await axios.post(apiUrl, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: multiRole ? 300000 : 180000 // 多角色识别300秒超时，单角色识别180秒超时
        });

        console.log("API响应状态:", response.status);
        console.log("API响应数据:", response.data);
        return response.data;
      } catch (error) {
        console.error("API请求失败:", error);
        if (axios.isAxiosError(error)) {
          console.error("Axios错误详情:", {
            message: error.message,
            code: error.code,
            status: error.response?.status,
            data: error.response?.data,
            config: error.config
          });
          if (error.code === 'ECONNABORTED') {
            throw new Error('API请求超时，请检查服务器是否正常运行');
          }
          throw new Error(error.response?.data?.error || error.message || "API请求失败");
        }
        throw error;
      }
    } catch (error) {
      console.error("分类错误详情:", error);
      console.error("错误堆栈:", error instanceof Error ? error.stack : null);
      throw new Error(error instanceof Error ? error.message : "分类过程中发生未知错误");
    }
  };

  const handleSend = useCallback(async () => {
    if ((!inputText.trim() && !selectedImage) || isProcessing) return;

    console.log("开始处理图片");
    console.log("selectedImage:", selectedImage);
    console.log("imagePreview:", imagePreview);

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputText,
      image: imagePreview || undefined,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    const currentImage = selectedImage;
    const currentImagePreview = imagePreview;
    removeImage();

    if (currentImage && currentImagePreview) {
      console.log("有图片需要处理");
      setIsProcessing(true);

      const processingMessageId = `processing_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
      const processingMessage: Message = {
        id: processingMessageId,
        role: "assistant",
        content: "",
        thoughts: ["正在分析图片特征...", "提取角色关键信息...", "匹配数据库中的角色..."],
        isThinkingFinished: false,
        timestamp: Date.now(),
      };
      console.log("创建processingMessage，ID:", processingMessageId);
      setMessages((prev) => {
        const updatedMessages = [...prev, processingMessage];
        console.log("添加processingMessage后的消息数量:", updatedMessages.length);
        console.log("消息列表中的最后一个消息ID:", updatedMessages[updatedMessages.length - 1].id);
        return updatedMessages;
      });

      try {
        console.log("调用classifyImage函数");
        console.log("useMultiRole:", useMultiRole);
        console.log("selectedModel:", selectedModel);
        
        const startTime = Date.now();
        console.log("开始API请求，时间:", startTime);
        
        const result = await classifyImage(currentImagePreview, useMultiRole);
        
        const endTime = Date.now();
        console.log("API请求完成，耗时:", endTime - startTime, "ms");
        console.log('API返回的完整结果:', result);
        console.log('result类型:', typeof result);
        
        // 检查result是否为有效对象
        if (!result || typeof result !== 'object') {
          throw new Error('API返回的数据格式不正确');
        }
        
        console.log('text_detections字段:', result.text_detections);
        console.log('text_detections类型:', typeof result.text_detections);
        console.log('text_detections长度:', result.text_detections ? result.text_detections.length : 0);

        let assistantMessage: Message;
        
        if ((useMultiRole || result.detection_mode === 'multi_role') && result.roles && result.roles.length > 0) {
          // 多角色识别结果
          console.log('使用多角色识别结果');
          const rolesList = result.roles.map((role: any) => {
            return `${role.role} (相似度: ${(role.similarity * 100).toFixed(1)}%)`;
          }).join('、');
          
          assistantMessage = {
            id: `assistant_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
            role: "assistant",
            content: `识别完成！共检测到 ${result.roles.length} 个角色：${rolesList}`,
            classification: {
              role: "multiple_roles",
              similarity: 1.0,
              confidence: "high",
            },
            attributes: [],
            text_detections: result.text_detections || [],
            multi_roles: result.roles,
            thoughts: ["正在分析图片特征...", "检测多个角色...", "对每个角色进行分类...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        } else {
          // 单角色识别结果
          console.log('使用单角色识别结果');
          assistantMessage = {
            id: `assistant_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
            role: "assistant",
            content: `识别完成！识别结果：${result.role || "未知角色"}，相似度：${(result.similarity * 100).toFixed(1)}%`,
            classification: {
              role: result.role || "未知角色",
              similarity: result.similarity || 0,
              confidence: (result.similarity || 0) >= 0.8 ? "high" : (result.similarity || 0) >= 0.5 ? "medium" : "low",
            },
            attributes: result.attributes || [],
            text_detections: result.text_detections || [],
            ai_predicted_role: result.ai_predicted_role || "未知角色",
            thoughts: ["正在分析图片特征...", "提取角色关键信息...", "匹配数据库中的角色...", "识别完成！"],
            isThinkingFinished: true,
            timestamp: Date.now(),
          };
        }

        console.log("更新消息列表，替换processingMessage为assistantMessage");
        console.log("processingMessageId:", processingMessageId);
        console.log("assistantMessage:", assistantMessage);
        
        // 直接创建新的消息列表，确保processingMessage被替换
        setMessages((prev) => {
          console.log("当前消息列表长度:", prev.length);
          console.log("当前消息列表ID:", prev.map(m => m.id));
          
          const newMessages = [];
          let processingMessageFound = false;
          
          for (let i = 0; i < prev.length; i++) {
            if (prev[i].id !== processingMessageId) {
              newMessages.push(prev[i]);
            } else {
              processingMessageFound = true;
              console.log("找到processingMessage，ID:", prev[i].id);
            }
          }
          
          if (!processingMessageFound) {
            console.warn("未找到processingMessage，ID:", processingMessageId);
          }
          
          newMessages.push(assistantMessage);
          console.log("新消息列表长度:", newMessages.length);
          console.log("新消息列表ID:", newMessages.map(m => m.id));
          return newMessages;
        });
        addToHistory(assistantMessage);
      } catch (error) {
        console.error("Classification error:", error);
        console.error("错误堆栈:", error instanceof Error ? error.stack : null);
        
        const errorMessage: Message = {
          id: `error_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
          role: "assistant",
          content: `抱歉，识别过程中出现错误：${error instanceof Error ? error.message : "未知错误"}，请重试。`,
          timestamp: Date.now(),
        };
        console.log("更新消息列表，替换processingMessage为errorMessage");
        console.log("processingMessageId:", processingMessageId);
        console.log("errorMessage:", errorMessage);
        
        // 直接创建新的消息列表，确保processingMessage被替换
        setMessages((prev) => {
          console.log("当前消息列表长度:", prev.length);
          console.log("当前消息列表ID:", prev.map(m => m.id));
          
          const newMessages = [];
          let processingMessageFound = false;
          
          for (let i = 0; i < prev.length; i++) {
            if (prev[i].id !== processingMessageId) {
              newMessages.push(prev[i]);
            } else {
              processingMessageFound = true;
              console.log("找到processingMessage，ID:", prev[i].id);
            }
          }
          
          if (!processingMessageFound) {
            console.warn("未找到processingMessage，ID:", processingMessageId);
          }
          
          newMessages.push(errorMessage);
          console.log("新消息列表长度:", newMessages.length);
          console.log("新消息列表ID:", newMessages.map(m => m.id));
          return newMessages;
        });
      } finally {
        console.log("设置isProcessing为false");
        setIsProcessing(false);
      }
    } else if (inputText.trim()) {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "请上传动漫角色图片，我会帮你识别角色名称。",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage, classifyImage, addToHistory]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const getConfidenceText = useCallback((confidence: string) => {
    switch (confidence) {
      case "high":
        return "高置信度";
      case "medium":
        return "中等置信度";
      case "low":
        return "低置信度";
      default:
        return "未知";
    }
  }, []);

  const handleCopyMessage = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopySuccess("复制成功！");
      setTimeout(() => setCopySuccess(null), 3000);
    } catch (err) {
      console.error("复制失败:", err);
    }
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

  return (
    <div 
      className="flex flex-col h-screen font-sans overflow-hidden bg-gradient-to-br from-[#f8fafc] to-[#e2e8f0]"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* 拖拽上传覆盖层 */}
      {isDragging && (
        <div className="fixed inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-[9999] border-2 border-dashed border-[#3b82f6] rounded-lg animate-pulse-glow">
          <div className="text-center p-8 glass rounded-xl shadow-2xl transform transition-transform hover:scale-105">
            <Upload className="h-16 w-16 mx-auto mb-4 text-[#3b82f6] animate-bounce" />
            <h3 className="text-xl font-semibold mb-2 text-[#1e293b] animate-fade-in">拖拽图片到这里</h3>
            <p className="text-[#64748b] animate-fade-in">松开鼠标即可上传图片进行识别</p>
          </div>
        </div>
      )}

      {/* 移动端顶部导航栏 */}
      <div className="md:hidden h-14 border-b border-[#e2e8f0] flex items-center justify-between px-4 flex-shrink-0 glass shadow-md">
        <div className="flex items-center gap-2.5">
          <button 
            className="p-1.5 rounded-lg hover:bg-[#f1f5f9] transition-all duration-300 transform hover:scale-105"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <Menu size={18} className="text-[#64748b]" />
          </button>
          <h2 className="text-base font-semibold gradient-text">动漫角色识别</h2>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="appearance-none pl-3 pr-8 py-1.5 border border-[#cbd5e1] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3b82f6] focus:border-transparent bg-white text-[#1e293b] text-sm transition-all duration-300 hover:border-[#3b82f6]/50"
            >
              {models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.description || (model.name === "default" ? "默认模型" : model.name)}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-[#94a3b8]">
              <svg className="h-4 w-4 transition-transform duration-300 hover:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
          <button className="p-1.5 rounded-lg hover:bg-[#f1f5f9] transition-all duration-300 transform hover:scale-105">
            <Moon className="h-4 w-4 text-[#64748b]" />
          </button>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧边栏（仅在中等及以上屏幕显示） */}
        <div className={`fixed md:relative top-14 left-0 z-30 flex flex-col items-center lg:items-start w-16 lg:w-56 h-[calc(100%-3.5rem)] glass border-r border-[#e2e8f0] p-4 transition-all duration-300 transform ${showSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 shadow-lg`}>
          {/* Logo */}
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 gradient-bg rounded-lg shadow-lg animate-pulse-glow transform hover:scale-110 transition-transform duration-300">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-lg font-bold gradient-text hidden lg:block animate-fade-in">动漫角色识别</h1>
          </div>

          {/* 导航菜单 */}
          <nav className="flex-1 w-full">
            <ul className="space-y-2">
              <li>
                <button className="flex items-center gap-3 w-full px-4 py-3 rounded-lg bg-[#3b82f6]/10 text-[#3b82f6] font-medium hover:bg-[#3b82f6]/20 transition-all duration-300 transform hover:translate-x-1">
                  <Search className="h-4 w-4" />
                  <span className="hidden lg:block">识别</span>
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setShowHistory(!showHistory)}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-lg hover:bg-[#f1f5f9] text-[#64748b] transition-all duration-300 transform hover:translate-x-1"
                >
                  <Layers className="h-4 w-4" />
                  <span className="hidden lg:block">历史记录</span>
                  {history.length > 0 && (
                    <span className="ml-auto bg-[#3b82f6] text-white text-xs px-2 py-0.5 rounded-full animate-pulse">
                      {history.length}
                    </span>
                  )}
                </button>
              </li>
              <li>
                <button className="flex items-center gap-3 w-full px-4 py-3 rounded-lg hover:bg-[#f1f5f9] text-[#64748b] transition-all duration-300 transform hover:translate-x-1">
                  <Settings className="h-4 w-4" />
                  <span className="hidden lg:block">设置</span>
                </button>
              </li>
              <li>
                <button className="flex items-center gap-3 w-full px-4 py-3 rounded-lg hover:bg-[#f1f5f9] text-[#64748b] transition-all duration-300 transform hover:translate-x-1">
                  <HelpCircle className="h-4 w-4" />
                  <span className="hidden lg:block">帮助</span>
                </button>
              </li>
            </ul>
          </nav>

          {/* 底部设置 */}
          <div className="w-full mt-auto">
            <button className="flex items-center gap-3 w-full px-4 py-3 rounded-lg hover:bg-[#f1f5f9] text-[#64748b] transition-all duration-300 transform hover:translate-x-1">
              <Moon className="h-4 w-4" />
              <span className="hidden lg:block">深色模式</span>
            </button>
          </div>
        </div>

        {/* 主内容区域 */}
        <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] overflow-hidden ml-0 md:ml-16 lg:ml-56 bg-gradient-to-br from-[#f0f9ff] to-[#e0f2fe]">
          {/* 顶部导航栏（在所有屏幕显示） */}
          <div className="flex h-16 border-b border-[#e2e8f0] items-center justify-between px-6 flex-shrink-0 glass shadow-md">
            <div className="flex items-center gap-4">
              <Zap className="h-6 w-6 gradient-text" />
              <h2 className="text-xl font-semibold gradient-text">{showHistory ? "历史记录" : "动漫角色识别"}</h2>
            </div>
            <div className="flex items-center gap-4">
              {/* 模型选择下拉框（仅在中等及以上屏幕显示） */}
              <div className="hidden sm:flex items-center space-x-4">
                <div className="relative">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="appearance-none pl-4 pr-10 py-3 border border-[#cbd5e1] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3b82f6] focus:border-transparent bg-white text-[#1e293b] text-sm transition-all duration-300 hover:border-[#3b82f6]/50 shadow-sm min-w-[180px]"
                  >
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.description || (model.name === "default" ? "默认模型" : model.name)}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-[#64748b]">
                    <svg className="h-4 w-4 transition-transform duration-300 hover:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="multi-role"
                    checked={useMultiRole}
                    onChange={(e) => setUseMultiRole(e.target.checked)}
                    className="w-4 h-4 rounded border-[#cbd5e1] text-[#3b82f6] focus:ring-[#3b82f6]/50"
                  />
                  <label htmlFor="multi-role" className="text-sm font-medium text-[#1e293b]">
                    多角色识别
                  </label>
                </div>
              </div>
              
              {/* 移动端模型选择按钮 */}
              <div className="sm:hidden">
                <button
                  onClick={() => setShowModelSelect(!showModelSelect)}
                  className="flex items-center gap-2 px-4 py-3 border border-[#cbd5e1] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3b82f6] focus:border-transparent bg-white text-[#1e293b] text-sm transition-all duration-300 hover:border-[#3b82f6]/50 shadow-sm"
                >
                  <Settings className="h-4 w-4 text-[#3b82f6]" />
                  <span>模型</span>
                </button>
                {showModelSelect && (
                  <div className="fixed top-24 right-6 glass rounded-2xl shadow-2xl border border-border-light p-2 z-[9999] transform transition-all duration-300 animate-slide-up w-48">
                    {models.map((model) => (
                      <button
                        key={model.name}
                        onClick={() => {
                          setSelectedModel(model.name);
                          setShowModelSelect(false);
                        }}
                        className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 ${selectedModel === model.name ? "bg-primary/10 text-primary" : "hover:bg-card-hover"}`}
                      >
                        {model.description || (model.name === "default" ? "默认模型" : model.name)}
                      </button>
                    ))}
                    <div className="border-t border-border-light my-2"></div>
                    <div className="flex items-center px-4 py-3">
                      <input
                        type="checkbox"
                        id="mobile-multi-role"
                        checked={useMultiRole}
                        onChange={(e) => setUseMultiRole(e.target.checked)}
                        className="w-4 h-4 rounded border-border text-primary focus:ring-primary/50"
                      />
                      <label htmlFor="mobile-multi-role" className="ml-2 text-sm font-medium">
                        多角色识别
                      </label>
                    </div>
                  </div>
                )}
              </div>
              
              {/* 主题切换 */}
              <button 
                onClick={() => setDarkMode(!darkMode)}
                className="p-3 rounded-lg hover:bg-card-hover transition-all duration-300 transform hover:scale-105"
                title={darkMode ? "切换到浅色模式" : "切换到深色模式"}
              >
                {darkMode ? <Sun className="h-5 w-5 text-warning" /> : <Moon className="h-5 w-5 text-primary" />}
              </button>
              
              {/* 历史记录按钮 */}
              <button 
                onClick={() => setShowHistory(!showHistory)}
                className="p-3 rounded-lg hover:bg-card-hover transition-all duration-300 transform hover:scale-105"
                title="查看历史记录"
              >
                <Layers className="h-5 w-5 text-primary" />
              </button>
              
              {showHistory && (
                <button 
                  onClick={clearHistory}
                  className="p-3 rounded-lg hover:bg-card-hover transition-all duration-300 transform hover:scale-105"
                  title="清空历史记录"
                >
                  <Trash2 className="h-5 w-5 text-danger" />
                </button>
              )}
            </div>
          </div>

          {/* 消息列表或历史记录 */}
          <div className="flex-1 overflow-y-auto p-6 md:p-8 scroll-smooth">
            {copySuccess && (
              <div className="fixed top-24 right-6 left-6 md:left-auto md:right-6 bg-[#10b981] text-white px-6 py-4 rounded-xl shadow-lg animate-slide-up z-[9999] transform transition-all duration-300 hover:scale-105">
                {copySuccess}
              </div>
            )}
            
            {/* 历史记录显示 */}
            {showHistory ? (
              <div className="max-w-3xl mx-auto space-y-8 pb-16">
                {history.length === 0 ? (
                  <div className="glass p-10 rounded-xl text-center shadow-lg transform transition-all duration-300 hover:scale-[1.02]">
                    <Layers className="h-16 w-16 mx-auto mb-6 text-text-light animate-float" />
                    <h3 className="text-xl font-semibold text-text-primary mb-3 animate-fade-in">暂无历史记录</h3>
                    <p className="text-text-light animate-fade-in">上传图片进行识别后，结果将显示在这里</p>
                  </div>
                ) : (
                  history.map((record, idx) => (
                    <div key={idx} className="glass border border-border-light rounded-xl p-6 shadow-lg animate-slide-up transition-all duration-300 hover:shadow-xl hover:border-primary/30">
                      <div className="flex items-center justify-between mb-6">
                        <span className="text-sm text-text-light">
                          {new Date(record.timestamp).toLocaleString()}
                        </span>
                        <span className={`text-sm px-3 py-1.5 rounded-full border ${record.classification?.confidence === "high" ? "bg-success/10 text-success border-success/30" : record.classification?.confidence === "medium" ? "bg-warning/10 text-warning border-warning/30" : "bg-danger/10 text-danger border-danger/30"}`}>
                          {record.classification && getConfidenceText(record.classification.confidence)}
                        </span>
                      </div>
                      {record.classification && (
                        <div className="mb-6">
                          <div className="text-xl font-bold text-text-primary mb-3 animate-fade-in">{record.classification.role}</div>
                          <div className="text-sm text-text-light mb-4">相似度: {(record.classification.similarity * 100).toFixed(1)}%</div>
                          <div className="progress-bar h-3 mb-6 bg-border-light rounded-full overflow-hidden">
                            <div
                              className={`progress-bar-fill h-3 rounded-full transition-all duration-1000 ease-out ${record.classification.confidence === "high" ? "bg-success" : record.classification.confidence === "medium" ? "bg-warning" : "bg-danger"}`}
                              style={{ width: `${record.classification.similarity * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      <div className="flex justify-end gap-3">
                        <button
                          onClick={() => {
                            setShowHistory(false);
                            // 可以选择将历史记录重新添加到消息列表中
                          }}
                          className="px-4 py-2 bg-primary/10 text-primary rounded-lg text-sm hover:bg-primary/20 transition-all duration-300 transform hover:scale-105"
                        >
                          查看详情
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : (
              /* 消息列表显示 */
              <div className="max-w-3xl mx-auto space-y-8 pb-16">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"} animate-slide-up`}>
                    {msg.role === "assistant" && (
                      <div className="w-12 h-12 rounded-full flex-shrink-0 flex items-center justify-center shadow-lg gradient-bg animate-float">
                        <Bot size={24} className="text-white" />
                      </div>
                    )}

                    <div className={`max-w-[85%] sm:max-w-[80%] md:max-w-[75%] rounded-2xl px-6 py-5 text-sm leading-6 shadow-lg message-bubble transition-all duration-300 ${msg.role === "user" ? "gradient-bg text-white" : "glass border border-border-light text-text-primary shadow-inner"} hover:shadow-xl transform hover:scale-[1.01] hover:-translate-y-1`}>
                      {msg.role === "assistant" ? (
                        <div className="flex flex-col gap-3">
                          {/* 思考过程展示 */}
                          {msg.thoughts && msg.thoughts.length > 0 && (
                            <div className="mb-3">
                              <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full transition-all cursor-pointer w-fit select-none border border-primary/30 bg-primary/10 text-primary shadow-sm hover:bg-primary/20 transform hover:scale-105">
                                <div className="relative">
                                  <Sparkles size={14} className="text-primary" />
                                  {!msg.isThinkingFinished && (
                                    <span className="absolute -top-1 -right-1 flex h-2 w-2">
                                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                                      <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary"></span>
                                    </span>
                                  )}
                                </div>
                                <span className="font-medium whitespace-nowrap">
                                  {msg.isThinkingFinished ? "思考完成" : "正在思考..."}
                                </span>
                              </div>
                              <div className="relative pl-3 border-l-2 border-border-light py-2">
                                <div className="text-sm text-text-light leading-relaxed font-serif italic whitespace-pre-wrap">
                                  {msg.thoughts.join("\n")}
                                  {!msg.isThinkingFinished && (
                                    <span className="inline-flex items-center gap-1 ml-1">
                                      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
                                      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                                      <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* 回复正文 */}
                          {msg.content && (
                            <div className="prose prose-sm max-w-none animate-fade-in">
                              <p className="whitespace-pre-wrap">{msg.content}</p>
                            </div>
                          )}

                          {/* 识别结果 */}
                          {msg.classification && (
                            <div className="mt-3 pt-3 border-t border-border-light">
                              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                                <span className="text-xs font-medium text-text-light">识别结果</span>
                                <span className={`text-xs px-3 py-1.5 rounded-full border ${msg.classification.confidence === "high" ? "bg-success/10 text-success border-success/30" : msg.classification.confidence === "medium" ? "bg-warning/10 text-warning border-warning/30" : "bg-danger/10 text-danger border-danger/30"}`}>
                                  {getConfidenceText(msg.classification.confidence)}
                                </span>
                              </div>
                              <div className="glass p-5 rounded-xl shadow-lg transition-all duration-300 hover:shadow-xl border border-border-light">
                                <div className="flex flex-col sm:flex-row items-center space-x-4">
                                  <div className="flex-1 w-full sm:w-auto">
                                    <div className="text-lg font-bold text-text-primary animate-fade-in">{msg.classification.role}</div>
                                    <div className="text-sm text-text-light mt-2 animate-fade-in">相似度: {(msg.classification.similarity * 100).toFixed(1)}%</div>
                                    <div className="mt-3">
                                      <div className="progress-bar h-2.5 bg-border-light rounded-full overflow-hidden">
                                        <div
                                          className={`progress-bar-fill h-2.5 rounded-full transition-all duration-1000 ease-out ${msg.classification.confidence === "high" ? "bg-success" : msg.classification.confidence === "medium" ? "bg-warning" : "bg-danger"}`}
                                          style={{ width: `${msg.classification.similarity * 100}%` }}
                                        ></div>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center shadow-md mt-4 sm:mt-0 animate-float">
                                    <CheckCircle
                                      className={`w-10 h-10 ${msg.classification.confidence === "high" ? "text-success" : msg.classification.confidence === "medium" ? "text-warning" : "text-danger"}`}
                                    />
                                  </div>
                                </div>
                                
                                {/* 属性标签展示 */}
                                {msg.attributes && msg.attributes.length > 0 && (
                                  <div className="mt-4 pt-4 border-t border-border-light">
                                    <div className="text-xs font-medium text-text-light mb-3">属性标签</div>
                                    <div className="flex flex-wrap gap-2">
                                      {msg.attributes.slice(0, 10).map((attr, idx) => (
                                        <span
                                          key={idx}
                                          className="px-3 py-1.5 bg-card-hover text-text-light text-xs rounded-full border border-border-light hover:bg-border-light transition-all duration-300"
                                        >
                                          {attr.tag} ({(attr.confidence * 100).toFixed(0)}%)
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                
                                {/* AI预测角色展示 */}
                                <div className="mt-4 pt-4 border-t border-border-light">
                                  <div className="text-xs font-medium text-text-light mb-3">AI预测角色</div>
                                  <div className="flex items-center gap-2">
                                    <span className="px-3 py-1.5 bg-warning/10 text-warning text-xs rounded-full border border-warning/30 hover:bg-warning/20 transition-all duration-300">
                                      {msg.ai_predicted_role || "未知角色"}
                                    </span>
                                  </div>
                                </div>
                                
                                {/* 文本检测结果展示 */}
                                <div className="mt-4 pt-4 border-t border-border-light">
                                  <div className="text-xs font-medium text-text-light mb-3">文本检测</div>
                                  <div className="flex flex-wrap gap-2">
                                    {msg.text_detections && msg.text_detections.length > 0 ? (
                                      msg.text_detections.slice(0, 10).map((text, idx) => (
                                        <span
                                          key={idx}
                                          className="px-3 py-1.5 bg-primary/10 text-primary text-xs rounded-full border border-primary/30 hover:bg-primary/20 transition-all duration-300"
                                        >
                                          {text.text} ({(text.confidence * 100).toFixed(0)}%)
                                        </span>
                                      ))
                                    ) : (
                                      <span className="text-xs text-text-light">未检测到文本</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              
                              {/* 识别时间 */}
                              <div className="mt-3 text-xs text-text-light flex items-center gap-2 justify-end">
                                <Clock className="h-3 w-3" />
                                <span>{msg.timestamp}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="flex flex-col gap-2">
                          {msg.image && (
                            <div className="mb-2">
                              <div className="relative transform transition-all duration-300 hover:scale-[1.02]">
                                <img src={msg.image} alt="Uploaded" className="max-w-xs max-h-64 h-auto rounded-xl shadow-lg animate-fade-in object-contain" />
                                <div className="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded-lg backdrop-blur-sm">{selectedImage?.name}</div>
                              </div>
                            </div>
                          )}
                          <div className="whitespace-pre-wrap">{msg.content}</div>
                          
                          {/* 发送时间 */}
                          <div className="text-xs text-white/70 flex items-center gap-2 justify-end mt-1">
                            <Clock className="h-3 w-3" />
                            <span>{msg.timestamp}</span>
                          </div>
                        </div>
                      )}

                      {/* 复制和下载按钮 */}
                      {msg.content && (
                        <div className="flex justify-end gap-2 mt-3">
                          <button
                            onClick={() => handleCopyMessage(msg.content)}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all duration-300 ${msg.role === "user" ? "bg-white/10 hover:bg-white/20 text-white" : "bg-[#f1f5f9] hover:bg-[#e2e8f0] text-[#64748b]"} transform hover:scale-105`}
                            title="复制消息内容"
                          >
                            <Copy size={14} />
                            <span className="hidden sm:inline">复制</span>
                          </button>
                          <button
                            onClick={() => handleDownloadMessage(msg.content, msg.role)}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all duration-300 ${msg.role === "user" ? "bg-white/10 hover:bg-white/20 text-white" : "bg-[#f1f5f9] hover:bg-[#e2e8f0] text-[#64748b]"} transform hover:scale-105`}
                            title="下载消息内容"
                          >
                            <Download size={14} />
                            <span className="hidden sm:inline">下载</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {msg.role === "user" && (
                      <div className="w-12 h-12 rounded-full flex-shrink-0 flex items-center justify-center shadow-lg bg-gradient-to-br from-secondary to-accent animate-float">
                        <User size={24} className="text-white" />
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} className="h-4" />
              </div>
            )}
          </div>

          {/* 输入区域 */}
          <div className="border-t border-border-light glass shadow-2xl">
            <div className="max-w-3xl mx-auto px-6 sm:px-8 lg:px-10 py-6">
              {/* 移除重复的模型选择下拉框，因为已经在顶部导航栏中添加了 */}
              
              {/* 移除预览图片显示 */}

              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0 relative">
                  <button
                    onClick={() => setShowUploadOptions(!showUploadOptions)}
                    className="p-4 rounded-full hover:bg-card-hover transition-all duration-300 transform hover:scale-110 hover:text-secondary shadow-md bg-gradient-to-br from-primary/10 to-secondary/10"
                    title="上传图片"
                  >
                    <Upload className="h-6 w-6 text-primary transition-colors duration-300 hover:text-secondary" />
                  </button>
                  {showUploadOptions && (
                    <div className="fixed bottom-32 left-6 right-6 md:left-auto md:right-6 md:w-64 glass rounded-2xl shadow-2xl border border-border-light p-4 z-[9999] transform transition-all duration-300 animate-slide-up">
                      <button
                        className="flex items-center px-6 py-3 hover:bg-card-hover rounded-xl w-full transition-all duration-300 transform hover:translate-x-1 shadow-sm"
                        onClick={() => {
                          fileInputRef.current?.click();
                          setShowUploadOptions(false);
                        }}
                      >
                        <ImageIcon className="h-5 w-5 mr-4 text-text-light transition-colors duration-300 hover:text-primary" />
                        <span className="text-sm text-text-primary transition-colors duration-300 hover:text-primary">上传图片</span>
                      </button>
                      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
                    </div>
                  )}
                </div>
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="输入消息或上传图片..."
                    className="w-full px-6 py-4 pr-16 glass border border-border-light rounded-xl focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent text-text-primary placeholder-text-placeholder input-glow transition-all duration-300 hover:border-primary/50 shadow-md text-sm"
                    disabled={isProcessing}
                  />
                  <button
                    onClick={() => setInputText("")}
                    className="absolute right-12 top-1/2 transform -translate-y-1/2 p-1.5 hover:bg-card-hover rounded-lg transition-colors duration-300 hover:scale-105 shadow-sm"
                    title="清空输入"
                    disabled={!inputText.trim() || isProcessing}
                  >
                    <X className={`h-5 w-5 transition-colors duration-300 ${inputText.trim() && !isProcessing ? "text-text-light hover:text-danger" : "text-text-light/50 cursor-not-allowed"}`} />
                  </button>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 hover:bg-card-hover rounded-lg transition-colors duration-300 hover:scale-110 shadow-sm"
                    disabled={isProcessing}
                    title="上传图片"
                  >
                    <ImageIcon className="h-5 w-5 text-text-light transition-colors duration-300 hover:text-primary" />
                  </button>
                </div>
                <div className="flex-shrink-0">
                  <button
                    onClick={handleSend}
                    disabled={(!inputText.trim() && !selectedImage) || isProcessing}
                    className={`btn-primary px-8 py-4 rounded-xl font-medium transition-all duration-300 flex items-center space-x-2 shadow-lg ${(!inputText.trim() && !selectedImage) || isProcessing ? "opacity-50 cursor-not-allowed" : "hover:shadow-xl hover:scale-105 hover:-translate-y-1"}`}
                  >
                    {isProcessing ? (
                      <>
                        <svg className="loading-spinner h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span className="hidden sm:inline">识别中</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-5 w-5 transition-transform duration-300 hover:rotate-12" />
                        <span className="hidden sm:inline">发送</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
              <div className="mt-4 text-xs text-text-light text-center transition-all duration-300 hover:text-text-secondary">按 Enter 发送，Shift + Enter 换行</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
