"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Bot, User, Image as ImageIcon, X, Sparkles, Upload, Copy, Download, CheckCircle, Menu, Layers, Trash2, Moon, Sun } from "lucide-react";
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
      document.body.style.backgroundColor = '#1a1a1a';
      document.body.style.color = '#e5e5e5';
    } else {
      document.documentElement.classList.remove('dark');
      document.body.style.backgroundColor = '#ffffff';
      document.body.style.color = '#333333';
    }
  }, [darkMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 快捷键功能
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Esc 关闭上传选项
      if (e.key === 'Escape') {
        setShowUploadOptions(false);
        setIsDragging(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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
    console.log("handleImageSelect 被调用，文件:", file);
    if (file) {
      console.log("开始处理文件:", file.name, file.type, file.size);
      // 压缩图片
      const reader = new FileReader();
      reader.onloadend = (e) => {
        console.log("FileReader 读取完成");
        const img = new Image();
        img.onload = () => {
          console.log("图片加载完成，尺寸:", img.width, "x", img.height);
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
            console.log("压缩完成，dataURL长度:", compressedDataUrl.length);
            
            // 创建压缩后的文件
            const byteString = atob(compressedDataUrl.split(',')[1]);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
              ia[i] = byteString.charCodeAt(i);
            }
            const compressedFile = new File([ab], file.name, { type: 'image/jpeg' });
            console.log("压缩文件创建完成:", compressedFile.name, compressedFile.size);
            
            setSelectedImage(compressedFile);
            setImagePreview(compressedDataUrl);
            console.log("状态已更新，selectedImage:", compressedFile, "imagePreview:", compressedDataUrl.substring(0, 50));
          } else {
            console.error("无法获取canvas上下文");
          }
        };
        img.onerror = () => {
          console.error("图片加载失败");
        };
        img.src = e.target?.result as string;
      };
      reader.onerror = () => {
        console.error("FileReader 读取失败");
      };
      reader.readAsDataURL(file);
    } else {
      console.log("没有选择文件");
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
    console.log("handleDrop 被调用，文件数量:", files?.length);
    if (files && files.length > 0) {
      const file = files[0];
      console.log("拖拽文件:", file.name, file.type, file.size);
      if (file.type.startsWith('image/')) {
        // 压缩图片
        const reader = new FileReader();
        reader.onloadend = (e) => {
          console.log("拖拽文件 FileReader 读取完成");
          const img = new Image();
          img.onload = () => {
            console.log("拖拽图片加载完成，尺寸:", img.width, "x", img.height);
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
              console.log("拖拽图片压缩完成，dataURL长度:", compressedDataUrl.length);
              
              // 创建压缩后的文件
              const byteString = atob(compressedDataUrl.split(',')[1]);
              const ab = new ArrayBuffer(byteString.length);
              const ia = new Uint8Array(ab);
              for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
              }
              const compressedFile = new File([ab], file.name, { type: 'image/jpeg' });
              console.log("拖拽压缩文件创建完成:", compressedFile.name, compressedFile.size);
              
              setSelectedImage(compressedFile);
              setImagePreview(compressedDataUrl);
              console.log("拖拽状态已更新，selectedImage:", compressedFile, "imagePreview:", compressedDataUrl.substring(0, 50));
            } else {
              console.error("拖拽无法获取canvas上下文");
            }
          };
          img.onerror = () => {
            console.error("拖拽图片加载失败");
          };
          img.src = e.target?.result as string;
        };
        reader.onerror = () => {
          console.error("拖拽FileReader 读取失败");
        };
        reader.readAsDataURL(file);
      } else {
        console.log("拖拽的文件不是图片类型:", file.type);
      }
    } else {
      console.log("没有拖拽文件");
    }
  }, []);

  const classifyImage = useCallback(async (imageData: string, multiRole: boolean = false): Promise<any> => {
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

      const apiUrl = "/api/classify";
      console.log(`准备发送API请求到 ${apiUrl}`);

      try {
        console.log("开始发送API请求");
        
        // 使用axios发送请求，不手动设置Content-Type，让axios自动设置
        const response = await axios.post(apiUrl, formData, {
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
  }, [selectedModel]);

  const handleSend = useCallback(async () => {
    console.log("点击发送按钮");
    console.log("inputText:", inputText);
    console.log("selectedImage:", selectedImage);
    console.log("isProcessing:", isProcessing);
    
    if ((!inputText.trim() && !selectedImage) || isProcessing) {
      console.log("发送按钮被禁用");
      return;
    }

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
        
        if (!result || typeof result !== 'object') {
          throw new Error('API返回的数据格式不正确');
        }
        
        console.log('text_detections字段:', result.text_detections);
        console.log('text_detections类型:', typeof result.text_detections);
        console.log('text_detections长度:', result.text_detections ? result.text_detections.length : 0);

        let assistantMessage: Message;
        
        if ((useMultiRole || result.detection_mode === 'multi_role') && result.roles && result.roles.length > 0) {
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
  }, [inputText, selectedImage, imagePreview, isProcessing, removeImage, classifyImage, addToHistory, useMultiRole, selectedModel]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

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
      className={`flex flex-col h-screen font-sans overflow-hidden ${darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* 拖拽上传覆盖层 */}
      {isDragging && (
        <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-[9999] border-2 border-dashed border-blue-500 rounded-lg">
          <div className="text-center p-8 bg-white dark:bg-gray-800 rounded-xl shadow-2xl">
            <Upload className="h-16 w-16 mx-auto mb-4 text-blue-500" />
            <h3 className="text-xl font-semibold mb-2">拖拽图片到这里</h3>
            <p className="text-gray-600 dark:text-gray-400">松开鼠标即可上传图片进行识别</p>
          </div>
        </div>
      )}

      {/* 顶部导航栏 */}
      <header className={`sticky top-0 z-50 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b`}>
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className={`p-2 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
              title="显示侧边栏"
            >
              <Menu className="h-5 w-5" />
            </button>
            <h1 className="text-xl font-semibold">动漫角色识别</h1>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
              title={darkMode ? "切换到浅色模式" : "切换到深色模式"}
            >
              {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* 侧边栏 */}
        <aside className={`fixed top-14 left-0 z-40 w-64 h-[calc(100vh-3.5rem)] ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-r transform transition-transform duration-300 ${showSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4">模型选择</h2>
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.name}
                  className={`flex items-center space-x-3 p-2 rounded-lg transition-colors cursor-pointer ${selectedModel === model.name ? (darkMode ? 'bg-blue-900/30 border-blue-700' : 'bg-blue-50 border-blue-200') : (darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100')} border`}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <div className={`w-2 h-2 rounded-full ${model.available ? 'bg-green-500' : 'bg-yellow-500'}`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{model.name}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{model.description}</p>
                  </div>
                  {selectedModel === model.name && (
                    <CheckCircle className="h-4 w-4 text-blue-500" />
                  )}
                </div>
              ))}
            </div>

            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2">识别设置</h3>
              <div className={`flex items-center justify-between p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors cursor-pointer`}>
                <div>
                  <p className="text-sm font-medium">多角色识别</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">同时识别图片中的多个角色</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useMultiRole}
                    onChange={(e) => setUseMultiRole(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className={`w-10 h-5 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-500`}></div>
                </label>
              </div>
            </div>

            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2">历史记录</h3>
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`w-full flex items-center justify-between p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
              >
                <div className="flex items-center space-x-2">
                  <Layers className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                  <span className="text-sm">查看历史记录</span>
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400">{history.length} 条</span>
              </button>
              {showHistory && (
                <div className="mt-2 max-h-60 overflow-y-auto space-y-2">
                  {history.map((item) => (
                    <div
                      key={item.id}
                      className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors cursor-pointer`}
                      onClick={() => {
                        setMessages([item]);
                        setShowHistory(false);
                      }}
                    >
                      <p className="text-xs font-medium truncate">
                        {item.role === "user" ? "用户: " : "助手: "} {item.content}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {new Date(item.timestamp).toLocaleString()}
                      </p>
                    </div>
                  ))}
                  {history.length === 0 && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 text-center py-4">
                      暂无历史记录
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h3 className="text-sm font-medium mb-2">管理</h3>
              <button
                onClick={clearHistory}
                className={`w-full flex items-center justify-center space-x-2 p-2 rounded-lg ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors`}
              >
                <Trash2 className="h-4 w-4" />
                <span className="text-sm">清除历史记录</span>
              </button>
            </div>
          </div>
        </aside>

        {/* 主内容区 */}
        <main className="flex-1 overflow-y-auto">
          <div className="container mx-auto px-4 py-6">
            <div className={`max-w-3xl mx-auto ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-sm border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                <h2 className="text-lg font-semibold">动漫角色识别</h2>
              </div>
              <div className="p-4 max-h-[calc(100vh-20rem)] overflow-y-auto space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] ${message.role === "user" ? "order-2" : "order-1"}`}
                    >
                      <div
                        className={`rounded-lg p-3 ${message.role === "user" ? 'bg-blue-500 text-white' : (darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900')}`}
                      >
                        {message.image && (
                          <div className="mb-3 rounded overflow-hidden">
                            <img
                              src={message.image}
                              alt="User uploaded image"
                              className="w-full h-auto object-cover"
                            />
                          </div>
                        )}
                        <p className="whitespace-pre-wrap break-words">{message.content}</p>

                        {message.classification && (
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500" />
                              <h4 className="font-semibold text-xs">识别结果</h4>
                            </div>
                            <div className={`grid grid-cols-2 gap-2 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                              <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                <p className="text-xs text-gray-500 dark:text-gray-400">角色</p>
                                <p className="text-sm font-medium">{message.classification.role}</p>
                              </div>
                              <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                <p className="text-xs text-gray-500 dark:text-gray-400">相似度</p>
                                <p className="text-sm font-medium">{(message.classification.similarity * 100).toFixed(1)}%</p>
                              </div>
                              <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded col-span-2`}>
                                <p className="text-xs text-gray-500 dark:text-gray-400">置信度</p>
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
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500" />
                              <h4 className="font-semibold text-xs">多角色识别结果</h4>
                            </div>
                            <div className="space-y-2">
                              {message.multi_roles.map((role, index) => (
                                <div key={index} className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                  <div className="flex justify-between items-center">
                                    <p className="text-sm font-medium">{role.role}</p>
                                    <div className="flex items-center space-x-2">
                                      <p className="text-xs">{(role.similarity * 100).toFixed(1)}%</p>
                                      <div
                                        className={`w-1.5 h-1.5 rounded-full ${role.similarity >= 0.8 ? "bg-green-500" : role.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                                      />
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {message.attributes && message.attributes.length > 0 && (
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500" />
                              <h4 className="font-semibold text-xs">角色属性</h4>
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {message.attributes.map((attr, index) => (
                                <span
                                  key={index}
                                  className={`px-2 py-1 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-xs font-medium`}
                                >
                                  {attr.tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {message.text_detections && message.text_detections.length > 0 && (
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500" />
                              <h4 className="font-semibold text-xs">文本检测</h4>
                            </div>
                            <div className="space-y-1">
                              {message.text_detections.map((text, index) => (
                                <div key={index} className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                                  <p className="text-sm font-medium">{text.text}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {message.ai_predicted_role && (
                          <div className="mt-3 space-y-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-green-500" />
                              <h4 className="font-semibold text-xs">AI预测角色</h4>
                            </div>
                            <div className={`p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded`}>
                              <p className="text-sm font-medium">{message.ai_predicted_role}</p>
                            </div>
                          </div>
                        )}

                        {message.thoughts && !message.isThinkingFinished && (
                          <div className="mt-3 space-y-1">
                            <div className="flex items-center space-x-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500" />
                              <h4 className="font-semibold text-xs">识别过程</h4>
                            </div>
                            <div className="space-y-1">
                              {message.thoughts.map((thought, index) => (
                                <div key={index} className="flex items-center space-x-2">
                                  <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                                  <p className="text-xs">{thought}</p>
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
                              className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                              title="复制内容"
                            >
                              <Copy className="h-3 w-3" />
                            </button>
                            <button
                              onClick={() => handleDownloadMessage(message.content, message.role)}
                              className={`p-1 rounded ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                              title="下载内容"
                            >
                              <Download className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div
                      className={`flex-shrink-0 mr-2 ml-2 ${message.role === "user" ? "order-1" : "order-2"}`}
                    >
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${message.role === "user" ? 'bg-blue-500 text-white' : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')}`}>
                        {message.role === "user" ? (
                          <User className="h-4 w-4" />
                        ) : (
                          <Bot className="h-4 w-4" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
              <div className="p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
                <div className="flex items-center space-x-3">
                  {/* 直接的文件输入元素 */}
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className={`px-3 py-2 rounded-lg ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-gray-50 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm`}
                  />
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="输入消息或上传图片..."
                      className={`w-full px-4 py-2 pr-12 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm`}
                      disabled={isProcessing}
                    />
                    <button
                      onClick={() => setInputText("")}
                      className={`absolute right-8 top-1/2 transform -translate-y-1/2 p-1 ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} rounded transition-colors`}
                      title="清空输入"
                      disabled={!inputText.trim() || isProcessing}
                    >
                      <X className={`h-4 w-4 ${inputText.trim() && !isProcessing ? '' : 'opacity-50 cursor-not-allowed'}`} />
                    </button>

                  </div>
                  <button
                    onClick={handleSend}
                    disabled={(!inputText.trim() && !selectedImage) || isProcessing}
                    className={`bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-1 ${(!inputText.trim() && !selectedImage) || isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isProcessing ? (
                      <>
                        <svg className="h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span className="text-sm">识别中</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        <span className="text-sm">发送</span>
                      </>
                    )}
                  </button>
                </div>
                {selectedImage && imagePreview && (
                  <div className={`mt-3 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'} border rounded-lg p-3 flex items-center space-x-3`}>
                    <div className="w-16 h-16 rounded overflow-hidden">
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
                      onClick={removeImage}
                      className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-red-900/20' : 'hover:bg-red-50'} text-red-500 transition-colors`}
                      title="移除图片"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* 页脚 */}
      <footer className={`py-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>动漫角色识别助手 © zhaoqi.cao arona 2026</p>
          <p className="mt-1">基于深度学习的动漫角色识别系统</p>
        </div>
      </footer>
    </div>
  );
}