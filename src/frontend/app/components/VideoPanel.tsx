"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Video, Play, Pause, X, Clock, AlertTriangle, CheckCircle } from "lucide-react";
import axios from "axios";

interface VideoResult {
  timestamp: number;
  frame_number: number;  // 后端返回的是 frame_number
  roles: {
    role: string;
    similarity: number;
    box?: { x: number; y: number; w: number; h: number };
  }[];
}

interface VideoPanelProps {
  darkMode: boolean;
  accessToken?: string;
}

export default function VideoPanel({ darkMode, accessToken }: VideoPanelProps) {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<VideoResult[]>([]);
  const [frameInterval, setFrameInterval] = useState(1.0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [recognitionMode, setRecognitionMode] = useState("search"); // 'search' or 'inference'
  const [modelName, setModelName] = useState("efficientnet_b3_loli_optimized_v2_20260529_133654");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (selectedVideo && videoPreview) {
      const video = document.createElement("video");
      video.src = videoPreview;
      video.onloadedmetadata = () => {
        setDuration(video.duration);
      };
    }
  }, [selectedVideo, videoPreview]);

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedVideo(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setVideoPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setResults([]);
    }
  };

  const removeVideo = useCallback(() => {
    setSelectedVideo(null);
    setVideoPreview(null);
    setResults([]);
    setCurrentTime(0);
    setDuration(0);
  }, []);

  const handleRecognize = useCallback(async () => {
    if (!selectedVideo || isProcessing) return;

    setIsProcessing(true);
    setResults([]);

    try {
      const formData = new FormData();
      formData.append("file", selectedVideo);
      
      // 添加识别模式参数
      const params = new URLSearchParams({
        frame_interval: frameInterval.toString(),
        confidence_threshold: confidenceThreshold.toString(),
        recognition_mode: recognitionMode,
        model_name: modelName,
      });

      const headers: any = {};
      if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
      }

      const response = await axios.post(`/api/video/recognize?${params}`, formData, { headers });
      
      if (response.data.success) {
        // 兼容两种返回格式
        const results = response.data.data?.results || response.data.results || [];
        setResults(results);
      }
    } catch (error) {
      console.error("视频识别失败:", error);
    } finally {
      setIsProcessing(false);
    }
  }, [selectedVideo, frameInterval, confidenceThreshold, isProcessing, accessToken]);

  const handleTimeUpdate = (e: React.ChangeEvent<HTMLVideoElement>) => {
    setCurrentTime(e.target.currentTime);
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className={`${darkMode ? "bg-gray-800" : "bg-white"} rounded-xl shadow-lg border ${darkMode ? "border-gray-700" : "border-gray-200"} overflow-hidden`}>
      {/* 标题栏 */}
      <div className={`p-4 border-b ${darkMode ? "border-gray-700" : "border-gray-200"}`}>
        <div className="flex items-center space-x-2">
          <Video className="h-5 w-5 text-green-500" />
          <h2 className="text-lg font-semibold">视频实时识别</h2>
        </div>
        <p className="text-sm text-gray-500 mt-1">上传视频进行实时抽帧识别角色</p>
      </div>

      {/* 内容区 */}
      <div className="p-4">
        {/* 视频上传区 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            上传视频
          </label>
          <input
            type="file"
            accept="video/*"
            onChange={handleVideoSelect}
            className={`w-full px-3 py-2 rounded-lg ${darkMode ? "bg-gray-700 border-gray-600 text-white" : "bg-gray-50 border-gray-200"} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
        </div>

        {/* 参数设置 */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              抽帧间隔: {frameInterval}s
            </label>
            <input
              type="range"
              min="0.1"
              max="10"
              step="0.1"
              value={frameInterval}
              onChange={(e) => setFrameInterval(Number(e.target.value))}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              置信度阈值: {(confidenceThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
        </div>

        {/* 识别模式选择 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">识别模式</label>
          <div className="flex space-x-3">
            <button
              onClick={() => setRecognitionMode("search")}
              className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                recognitionMode === "search"
                  ? "bg-green-500 text-white border-green-500"
                  : darkMode
                  ? "bg-gray-700 border-gray-600 text-gray-300 hover:border-green-500"
                  : "bg-white border-gray-300 text-gray-700 hover:border-green-500"
              }`}
            >
              <div className="font-medium">搜图模式</div>
              <div className="text-xs mt-1 opacity-80">快速，适合实时处理</div>
            </button>
            <button
              onClick={() => setRecognitionMode("inference")}
              className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                recognitionMode === "inference"
                  ? "bg-blue-500 text-white border-blue-500"
                  : darkMode
                  ? "bg-gray-700 border-gray-600 text-gray-300 hover:border-blue-500"
                  : "bg-white border-gray-300 text-gray-700 hover:border-blue-500"
              }`}
            >
              <div className="font-medium">模型推理</div>
              <div className="text-xs mt-1 opacity-80">更准确，速度较慢</div>
            </button>
          </div>
        </div>

        {/* 视频预览 */}
        {videoPreview && (
          <div className={`mb-4 relative ${darkMode ? "bg-gray-900" : "bg-gray-900"} rounded-lg overflow-hidden aspect-video`}>
            <video
              ref={videoRef}
              src={videoPreview}
              className="w-full h-full"
              controls={false}
              onTimeUpdate={handleTimeUpdate}
              onClick={togglePlay}
            />
            
            {/* 播放控制覆盖层 */}
            <div className="absolute inset-0 flex items-center justify-center bg-black/30">
              <button
                onClick={togglePlay}
                className="p-4 bg-white/20 backdrop-blur-sm rounded-full text-white hover:bg-white/30 transition-colors"
              >
                {isPlaying ? <Pause className="h-8 w-8" /> : <Play className="h-8 w-8" />}
              </button>
            </div>

            {/* 时间显示 */}
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
              <div className="flex items-center justify-between text-white text-sm">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
              <input
                type="range"
                min="0"
                max={duration || 1}
                value={currentTime}
                onChange={(e) => {
                  const time = Number(e.target.value);
                  setCurrentTime(time);
                  if (videoRef.current) {
                    videoRef.current.currentTime = time;
                  }
                }}
                className="w-full h-1 bg-white/30 rounded-full appearance-none cursor-pointer mt-1"
              />
            </div>

            {/* 关闭按钮 */}
            <button
              onClick={removeVideo}
              className="absolute top-2 right-2 p-1 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* 识别按钮 */}
        <button
          onClick={handleRecognize}
          disabled={!selectedVideo || isProcessing}
          className={`w-full bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${!selectedVideo || isProcessing ? "opacity-50 cursor-not-allowed" : "transform hover:scale-[1.02]"}`}
        >
          {isProcessing ? (
            <>
              <svg className="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>识别中...</span>
            </>
          ) : (
            <>
              <Video className="h-5 w-5" />
              <span>开始识别</span>
            </>
          )}
        </button>

        {/* 识别结果 */}
        {results.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium mb-3">识别结果 ({results.length} 帧)</h3>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {results.map((result, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg ${darkMode ? "bg-gray-700" : "bg-gray-50"}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium">第 {result.frame_number} 帧</span>
                      <span className="text-xs text-gray-500">({result.timestamp.toFixed(2)}s)</span>
                    </div>
                    {result.roles.length > 0 ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {result.roles.length > 0 ? (
                      result.roles.map((role, rIndex) => (
                        <span
                          key={rIndex}
                          className={`px-2 py-1 rounded-full text-xs font-medium ${
                            role.similarity > 0.8
                              ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                              : role.similarity > 0.5
                              ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                              : "bg-gray-100 text-gray-800 dark:bg-gray-600 dark:text-gray-200"
                          }`}
                        >
                          {role.role} ({(role.similarity * 100).toFixed(0)}%)
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-gray-500">未检测到角色</span>
                    )}
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