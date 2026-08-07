"use client";

import { useState, useCallback, useRef } from "react";
import { Search, Image, X, Download, ExternalLink, UploadCloud, Sparkles, SlidersHorizontal } from "lucide-react";
import axios from "axios";
import EmptyState from "./EmptyState";
import { useAppStore } from "../store/useAppStore";

interface SearchResult {
  role: string;
  similarity: number;
  image?: string;
}

interface SearchPanelProps {
  darkMode: boolean;
  accessToken?: string;
}

export default function SearchPanel({ darkMode, accessToken }: SearchPanelProps) {
  const addToast = useAppStore((s) => s.addToast);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [topK, setTopK] = useState(10);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults([]);
  }, []);

  const handleSearch = useCallback(async () => {
    if (!selectedImage || isSearching) return;

    setIsSearching(true);
    setResults([]);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("top_k", topK.toString());

      const headers: any = {};
      if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
      }

      const response = await axios.post("/api/search/image", formData, { headers });

      if (response.data.results) {
        setResults(response.data.results || []);
        if (response.data.results.length > 0) {
          addToast(`找到 ${response.data.results.length} 个相似角色`, 'success');
        } else {
          addToast('未找到相似角色，换个图试试', 'info');
        }
      }
    } catch (error) {
      console.error("搜索失败:", error);
      addToast('搜索失败，请检查服务是否可用', 'error');
    } finally {
      setIsSearching(false);
    }
  }, [selectedImage, topK, isSearching, accessToken, addToast]);

  // 排名徽章颜色
  const rankColor = (index: number) =>
    index === 0 ? "bg-gradient-to-r from-amber-400 to-yellow-500" :
    index === 1 ? "bg-gradient-to-r from-slate-400 to-slate-500" :
    index === 2 ? "bg-gradient-to-r from-orange-400 to-amber-600" :
    (darkMode ? "bg-gray-600" : "bg-gray-300");

  const rankText = (index: number) =>
    index < 3 ? ["🥇", "🥈", "🥉"][index] : `#${index + 1}`;

  return (
    <div className={`${darkMode ? "bg-gray-800" : "bg-white"} rounded-xl shadow-lg border ${darkMode ? "border-gray-700" : "border-gray-200"} overflow-hidden animate-fade-in`}>
      {/* 标题栏 */}
      <div className={`p-4 border-b ${darkMode ? "border-gray-700" : "border-gray-200"}`}>
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center">
            <Search className="h-4.5 w-4.5 text-purple-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold leading-tight">以图搜图</h2>
            <p className="text-xs text-gray-500 mt-0.5">上传图片，检索相似动漫角色</p>
          </div>
        </div>
      </div>

      {/* 内容区 */}
      <div className="p-4 md:p-5">
        {/* 图片上传区（拖拽） */}
        {!imagePreview ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-8 md:p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
              isDragging
                ? "border-purple-500 bg-purple-50 dark:bg-purple-900/20 scale-[1.01]"
                : darkMode
                ? "border-gray-600 hover:border-purple-500 hover:bg-gray-700/40"
                : "border-gray-300 hover:border-purple-400 hover:bg-purple-50/40"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="hidden"
            />
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-3 transition-transform ${isDragging ? "scale-110" : ""} ${darkMode ? "bg-gray-700 text-purple-400" : "bg-purple-50 text-purple-500"}`}>
              <UploadCloud className="h-8 w-8" />
            </div>
            <p className="font-medium text-sm md:text-base">点击选择或拖拽图片到此处</p>
            <p className={`text-xs mt-1.5 ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
              支持 JPG / PNG / WebP，≤ 20MB
            </p>
          </div>
        ) : (
          <div className={`relative rounded-xl overflow-hidden border ${darkMode ? "bg-gray-900 border-gray-700" : "bg-gray-50 border-gray-200"}`}>
            <img src={imagePreview} alt="预览" className="w-full h-56 object-contain" />
            <button
              onClick={removeImage}
              className="absolute top-2 right-2 p-1.5 bg-black/50 rounded-full text-white hover:bg-black/70 hover:scale-110 transition-all"
              title="移除图片"
            >
              <X className="h-4 w-4" />
            </button>
            <div className={`absolute bottom-0 left-0 right-0 px-3 py-1.5 bg-black/60 text-white text-xs flex items-center justify-between`}>
              <span className="truncate">{selectedImage?.name}</span>
              <span className="ml-2 shrink-0">{(selectedImage && Math.round(selectedImage.size / 1024))} KB</span>
            </div>
          </div>
        )}

        {/* 搜索数量设置 */}
        <div className={`mt-4 p-3 rounded-lg border ${darkMode ? "bg-gray-700/50 border-gray-600" : "bg-gray-50 border-gray-200"}`}>
          <div className="flex items-center justify-between mb-2">
            <label className="flex items-center space-x-2 text-sm font-medium">
              <SlidersHorizontal className={`h-4 w-4 ${darkMode ? "text-purple-400" : "text-purple-500"}`} />
              <span>返回数量</span>
            </label>
            <span className={`text-sm font-semibold ${darkMode ? "text-purple-400" : "text-purple-600"}`}>{topK}</span>
          </div>
          <input
            type="range"
            min="1"
            max="50"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>

        {/* 搜索按钮 */}
        <button
          onClick={handleSearch}
          disabled={!selectedImage || isSearching}
          className={`w-full mt-4 bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${!selectedImage || isSearching ? "opacity-50 cursor-not-allowed" : "transform hover:scale-[1.01] hover:shadow-lg"}`}
        >
          {isSearching ? (
            <>
              <svg className="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>搜索中...</span>
            </>
          ) : (
            <>
              <Search className="h-5 w-5" />
              <span>搜索相似角色</span>
            </>
          )}
        </button>

        {/* 搜索结果 */}
        {results.length > 0 && (
          <div className="mt-6 animate-fade-in">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium flex items-center space-x-2">
                <Sparkles className={`h-4 w-4 ${darkMode ? "text-purple-400" : "text-purple-500"}`} />
                <span>搜索结果 ({results.length})</span>
              </h3>
              <button
                onClick={() => { setResults([]); removeImage(); }}
                className={`text-xs px-2 py-1 rounded transition-colors ${darkMode ? "bg-gray-700 text-gray-300 hover:bg-gray-600" : "bg-gray-100 text-gray-600 hover:bg-gray-200"}`}
              >
                清空
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {results.map((result, index) => (
                <div
                  key={index}
                  className={`relative rounded-xl overflow-hidden cursor-pointer group ${darkMode ? "bg-gray-700" : "bg-gray-100"} border ${darkMode ? "border-gray-600" : "border-transparent"} transition-all hover:shadow-xl hover:-translate-y-0.5`}
                >
                  {/* 排名徽章 */}
                  <span className={`absolute top-1.5 left-1.5 z-10 text-[10px] font-bold text-white px-1.5 py-0.5 rounded-full shadow ${rankColor(index)}`}>
                    {rankText(index)}
                  </span>
                  <div className="aspect-square bg-gray-200 dark:bg-gray-600 flex items-center justify-center">
                    {result.image ? (
                      <img
                        src={result.image}
                        alt={result.role}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    ) : (
                      <Image className="h-8 w-8 text-gray-400" />
                    )}
                  </div>
                  <div className={`p-2 ${darkMode ? "bg-gray-800" : "bg-white"}`}>
                    <p className="text-xs font-medium truncate">{result.role}</p>
                    <div className="flex items-center space-x-1.5 mt-1">
                      <div className={`flex-1 h-1.5 rounded-full overflow-hidden ${darkMode ? "bg-gray-600" : "bg-gray-200"}`}>
                        <div
                          className={`h-full rounded-full ${result.similarity >= 0.8 ? "bg-green-500" : result.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                          style={{ width: `${Math.min(100, result.similarity * 100)}%` }}
                        />
                      </div>
                      <span className={`text-[10px] font-semibold ${result.similarity >= 0.8 ? "text-green-500" : result.similarity >= 0.5 ? "text-yellow-500" : "text-red-500"}`}>
                        {(result.similarity * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  {/* 悬停效果 */}
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <div className="flex space-x-2">
                      <button className="p-2 bg-white rounded-full text-gray-800 hover:bg-purple-500 hover:text-white transition-colors" title="查看详情">
                        <ExternalLink className="h-4 w-4" />
                      </button>
                      <button className="p-2 bg-white rounded-full text-gray-800 hover:bg-purple-500 hover:text-white transition-colors" title="下载图片">
                        <Download className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 空状态：已上传但未搜索 */}
        {imagePreview && !isSearching && results.length === 0 && (
          <div className="mt-6">
            <EmptyState
              compact
              darkMode={darkMode}
              icon={<Search className="h-6 w-6" />}
              title="图片已就绪"
              description="点击上方「搜索相似角色」按钮开始检索"
            />
          </div>
        )}
      </div>
    </div>
  );
}
