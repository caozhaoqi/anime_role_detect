"use client";

import { useState, useCallback } from "react";
import { Search, Image, X, Download, ExternalLink, ArrowRight } from "lucide-react";
import axios from "axios";

interface SearchResult {
  role: string;
  similarity: number;
  image_path?: string;
}

interface SearchPanelProps {
  darkMode: boolean;
  accessToken?: string;
}

export default function SearchPanel({ darkMode, accessToken }: SearchPanelProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [topK, setTopK] = useState(10);

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
      }
    } catch (error) {
      console.error("搜索失败:", error);
    } finally {
      setIsSearching(false);
    }
  }, [selectedImage, topK, isSearching, accessToken]);

  return (
    <div className={`${darkMode ? "bg-gray-800" : "bg-white"} rounded-xl shadow-lg border ${darkMode ? "border-gray-700" : "border-gray-200"} overflow-hidden`}>
      {/* 标题栏 */}
      <div className={`p-4 border-b ${darkMode ? "border-gray-700" : "border-gray-200"}`}>
        <div className="flex items-center space-x-2">
          <Search className="h-5 w-5 text-blue-500" />
          <h2 className="text-lg font-semibold">以图搜图</h2>
        </div>
        <p className="text-sm text-gray-500 mt-1">上传图片搜索相似的动漫角色</p>
      </div>

      {/* 内容区 */}
      <div className="p-4">
        {/* 图片上传区 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            上传图片
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className={`w-full px-3 py-2 rounded-lg ${darkMode ? "bg-gray-700 border-gray-600 text-white" : "bg-gray-50 border-gray-200"} border focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
        </div>

        {/* 搜索数量设置 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            返回数量: {topK}
          </label>
          <input
            type="range"
            min="1"
            max="50"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        {/* 图片预览 */}
        {imagePreview && (
          <div className={`mb-4 relative ${darkMode ? "bg-gray-700" : "bg-gray-50"} rounded-lg overflow-hidden`}>
            <img
              src={imagePreview}
              alt="预览"
              className="w-full h-48 object-contain"
            />
            <button
              onClick={removeImage}
              className="absolute top-2 right-2 p-1 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* 搜索按钮 */}
        <button
          onClick={handleSearch}
          disabled={!selectedImage || isSearching}
          className={`w-full bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${!selectedImage || isSearching ? "opacity-50 cursor-not-allowed" : "transform hover:scale-[1.02]"}`}
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
          <div className="mt-6">
            <h3 className="text-sm font-medium mb-3">搜索结果 ({results.length})</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {results.map((result, index) => (
                <div
                  key={index}
                  className={`relative rounded-lg overflow-hidden cursor-pointer group ${darkMode ? "bg-gray-700" : "bg-gray-100"} transition-all hover:shadow-lg`}
                >
                  <div className="aspect-square bg-gray-200 dark:bg-gray-600 flex items-center justify-center">
                    {result.image ? (
                      <img
                        src={result.image}
                        alt={result.role}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <Image className="h-8 w-8 text-gray-400" />
                    )}
                  </div>
                  <div className={`p-2 ${darkMode ? "bg-gray-800" : "bg-white"}`}>
                    <p className="text-xs font-medium truncate">{result.role}</p>
                    <p className="text-xs text-gray-500">相似度: {(result.similarity * 100).toFixed(1)}%</p>
                  </div>
                  {/* 悬停效果 */}
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button className="p-2 bg-white rounded-full text-gray-800 hover:bg-blue-500 hover:text-white transition-colors">
                      <ExternalLink className="h-4 w-4" />
                    </button>
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