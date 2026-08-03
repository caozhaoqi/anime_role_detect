import React, { useState, useEffect } from 'react';
import { Clock, Trash2, Image, Brain, Tag, AlertTriangle, FileText, Check, X, LogIn, RefreshCw } from 'lucide-react';
import { Message } from '../types';
import EmptyState from './EmptyState';

interface HistoryPanelProps {
  darkMode: boolean;
  onViewRecord: (record: any) => void;
  onDeleteRecord: (recordId: string) => void;
  onClose: () => void;
  onAuthError?: () => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ darkMode, onViewRecord, onDeleteRecord, onClose, onAuthError }) => {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAuthError, setIsAuthError] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    setIsAuthError(false);
    try {
      const token = localStorage.getItem('accessToken');
      if (!token) {
        setIsAuthError(true);
        setError('请先登录后再查看历史记录');
        setHistory([]);
        setLoading(false);
        if (onAuthError) onAuthError();
        return;
      }
      const response = await fetch('/api/history', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.status === 401) {
        setIsAuthError(true);
        setError('登录已过期，请重新登录');
        setHistory([]);
        if (onAuthError) onAuthError();
        return;
      }
      const data = await response.json();
      if (data.success) {
        setHistory(data.data);
      } else {
        setError(data.message || '获取历史记录失败');
      }
    } catch (err) {
      setError('网络错误，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (recordId: string) => {
    if (confirm('确定要删除这条记录吗？')) {
      try {
        const response = await fetch(`/api/history/${recordId}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
          }
        });
        const data = await response.json();
        if (data.success) {
          setHistory(history.filter(record => record.id !== recordId));
        } else {
          alert(data.message || '删除失败');
        }
      } catch (err) {
        alert('删除失败，请稍后重试');
      }
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'} shadow-lg h-full`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className={`w-8 h-8 rounded-lg flex items-center justify-center ${darkMode ? 'bg-gray-700 text-blue-400' : 'bg-blue-50 text-blue-500'}`}>
            <Clock className="h-4 w-4" />
          </span>
          识别历史
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchHistory}
            className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-sm transition-colors ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>刷新</span>
          </button>
          <button
            onClick={onClose}
            className={`p-1.5 rounded-lg transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'}`}
            title="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : error ? (
        <div className={`p-4 rounded-md ${darkMode ? 'bg-red-900/30' : 'bg-red-100'}`}>
          <div className={`flex items-center gap-2 ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
            {isAuthError ? <LogIn className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
            <span className="text-sm font-medium">{error}</span>
          </div>
          {isAuthError && (
            <button
              onClick={() => {
                if (onAuthError) onAuthError();
                window.dispatchEvent(new CustomEvent('open-login'));
              }}
              className={`mt-3 w-full px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                darkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              登录
            </button>
          )}
        </div>
      ) : history.length === 0 ? (
        <EmptyState
          darkMode={darkMode}
          icon={<Image className="h-7 w-7" />}
          title="暂无识别历史"
          description="完成一次角色识别后，记录会自动保存在这里"
        />
      ) : (
        <div className="space-y-3 max-h-[calc(100vh-280px)] overflow-y-auto pr-2">
          {history.map((record) => (
            <div
              key={record.id}
              className={`p-3 rounded-xl border cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5 ${darkMode ? 'bg-gray-700 border-gray-600 hover:border-blue-500' : 'bg-gray-50 border-gray-200 hover:border-blue-300'}`}
              onClick={() => onViewRecord(record)}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Image className="h-4 w-4" />
                    <span className="font-medium truncate">{record.image_filename}</span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                    <span className="flex items-center gap-1">
                      <Brain className="h-3 w-3" />
                      {record.model_used}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatTime(record.timestamp)}
                    </span>
                    <span className="flex items-center gap-1">
                      <FileText className="h-3 w-3" />
                      {record.detected_text ? '有文字' : '无文字'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(record.id);
                  }}
                  className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors`}
                  title="删除记录"
                >
                  <Trash2 className="h-4 w-4 text-red-500" />
                </button>
              </div>
              
              <div className="mt-2 flex flex-wrap gap-1">
                {record.nsfw_status && (
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    NSFW
                  </span>
                )}
                {record.is_multi_role && (
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400 flex items-center gap-1">
                    <Brain className="h-3 w-3" />
                    多角色
                  </span>
                )}
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400 flex items-center gap-1">
                  <Check className="h-3 w-3" />
                  成功
                </span>
              </div>
              
              {record.recognition_result.role && (
                <div className="mt-2">
                  <div className="flex items-center gap-2 mb-1">
                    <Tag className={`h-4 w-4 ${darkMode ? 'text-purple-400' : 'text-purple-500'}`} />
                    <span className="text-sm font-medium">{record.recognition_result.role}</span>
                    {record.recognition_result.similarity && (
                      <span className={`text-xs font-semibold ${record.recognition_result.similarity >= 0.8 ? 'text-green-500' : record.recognition_result.similarity >= 0.5 ? 'text-yellow-500' : 'text-red-500'}`}>
                        {(record.recognition_result.similarity * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {record.recognition_result.similarity && (
                    <div className={`h-1.5 rounded-full overflow-hidden ${darkMode ? 'bg-gray-600' : 'bg-gray-200'}`}>
                      <div
                        className={`h-full rounded-full ${record.recognition_result.similarity >= 0.8 ? 'bg-green-500' : record.recognition_result.similarity >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.min(100, record.recognition_result.similarity * 100)}%` }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default HistoryPanel;