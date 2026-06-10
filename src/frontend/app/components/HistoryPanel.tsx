import React, { useState, useEffect } from 'react';
import { Clock, Trash2, Image, Brain, Tag, AlertTriangle, FileText, Check, X } from 'lucide-react';
import { Message } from '../types';

interface HistoryPanelProps {
  darkMode: boolean;
  onViewRecord: (record: any) => void;
  onDeleteRecord: (recordId: string) => void;
  onClose: () => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ darkMode, onViewRecord, onDeleteRecord, onClose }) => {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/history', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
        }
      });
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
    <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Clock className="h-5 w-5" />
          识别历史
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchHistory}
            className={`px-3 py-1 rounded-md text-sm ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'} transition-colors`}
          >
            刷新
          </button>
          <button
            onClick={onClose}
            className={`p-1 rounded-md transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'}`}
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
        <div className={`p-4 rounded-md ${darkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-600'}`}>
          {error}
        </div>
      ) : history.length === 0 ? (
        <div className={`p-8 text-center ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <Image className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>暂无识别历史</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {history.map((record) => (
            <div
              key={record.id}
              className={`p-3 rounded-lg cursor-pointer transition-all hover:shadow-md ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'}`}
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
                <div className="mt-2 flex items-center gap-2">
                  <Tag className="h-4 w-4 text-purple-500" />
                  <span className="text-sm font-medium">{record.recognition_result.role}</span>
                  {record.recognition_result.similarity && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      ({(record.recognition_result.similarity * 100).toFixed(1)}%)
                    </span>
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