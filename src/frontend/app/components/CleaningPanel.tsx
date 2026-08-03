"use client";

import { useState, useEffect, useCallback } from 'react';
import { Play, Settings, FolderOpen, Clock, CheckCircle, XCircle, Loader2, RefreshCw, Trash2, Download, ChevronDown, ChevronUp, AlertCircle, X, ChevronLeft, Folder, Sparkles } from 'lucide-react';
import axios from 'axios';
import { CleaningConfig, CleaningResponse, CleaningTask, CleaningProgress } from '../types';
import EmptyState from './EmptyState';

interface CleaningPanelProps {
  darkMode: boolean;
  accessToken?: string;
}

export default function CleaningPanel({ darkMode, accessToken }: CleaningPanelProps) {
  const [inputDir, setInputDir] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [config, setConfig] = useState<CleaningConfig>({
    enable_deduplication: true,
    enable_consistency_filter: true,
    enable_cluster_filter: true,
    enable_mislabeled_detector: true,
    enable_danbooru_enrichment: false,
    similarity_threshold: 0.95,
    consistency_threshold: 0.25,
    outlier_threshold: 0.7,
    text_threshold: 0.2,
    confusion_gap: 0.08,
    dry_run: false,
    min_images_per_character: 5,
    max_workers: 4,
  });
  
  const [isRunning, setIsRunning] = useState(false);
  const [isAsync, setIsAsync] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [currentTaskStatus, setCurrentTaskStatus] = useState<CleaningTask | null>(null);
  const [lastResult, setLastResult] = useState<CleaningResponse | null>(null);
  const [tasks, setTasks] = useState<CleaningTask[]>([]);
  const [showConfig, setShowConfig] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [pollInterval, setPollInterval] = useState<number | null>(null);
  
  // 数据清理进度状态
  const [cleaningProgress, setCleaningProgress] = useState<CleaningProgress | null>(null);
  const [progressPollInterval, setProgressPollInterval] = useState<number | null>(null);
  const [showProgressPanel, setShowProgressPanel] = useState(true);
  
  // 目录浏览器状态
  const [showBrowser, setShowBrowser] = useState(false);
  const [browseTarget, setBrowseTarget] = useState<'input' | 'output'>('input');
  const [browsePath, setBrowsePath] = useState('/');
  const [browseEntries, setBrowseEntries] = useState<{name: string, path: string}[]>([]);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseParent, setBrowseParent] = useState('/');
  const [browseHistory, setBrowseHistory] = useState<string[]>([]);

  useEffect(() => {
    loadDefaultConfig();
    loadTasks();
    loadCleaningProgress();
  }, []);

  useEffect(() => {
    if (pollInterval) {
      const interval = setInterval(() => {
        if (currentTaskId) {
          checkTaskStatus(currentTaskId);
        }
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [pollInterval, currentTaskId]);

  // 进度轮询
  useEffect(() => {
    if (progressPollInterval) {
      const interval = setInterval(() => {
        loadCleaningProgress();
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [progressPollInterval]);

  const loadDefaultConfig = async () => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.get('/api/cleaning/config/default', { headers });
      if (response.data.success) {
        setConfig(prev => ({ ...prev, ...response.data.data }));
      }
    } catch (err) {
      console.error('加载默认配置失败:', err);
    }
  };

  const loadTasks = async () => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.get('/api/cleaning/tasks', { headers });
      if (response.data.success) {
        setTasks(response.data.data.tasks || []);
      }
    } catch (err) {
      console.error('加载任务列表失败:', err);
    }
  };

  // 加载清理进度
  const loadCleaningProgress = async () => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.get('/api/cleaning/progress', { headers });
      if (response.data.success) {
        setCleaningProgress(response.data.data);
      }
    } catch (err) {
      console.error('加载清理进度失败:', err);
    }
  };

  // 重置清理进度
  const resetCleaningProgress = async () => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.post('/api/cleaning/progress/reset', {}, { headers });
      if (response.data.success) {
        loadCleaningProgress();
      }
    } catch (err) {
      console.error('重置清理进度失败:', err);
    }
  };

  // 目录浏览函数
  const openBrowser = async (target: 'input' | 'output') => {
    setBrowseTarget(target);
    setBrowsePath('/');
    setBrowseHistory([]);
    setShowBrowser(true);
    await loadDirectory('/');
  };

  const loadDirectory = async (path: string) => {
    setBrowseLoading(true);
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.get('/api/cleaning/browse', { params: { path }, headers });
      if (response.data.success) {
        setBrowseEntries(response.data.data.entries);
        setBrowsePath(response.data.data.current_path);
        setBrowseParent(response.data.data.parent_path);
      }
    } catch (err) {
      console.error('加载目录失败:', err);
    } finally {
      setBrowseLoading(false);
    }
  };

  const navigateToDir = async (path: string) => {
    setBrowseHistory(prev => [...prev, browsePath]);
    await loadDirectory(path);
  };

  const navigateUp = async () => {
    if (browseHistory.length > 0) {
      const prev = browseHistory[browseHistory.length - 1];
      setBrowseHistory(prev => prev.slice(0, -1));
      await loadDirectory(prev);
    } else {
      await loadDirectory(browseParent);
    }
  };

  const selectDirectory = (path: string) => {
    if (browseTarget === 'input') {
      setInputDir(path);
    } else {
      setOutputDir(path);
    }
    setShowBrowser(false);
  };

  const checkTaskStatus = async (taskId: string) => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.get(`/api/cleaning/task/${taskId}`, { headers });
      if (response.data.success) {
        const data = response.data.data;
        const task: CleaningTask = {
          task_id: taskId,
          status: data.status as CleaningTask['status'],
          input_dir: data.input_dir,
          output_dir: data.output_dir,
          start_time: data.start_time,
          end_time: data.end_time,
          duration_seconds: data.duration_seconds,
          result: data.result,
          error: data.error_message || data.error,
          // 新增进度字段
          user_id: data.user_id,
          username: data.username,
          total_files: data.total_files,
          processed_files: data.processed_files,
          valid_files: data.valid_files,
          rejected_files: data.rejected_files,
          duplicate_files: data.duplicate_files,
          progress_percent: data.progress_percent,
          report_path: data.report_path,
          error_message: data.error_message,
        };
        setCurrentTaskStatus(task);
        
        if (task.status === 'completed' || task.status === 'failed') {
          setPollInterval(null);
          setLastResult({
            success: task.status === 'completed',
            message: task.status === 'completed' ? '清洗完成' : task.error_message || task.error || '清洗失败',
            data: task.result,
            task_id: taskId,
          });
          loadTasks();
        }
      }
    } catch (err) {
      console.error('查询任务状态失败:', err);
    }
  };

  const runCleaning = useCallback(async () => {
    if (!inputDir || !outputDir) {
      setError('请输入输入目录和输出目录');
      return;
    }

    setError(null);
    setIsRunning(true);

    try {
      const headers: any = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const formData = new FormData();
      formData.append('input_dir', inputDir);
      formData.append('output_dir', outputDir);
      formData.append('enable_deduplication', config.enable_deduplication.toString());
      formData.append('enable_consistency_filter', config.enable_consistency_filter.toString());
      formData.append('enable_cluster_filter', config.enable_cluster_filter.toString());
      formData.append('enable_mislabeled_detector', config.enable_mislabeled_detector.toString());
      formData.append('enable_danbooru_enrichment', config.enable_danbooru_enrichment.toString());
      formData.append('similarity_threshold', config.similarity_threshold.toString());
      formData.append('consistency_threshold', config.consistency_threshold.toString());
      formData.append('outlier_threshold', config.outlier_threshold.toString());
      formData.append('text_threshold', config.text_threshold.toString());
      formData.append('confusion_gap', config.confusion_gap.toString());
      formData.append('dry_run', config.dry_run.toString());
      formData.append('min_images_per_character', config.min_images_per_character.toString());
      formData.append('max_workers', config.max_workers.toString());

      const endpoint = isAsync ? '/api/cleaning/run/async' : '/api/cleaning/run';
      const response = await axios.post(endpoint, formData, { headers });

      if (response.data.success) {
        if (isAsync) {
          const taskId = response.data.task_id;
          setCurrentTaskId(taskId);
          setPollInterval(Date.now());
          setCurrentTaskStatus({
            task_id: taskId,
            status: 'pending',
            input_dir: inputDir,
            output_dir: outputDir,
          });
        } else {
          setLastResult(response.data);
        }
      } else {
        setError(response.data.message || '清洗失败');
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.response?.data?.message || err.message || '清洗失败';
      setError(errorMsg);
      console.error('清洗失败:', err);
    } finally {
      setIsRunning(false);
      if (!isAsync) {
        loadTasks();
      }
    }
  }, [inputDir, outputDir, config, isAsync, accessToken]);

  const deleteTask = async (taskId: string) => {
    try {
      const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
      const response = await axios.delete(`/api/cleaning/task/${taskId}`, { headers });
      if (response.data.success) {
        loadTasks();
      }
    } catch (err) {
      console.error('删除任务失败:', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return '等待中';
      case 'running':
        return '运行中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      default:
        return status;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-500';
      case 'running':
        return 'text-blue-500';
      case 'completed':
        return 'text-green-500';
      case 'failed':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}分${secs}秒`;
  };

  const formatTime = (timestamp?: number) => {
    if (!timestamp) return '-';
    return new Date(timestamp * 1000).toLocaleString('zh-CN');
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* 标题区域 */}
      <div className={`p-4 md:p-5 rounded-xl border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-lg`}>
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg md:text-xl font-semibold leading-tight">数据清洗流水线</h2>
            <p className={`text-xs mt-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              去重 · 一致性过滤 · 聚类 · 错误标签检测
            </p>
          </div>
        </div>
      </div>

      {/* 输入输出目录 */}
      <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              输入目录
            </label>
            <div className="relative flex gap-2">
              <input
                type="text"
                value={inputDir}
                onChange={(e) => setInputDir(e.target.value)}
                placeholder="输入包含角色子目录的路径"
                className={`flex-1 px-4 py-2 pr-10 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-gray-100' : 'bg-white border-gray-300 text-gray-900'} focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none`}
              />
              <button
                onClick={() => openBrowser('input')}
                className={`px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                  darkMode ? 'bg-gray-700 border-gray-600 text-gray-200 hover:bg-gray-600' : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
                title="浏览目录"
              >
                <FolderOpen className="h-4 w-4 inline mr-1" />
                浏览
              </button>
            </div>
          </div>
          <div>
            <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              输出目录
            </label>
            <div className="relative flex gap-2">
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="清洗后数据输出路径"
                className={`flex-1 px-4 py-2 pr-10 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600 text-gray-100' : 'bg-white border-gray-300 text-gray-900'} focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none`}
              />
              <button
                onClick={() => openBrowser('output')}
                className={`px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                  darkMode ? 'bg-gray-700 border-gray-600 text-gray-200 hover:bg-gray-600' : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
                title="浏览目录"
              >
                <FolderOpen className="h-4 w-4 inline mr-1" />
                浏览
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 异步模式切换 */}
      <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="flex items-center justify-between">
          <div>
            <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              执行模式
            </label>
            <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              同步模式：等待清洗完成后返回结果；异步模式：后台运行，可查询进度
            </p>
          </div>
          <button
            onClick={() => setIsAsync(!isAsync)}
            className={`relative w-14 h-7 rounded-full transition-colors ${isAsync ? 'bg-blue-600' : 'bg-gray-600'}`}
          >
            <span className={`absolute top-1 w-5 h-5 rounded-full bg-white transition-transform ${isAsync ? 'translate-x-8' : 'translate-x-1'}`} />
          </button>
        </div>
        <p className={`text-sm mt-2 ${isAsync ? 'text-blue-400' : 'text-gray-500'}`}>
          {isAsync ? '当前模式：异步执行' : '当前模式：同步执行'}
        </p>
      </div>

      {/* 配置面板 */}
      <div className={`rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="w-full p-4 flex items-center justify-between"
        >
          <div className="flex items-center space-x-2">
            <Settings className={`h-5 w-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
            <span className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>清洗配置</span>
          </div>
          {showConfig ? (
            <ChevronUp className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
          ) : (
            <ChevronDown className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
          )}
        </button>

        {showConfig && (
          <div className="p-4 pt-0 space-y-4">
            {/* 阶段开关 */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              <label className={`flex items-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <input
                  type="checkbox"
                  checked={config.enable_deduplication}
                  onChange={(e) => setConfig(prev => ({ ...prev, enable_deduplication: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>CLIP去重</span>
              </label>
              <label className={`flex items-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <input
                  type="checkbox"
                  checked={config.enable_consistency_filter}
                  onChange={(e) => setConfig(prev => ({ ...prev, enable_consistency_filter: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>一致性过滤</span>
              </label>
              <label className={`flex items-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <input
                  type="checkbox"
                  checked={config.enable_cluster_filter}
                  onChange={(e) => setConfig(prev => ({ ...prev, enable_cluster_filter: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>HDBSCAN过滤</span>
              </label>
              <label className={`flex items-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <input
                  type="checkbox"
                  checked={config.enable_mislabeled_detector}
                  onChange={(e) => setConfig(prev => ({ ...prev, enable_mislabeled_detector: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>错误标签检测</span>
              </label>
              <label className={`flex items-center space-x-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <input
                  type="checkbox"
                  checked={config.enable_danbooru_enrichment}
                  onChange={(e) => setConfig(prev => ({ ...prev, enable_danbooru_enrichment: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Danbooru增强</span>
              </label>
            </div>

            {/* 参数设置 */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  去重相似度阈值 ({config.similarity_threshold})
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="0.99"
                  step="0.01"
                  value={config.similarity_threshold}
                  onChange={(e) => setConfig(prev => ({ ...prev, similarity_threshold: parseFloat(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  一致性阈值 ({config.consistency_threshold})
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.01"
                  value={config.consistency_threshold}
                  onChange={(e) => setConfig(prev => ({ ...prev, consistency_threshold: parseFloat(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  异常检测阈值 ({config.outlier_threshold})
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.01"
                  value={config.outlier_threshold}
                  onChange={(e) => setConfig(prev => ({ ...prev, outlier_threshold: parseFloat(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  文本匹配阈值 ({config.text_threshold})
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.01"
                  value={config.text_threshold}
                  onChange={(e) => setConfig(prev => ({ ...prev, text_threshold: parseFloat(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  混淆差距阈值 ({config.confusion_gap})
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="0.2"
                  step="0.01"
                  value={config.confusion_gap}
                  onChange={(e) => setConfig(prev => ({ ...prev, confusion_gap: parseFloat(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  角色最小图片数 ({config.min_images_per_character})
                </label>
                <input
                  type="range"
                  min="2"
                  max="20"
                  step="1"
                  value={config.min_images_per_character}
                  onChange={(e) => setConfig(prev => ({ ...prev, min_images_per_character: parseInt(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  并发线程数 ({config.max_workers})
                </label>
                <input
                  type="range"
                  min="1"
                  max="16"
                  step="1"
                  value={config.max_workers}
                  onChange={(e) => setConfig(prev => ({ ...prev, max_workers: parseInt(e.target.value) }))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 dark:bg-gray-700 accent-blue-500"
                />
              </div>
            </div>

            {/* 干运行模式 */}
            <label className={`flex items-center space-x-3 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-amber-50'}`}>
              <input
                type="checkbox"
                checked={config.dry_run}
                onChange={(e) => setConfig(prev => ({ ...prev, dry_run: e.target.checked }))}
                className="rounded border-gray-300 text-amber-600 focus:ring-amber-500"
              />
              <div>
                <span className={`text-sm font-medium ${darkMode ? 'text-amber-300' : 'text-amber-800'}`}>干运行模式</span>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-600'}`}>
                  仅预览清洗效果，不实际删除文件
                </p>
              </div>
            </label>
          </div>
        )}
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* 运行按钮 */}
      <div className="flex justify-center">
        <button
          onClick={runCleaning}
          disabled={isRunning}
          className={`flex items-center space-x-2 px-8 py-3 rounded-lg font-medium transition-all ${
            isRunning
              ? 'bg-gray-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-lg hover:shadow-xl'
          }`}
        >
          {isRunning ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>{isAsync ? '提交中...' : '清洗中...'}</span>
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              <span>开始清洗</span>
            </>
          )}
        </button>
      </div>

      {/* 数据清理进度面板 */}
      <div className={`rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <button
          onClick={() => setShowProgressPanel(!showProgressPanel)}
          className="w-full p-4 flex items-center justify-between"
        >
          <div className="flex items-center space-x-2">
            <RefreshCw className={`h-5 w-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
            <span className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>数据清理进度</span>
            {cleaningProgress && (
              <span className={`text-xs px-2 py-0.5 rounded ${darkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700'}`}>
                总样本: {cleaningProgress.total_samples}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                resetCleaningProgress();
              }}
              className={`p-1.5 rounded-lg transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`}
              title="重置进度"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            {showProgressPanel ? (
              <ChevronUp className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
            ) : (
              <ChevronDown className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
            )}
          </div>
        </button>

        {showProgressPanel && cleaningProgress && (
          <div className="p-4 pt-0 space-y-4">
            {/* 汇总统计卡片 */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                  {cleaningProgress.summary.total_processed}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>总处理数</div>
              </div>
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-green-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-green-400' : 'text-green-600'}`}>
                  {cleaningProgress.summary.total_valid}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>有效样本</div>
              </div>
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-red-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
                  {cleaningProgress.summary.total_rejected}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>已过滤</div>
              </div>
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-purple-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                  {cleaningProgress.summary.total_duplicates}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>重复数</div>
              </div>
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-yellow-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>
                  {cleaningProgress.summary.avg_confidence.toFixed(2)}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>平均置信度</div>
              </div>
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-cyan-50'}`}>
                <div className={`text-lg font-bold ${darkMode ? 'text-cyan-400' : 'text-cyan-600'}`}>
                  {cleaningProgress.summary.avg_quality_score.toFixed(2)}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>平均质量分</div>
              </div>
            </div>

            {/* 任务进度列表 */}
            <div className="space-y-3">
              {Object.entries(cleaningProgress.tasks).map(([taskId, task]) => (
                <div
                  key={taskId}
                  className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        {task.name}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        task.status === 'completed' ? (darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-700') :
                        task.status === 'running' ? (darkMode ? 'bg-blue-900 text-blue-400' : 'bg-blue-100 text-blue-700') :
                        task.status === 'failed' ? (darkMode ? 'bg-red-900 text-red-400' : 'bg-red-100 text-red-700') :
                        (darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-200 text-gray-600')
                      }`}>
                        {task.status === 'pending' ? '等待中' :
                         task.status === 'running' ? '运行中' :
                         task.status === 'completed' ? '已完成' : '失败'}
                      </span>
                    </div>
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {task.completed}/{task.total} ({task.progress.toFixed(1)}%)
                    </span>
                  </div>
                  <div className={`h-2 rounded-full overflow-hidden ${darkMode ? 'bg-gray-600' : 'bg-gray-200'}`}>
                    <div
                      className={`h-full transition-all duration-300 ${
                        task.status === 'completed' ? 'bg-gradient-to-r from-green-500 to-green-400' :
                        task.status === 'running' ? 'bg-gradient-to-r from-blue-500 to-blue-400 animate-pulse' :
                        task.status === 'failed' ? 'bg-gradient-to-r from-red-500 to-red-400' :
                        'bg-gray-400'
                      }`}
                      style={{ width: `${task.progress}%` }}
                    />
                  </div>
                  {task.message && (
                    <p className={`text-xs mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      {task.message}
                    </p>
                  )}
                </div>
              ))}
            </div>

            {/* 更新时间 */}
            <div className={`flex items-center justify-between text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              <span>更新时间: {new Date(cleaningProgress.last_updated).toLocaleString('zh-CN')}</span>
              <button
                onClick={() => {
                  setProgressPollInterval(progressPollInterval ? null : Date.now());
                }}
                className={`px-2 py-1 rounded transition-colors ${
                  progressPollInterval ? (darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-700') :
                  (darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-200 text-gray-600')
                }`}
              >
                {progressPollInterval ? '🔄 自动刷新中' : '⏸ 已暂停'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 当前任务状态（异步模式） */}
      {currentTaskStatus && (
        <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>当前任务状态</h3>
            <div className={`flex items-center space-x-2 ${getStatusColor(currentTaskStatus.status)}`}>
              {getStatusIcon(currentTaskStatus.status)}
              <span>{getStatusText(currentTaskStatus.status)}</span>
            </div>
          </div>
          
          {/* 任务ID和进度 */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>任务ID</span>
              <span className={`text-sm font-mono ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} title={currentTaskStatus.task_id}>
                {currentTaskStatus.task_id}
              </span>
            </div>
            
            {/* 进度条 */}
            {currentTaskStatus.progress_percent !== undefined && (
              <div className="mb-2">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>处理进度</span>
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
                    {currentTaskStatus.processed_files || 0}/{currentTaskStatus.total_files || 0} ({currentTaskStatus.progress_percent.toFixed(1)}%)
                  </span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-300 ${
                      currentTaskStatus.status === 'running' 
                        ? 'bg-gradient-to-r from-blue-500 to-purple-500' 
                        : currentTaskStatus.status === 'completed'
                        ? 'bg-gradient-to-r from-green-500 to-green-400'
                        : currentTaskStatus.status === 'failed'
                        ? 'bg-gradient-to-r from-red-500 to-red-400'
                        : 'bg-gray-400'
                    }`} 
                    style={{ width: `${currentTaskStatus.progress_percent}%` }} 
                  />
                </div>
              </div>
            )}
          </div>
          
          {/* 统计信息 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-4">
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>耗时</span>
              <span className={darkMode ? 'text-gray-200' : 'text-gray-800'}>{formatDuration(currentTaskStatus.duration_seconds)}</span>
            </div>
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>有效图片</span>
              <span className={`font-medium ${darkMode ? 'text-green-400' : 'text-green-600'}`}>{currentTaskStatus.valid_files || 0}</span>
            </div>
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>已移除</span>
              <span className={`font-medium ${darkMode ? 'text-red-400' : 'text-red-600'}`}>{currentTaskStatus.rejected_files || 0}</span>
            </div>
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>重复数</span>
              <span className={`font-medium ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>{currentTaskStatus.duplicate_files || 0}</span>
            </div>
          </div>
          
          {/* 目录信息 */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>输入目录</span>
              <span className={`truncate ${darkMode ? 'text-gray-200' : 'text-gray-800'}`} title={currentTaskStatus.input_dir}>{currentTaskStatus.input_dir}</span>
            </div>
            <div>
              <span className={`block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>输出目录</span>
              <span className={`truncate ${darkMode ? 'text-gray-200' : 'text-gray-800'}`} title={currentTaskStatus.output_dir}>{currentTaskStatus.output_dir}</span>
            </div>
          </div>
          
          {/* 错误信息 */}
          {currentTaskStatus.status === 'failed' && currentTaskStatus.error_message && (
            <div className="mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700">
              <div className="flex items-start space-x-2">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                <span className="text-sm text-red-700 dark:text-red-400">{currentTaskStatus.error_message}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 清洗结果 */}
      {lastResult && (
        <div className={`p-4 rounded-xl ${lastResult.success ? (darkMode ? 'bg-gray-800' : 'bg-green-50') : (darkMode ? 'bg-gray-800' : 'bg-red-50')} border ${lastResult.success ? (darkMode ? 'border-green-700' : 'border-green-200') : (darkMode ? 'border-red-700' : 'border-red-200')}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className={`font-medium ${lastResult.success ? (darkMode ? 'text-green-400' : 'text-green-800') : (darkMode ? 'text-red-400' : 'text-red-800')}`}>
              {lastResult.success ? '清洗完成' : '清洗失败'}
            </h3>
            {lastResult.success && lastResult.data?.report_path && (
              <button className={`flex items-center space-x-1 text-sm ${darkMode ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'}`}>
                <Download className="h-4 w-4" />
                <span>下载报告</span>
              </button>
            )}
          </div>

          {lastResult.success && lastResult.data && (
            <div className="space-y-4">
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {lastResult.message}
              </p>
              
              {/* 统计卡片 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}`}>
                  <div className={`text-2xl font-bold ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                    {lastResult.data.total_characters || 0}
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>角色数</div>
                </div>
                <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}`}>
                  <div className={`text-2xl font-bold ${darkMode ? 'text-green-400' : 'text-green-600'}`}>
                    {lastResult.data.total_cleaned_images || 0}
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>保留图片</div>
                </div>
                <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}`}>
                  <div className={`text-2xl font-bold ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
                    {lastResult.data.total_removed_images || 0}
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>移除图片</div>
                </div>
                <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'}`}>
                  <div className={`text-2xl font-bold ${darkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                    {((lastResult.data.overall_keep_rate || 0) * 100).toFixed(1)}%
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>保留率</div>
                </div>
              </div>

              {/* 各阶段移除统计 */}
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <div className="text-sm font-medium mb-2">各阶段移除统计</div>
                <div className="flex flex-wrap gap-3">
                  <span className="px-2 py-1 rounded text-xs bg-blue-100 text-blue-700">
                    CLIP去重: {lastResult.data.dedup_removed || 0}
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-green-100 text-green-700">
                    一致性过滤: {lastResult.data.consistency_removed || 0}
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-purple-100 text-purple-700">
                    HDBSCAN过滤: {lastResult.data.cluster_removed || 0}
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-red-100 text-red-700">
                    错误标签: {lastResult.data.mislabeled_removed || 0}
                  </span>
                </div>
              </div>

              {/* 耗时 */}
              <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                耗时: {formatDuration(lastResult.data.duration_seconds)}
              </div>
            </div>
          )}

          {!lastResult.success && lastResult.message && (
            <p className={`text-sm ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
              {lastResult.message}
            </p>
          )}
        </div>
      )}

      {/* 任务历史 */}
      <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <h3 className={`font-medium mb-4 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>任务历史</h3>
        {tasks.length === 0 ? (
          <EmptyState
            compact
            darkMode={darkMode}
            icon={<Clock className="h-6 w-6" />}
            title="暂无任务记录"
            description="运行一次清洗任务后，历史记录会显示在这里"
          />
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {tasks.map((task) => (
              <div
                key={task.task_id}
                className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${activeTaskId === task.task_id ? (darkMode ? 'border-blue-500' : 'border-blue-500') : 'border-transparent'}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(task.status)}
                    <div>
                      <div className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        {task.task_id}
                      </div>
                      <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {task.input_dir}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-xs ${getStatusColor(task.status)}`}>
                      {getStatusText(task.status)}
                    </span>
                    {task.status === 'completed' && task.result && (
                      <span className={`text-xs px-2 py-0.5 rounded ${darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-700'}`}>
                        {((task.result.overall_keep_rate || 0) * 100).toFixed(0)}%
                      </span>
                    )}
                    <button
                      onClick={() => deleteTask(task.task_id)}
                      className={`p-1 rounded hover:bg-red-100 dark:hover:bg-red-900 ${darkMode ? 'text-gray-400 hover:text-red-400' : 'text-gray-500 hover:text-red-500'}`}
                      title="删除任务"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 目录浏览器对话框 */}
      {showBrowser && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black bg-opacity-50">
          <div className={`w-full max-w-lg mx-4 rounded-xl shadow-2xl overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
            {/* 标题栏 */}
            <div className={`px-4 py-3 border-b flex items-center justify-between ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h3 className={`text-lg font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                选择{browseTarget === 'input' ? '输入' : '输出'}目录
              </h3>
              <button
                onClick={() => setShowBrowser(false)}
                className={`p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            {/* 路径栏 */}
            <div className={`px-4 py-2 flex items-center gap-2 border-b ${darkMode ? 'border-gray-700 bg-gray-750' : 'border-gray-100 bg-gray-50'}`}>
              <button
                onClick={navigateUp}
                disabled={browsePath === '/'}
                className={`p-1 rounded ${browsePath === '/' ? 'opacity-40 cursor-not-allowed' : 'hover:bg-gray-200 dark:hover:bg-gray-600'} ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
                title="上级目录"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className={`flex-1 text-xs truncate font-mono ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {browsePath}
              </span>
            </div>
            {/* 目录列表 */}
            <div className="max-h-72 overflow-y-auto px-2 py-2">
              {browseLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                </div>
              ) : browseEntries.length === 0 ? (
                <p className={`text-center py-8 text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  此目录下没有子目录
                </p>
              ) : (
                <div className="space-y-1">
                  {browseEntries.map(entry => (
                    <div
                      key={entry.path}
                      onClick={() => navigateToDir(entry.path)}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                        darkMode ? 'hover:bg-gray-700 text-gray-200' : 'hover:bg-gray-100 text-gray-700'
                      }`}
                    >
                      <Folder className="h-4 w-4 text-yellow-500 shrink-0" />
                      <span className="truncate flex-1">{entry.name}</span>
                      <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>文件夹</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {/* 底部按钮 */}
            <div className={`px-4 py-3 border-t flex justify-between items-center ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <button
                onClick={() => selectDirectory(browsePath)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  darkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                选择当前目录
              </button>
              <button
                onClick={() => setShowBrowser(false)}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${
                  darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}