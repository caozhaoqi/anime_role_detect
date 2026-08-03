import React, { useState } from 'react';
import { Bot, User, Copy, Download, ChevronDown, Flame, Pencil, Target, TriangleAlert, Bug, CheckCircle, Loader2 } from 'lucide-react';
import { Message } from '../types';
import CollapsibleSection from './CollapsibleSection';
import AutoCollapse from './AutoCollapse';
import { useGradcam } from '../hooks/useGradcam';
import { useFeedback } from '../hooks/useFeedback';

interface MessageItemProps {
  message: Message;
  darkMode: boolean;
  handleCopyMessage: (content: string) => void;
  handleDownloadMessage: (content: string, role: string) => void;
}

// ---- 置信度 → 颜色工具 ----
const confidenceGradient = (v: number) =>
  v >= 0.8 ? 'from-green-500 to-green-600' :
  v >= 0.5 ? 'from-yellow-500 to-yellow-600' : 'from-red-500 to-red-600';

const confidenceText = (v: number) =>
  v >= 0.8 ? 'text-green-600 dark:text-green-400' :
  v >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400';

/** 相似度进度条 */
const ConfidenceBar: React.FC<{ value: number; gradient?: string; className?: string }> = ({ value, gradient, className }) => {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className={`h-2 ${className ?? ''} rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden`}>
      <div
        className={`h-full rounded-full bg-gradient-to-r ${gradient ?? confidenceGradient(value)} transition-all duration-500`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
};

/** Grad-CAM 结果（可折叠，默认收起，新生成自动展开） */
const GradcamBlock: React.FC<{
  result: { target_label: string; confidence: number; cam_heatmap_base64: string };
  collapsed: boolean;
  onToggle: () => void;
}> = ({ result, collapsed, onToggle }) => (
  <div className="mt-2 rounded-lg overflow-hidden border border-orange-200 dark:border-orange-800">
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between gap-2 text-left px-2 py-1.5 text-xs text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/30 hover:bg-orange-100 dark:hover:bg-orange-900/50 transition-colors"
      title={collapsed ? '展开热力图' : '收起热力图'}
    >
      <span className="flex items-center gap-1 truncate">
        <Flame className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">Grad-CAM：模型关注区域（目标: {result.target_label}，置信度: {(result.confidence * 100).toFixed(1)}%）</span>
      </span>
      <ChevronDown className={`h-3.5 w-3.5 shrink-0 transition-transform ${collapsed ? '' : 'rotate-180'}`} />
    </button>
    {!collapsed && (
      <img src={result.cam_heatmap_base64} alt="grad-cam" className="w-full h-auto" />
    )}
  </div>
);

const MessageItem: React.FC<MessageItemProps> = ({ message, darkMode, handleCopyMessage, handleDownloadMessage }) => {
  const getCategoryInfo = (key: string) => {
    const categoryMap: Record<string, { label: string; color: string; bgColor: string }> = {
      'drawings': { label: '绘画', color: 'text-purple-500', bgColor: 'bg-purple-500' },
      'hentai': { label: '色情动漫', color: 'text-pink-500', bgColor: 'bg-pink-500' },
      'neutral': { label: '正常', color: 'text-gray-500', bgColor: 'bg-gray-500' },
      'porn': { label: '色情', color: 'text-red-500', bgColor: 'bg-red-500' },
      'sexy': { label: '性感', color: 'text-orange-500', bgColor: 'bg-orange-500' }
    };
    return categoryMap[key] || { label: key, color: 'text-blue-500', bgColor: 'bg-blue-500' };
  };

  const getHighestCategory = (details?: Record<string, number>): string => {
    if (!details) return 'unknown';
    const entries = Object.entries(details);
    if (entries.length === 0) return 'unknown';
    return entries.reduce((a, b) => Number(a[1]) > Number(b[1]) ? a : b)[0];
  };

  // Phase1: Grad-CAM 热力图 + 纠错反馈
  const gradcam = useGradcam();
  const feedback = useFeedback();
  const [gradcamLoading, setGradcamLoading] = useState<number | null>(null);
  const [gradcamResult, setGradcamResult] = useState<{ [key: number]: any }>({});
  const [gradcamCollapsed, setGradcamCollapsed] = useState<{ [key: number]: boolean }>({});
  const [correctingRole, setCorrectingRole] = useState<number | null>(null);
  const [correctionSelect, setCorrectionSelect] = useState<string>('');
  const [submittedCorrections, setSubmittedCorrections] = useState<Set<number>>(new Set());

  // 原图 base64 → File（gradcam 端点需要 file upload）
  const imageToFile = async (): Promise<File | null> => {
    try {
      if (!message.image) return null;
      const res = await fetch(message.image);
      const blob = await res.blob();
      return new File([blob], 'image.jpg', { type: blob.type || 'image/jpeg' });
    } catch { return null; }
  };

  // 触发 Grad-CAM
  const handleGradcam = async (roleIndex: number, targetClass?: number) => {
    setGradcamLoading(roleIndex);
    const file = await imageToFile();
    if (!file) { setGradcamLoading(null); return; }
    const result = await gradcam.generate(file, targetClass);
    if (result) {
      setGradcamResult(prev => ({ ...prev, [roleIndex]: result }));
      // 新生成的热力图自动展开，便于用户立即查看
      setGradcamCollapsed(prev => ({ ...prev, [roleIndex]: false }));
    }
    setGradcamLoading(null);
  };

  // 提交纠错
  const handleCorrection = async (roleIndex: number, originalRole: string, confidence: number) => {
    if (!correctionSelect || correctionSelect === originalRole) return;
    const ok = await feedback.submit({
      recognition_id: message.id,
      endpoint: message.multi_roles ? 'multi-role' : 'yolo-detect',
      original_prediction: originalRole,
      original_confidence: confidence,
      corrected_label: correctionSelect,
      image_data: message.image || undefined,
      timestamp: new Date().toISOString(),
    });
    if (ok) {
      setSubmittedCorrections(prev => new Set(prev).add(roleIndex));
      setCorrectingRole(null);
      setCorrectionSelect('');
    }
  };

  // 独立 debug 面板统计（不依赖 classification / multi_roles 是否存在）
  const debugStats = message.debug && message.debug.enabled ? (() => {
    const boxes = message.debug!.boxes;
    const keptKnown = boxes.filter((b) => b.kept && b.is_known_character).length;
    const keptUnknown = boxes.filter((b) => b.kept && !b.is_known_character).length;
    const filtered = boxes.filter((b) => !b.kept).length;
    const discardCounts: Record<string, number> = {};
    boxes.forEach((b) => {
      if (!b.kept && b.discard_reason) {
        discardCounts[b.discard_reason] = (discardCounts[b.discard_reason] || 0) + 1;
      }
    });
    return { keptKnown, keptUnknown, filtered, discardCounts };
  })() : null;

  return (
    <div
      key={message.id}
      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
    >
      <div
        className={`flex-shrink-0 mr-2 ml-2 ${message.role === "user" ? "order-2" : "order-1"}`}
      >
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${message.role === "user" ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')} transition-transform hover:scale-110`}>
          {message.role === "user" ? (
            <User className="h-5 w-5" />
          ) : (
            <Bot className="h-5 w-5" />
          )}
        </div>
      </div>
      <div
        className={`max-w-full ${message.role === "user" ? "order-1" : "order-2"}`}
      >
        <div
          className={`rounded-xl p-3 ${message.role === "user" ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' : (darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-100 text-gray-900')} shadow-sm transition-all hover:shadow-md`}
        >
          {message.image && (
            <div className="mb-3 rounded-lg overflow-hidden shadow-md transform hover:scale-[1.02] transition-transform">
              <img
                src={message.image}
                alt="User uploaded image"
                className="w-full h-auto object-cover"
              />
            </div>
          )}
          {message.content && (
            <AutoCollapse
              maxHeight={220}
              overlayFromClass={
                message.role === "user"
                  ? "from-blue-600 dark:from-blue-600"
                  : darkMode
                  ? "from-gray-700 dark:from-gray-700"
                  : "from-gray-100 dark:from-gray-100"
              }
            >
              <p className="whitespace-pre-wrap break-words mb-0">{message.content}</p>
            </AutoCollapse>
          )}

          {message.classification && (
            <div className={`mt-3 p-3 md:p-4 rounded-2xl border-2 animate-fade-in ${
              darkMode ? 'border-blue-500/40 bg-blue-500/5' : 'border-blue-400 bg-blue-50/80'
            } shadow-lg`}>
              <div className="flex items-center space-x-2 mb-3">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse" />
                <Target className="h-4 w-4 text-blue-500" />
                <h3 className="font-bold text-base">识别结果</h3>
                <span className={`ml-auto px-2 py-0.5 text-xs rounded-full ${darkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100 text-blue-600'}`}>主体</span>
              </div>
              <div className={`grid grid-cols-1 sm:grid-cols-2 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-4 ${darkMode ? 'bg-gray-600' : 'bg-white'} rounded-xl shadow-sm transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">角色</p>
                  <p className="text-xl md:text-2xl font-bold leading-tight break-words">{message.classification.role}</p>
                  {message.classification.role_cn && message.classification.role_cn !== message.classification.role && (
                    <p className="text-sm text-blue-500 mt-2">{message.classification.role_cn}</p>
                  )}
                  {message.classification.role_jp && (
                    <p className="text-sm text-pink-500 mt-1">{message.classification.role_jp}</p>
                  )}
                  {message.classification.role_anime && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{message.classification.role_anime}</p>
                  )}
                </div>
                <div className={`p-4 ${darkMode ? 'bg-gray-600' : 'bg-white'} rounded-xl shadow-sm`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">相似度</p>
                  <p className="text-2xl md:text-3xl font-extrabold text-blue-500 leading-none">{(message.classification.similarity * 100).toFixed(1)}%</p>
                  <div className={`mt-3 h-2.5 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded-full overflow-hidden`}>
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(100, message.classification.similarity * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              {/* Phase1: Grad-CAM 热力图 + 纠错反馈 */}
              <div className="flex items-center space-x-2 flex-wrap gap-y-2 mt-4">
                {message.classification?.used_model !== false && (
                <button
                  onClick={() => handleGradcam(-1)}
                  disabled={gradcamLoading === -1}
                  className="px-2 py-0.5 text-xs rounded-full bg-orange-100 text-orange-600 dark:bg-orange-900/50 dark:text-orange-400 hover:scale-105 transition-transform disabled:opacity-50"
                  title="查看 Grad-CAM 热力图"
                >
                  {gradcamLoading === -1 ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin inline mr-1" />
                      生成中...
                    </>
                  ) : (
                    <>
                      <Flame className="h-3 w-3 inline mr-1" />
                      热力图
                    </>
                  )}
                </button>
                )}
                {submittedCorrections.has(-1) ? (
                  <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-600 dark:bg-green-900/50 dark:text-green-400">
                    ✓ 已纠错
                  </span>
                ) : correctingRole === -1 ? (
                  <div className="flex items-center space-x-1">
                    <select
                      value={correctionSelect}
                      onChange={(e) => setCorrectionSelect(e.target.value)}
                      className={`text-xs rounded border ${darkMode ? 'bg-gray-700 border-gray-600 text-gray-100' : 'bg-white border-gray-300 text-gray-900'} px-1 py-0.5 max-w-[140px]`}
                    >
                      <option value="">选择正确角色...</option>
                      {feedback.roles.map(r => <option key={r.idx} value={r.name}>{r.name}</option>)}
                    </select>
                    <button
                      onClick={() => handleCorrection(-1, message.classification!.role, message.classification!.similarity)}
                      disabled={!correctionSelect || feedback.submitting}
                      className="px-2 py-0.5 text-xs rounded bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50"
                    >提交</button>
                    <button onClick={() => { setCorrectingRole(null); setCorrectionSelect(''); }}
                      className="px-1 py-0.5 text-xs rounded text-gray-500 hover:text-gray-700">✕</button>
                  </div>
                ) : (
                  <button
                    onClick={() => { setCorrectingRole(-1); setCorrectionSelect(''); }}
                    className="px-2 py-0.5 text-xs rounded-full bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300 hover:scale-105 transition-transform"
                    title="纠错：这个角色识别错了？"
                  >
                    <Pencil className="h-3 w-3 inline mr-1" />
                    纠错
                  </button>
                )}
              </div>
              {gradcamResult[-1] && (
                <GradcamBlock
                  result={gradcamResult[-1]}
                  collapsed={gradcamCollapsed[-1] ?? true}
                  onToggle={() => setGradcamCollapsed(prev => ({ ...prev, [-1]: !(prev[-1] ?? true) }))}
                />
              )}
            </div>
          )}

          {message.multi_roles && message.multi_roles.length > 0 && (
            <div className={`mt-3 p-3 md:p-4 rounded-2xl border-2 animate-fade-in ${
              darkMode ? 'border-indigo-500/40 bg-indigo-500/5' : 'border-indigo-300 bg-indigo-50/70'
            } shadow-lg`}>
              <div className="flex items-center space-x-2 mb-3">
                <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
                <Target className="h-4 w-4 text-indigo-500" />
                <h3 className="font-bold text-base">多角色识别结果</h3>
                <span className={`ml-auto px-2 py-0.5 text-xs rounded-full ${darkMode ? 'bg-indigo-900/50 text-indigo-300' : 'bg-indigo-100 text-indigo-600'}`}>{message.multi_roles.length} 个角色</span>
              </div>
              {message.fallback && (
                <div className="mt-2 mb-2 p-2 rounded-lg bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300 text-xs flex items-center space-x-1">
                  <TriangleAlert className="h-3.5 w-3.5 shrink-0" />
                  <span>未检出多个人体框，已使用整图识别（单角色兜底），结果仅供参考。</span>
                </div>
              )}
              <div className="space-y-2">
                {message.multi_roles.map((role, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-white'} rounded-xl shadow-sm transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-start">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                          <p className="text-base font-semibold break-words">{role.role}</p>
                          {role.is_unknown && (
                            <span className="px-2 py-0.5 text-xs bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400 rounded-full">未知</span>
                          )}
                          {role.is_fuzzy && !role.is_unknown && (
                            <span className="px-2 py-0.5 text-xs bg-yellow-100 text-yellow-600 dark:bg-yellow-900 dark:text-yellow-400 rounded-full">模糊</span>
                          )}
                          {role.decision === "known" && !role.is_fuzzy && !role.is_unknown && (
                            <span className="px-2 py-0.5 text-xs bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400 rounded-full">已知</span>
                          )}
                          {/* Phase1: Grad-CAM 热力图按钮 */}
                          {role.used_model !== false && (
                          <button
                            onClick={() => handleGradcam(index)}
                            disabled={gradcamLoading === index}
                            className="px-2 py-0.5 text-xs rounded-full bg-orange-100 text-orange-600 dark:bg-orange-900/50 dark:text-orange-400 hover:scale-105 transition-transform disabled:opacity-50"
                            title="查看 Grad-CAM 热力图"
                          >
                            {gradcamLoading === index ? (
                              <>
                                <Loader2 className="h-3 w-3 animate-spin inline mr-1" />
                                生成中...
                              </>
                            ) : (
                              <>
                                <Flame className="h-3 w-3 inline mr-1" />
                                热力图
                              </>
                            )}
                          </button>
                          )}
                          {/* Phase1: 纠错反馈 */}
                          {submittedCorrections.has(index) ? (
                            <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-600 dark:bg-green-900/50 dark:text-green-400">
                              ✓ 已纠错
                            </span>
                          ) : correctingRole === index ? (
                            <div className="flex items-center space-x-1">
                              <select
                                value={correctionSelect}
                                onChange={(e) => setCorrectionSelect(e.target.value)}
                                className={`text-xs rounded border ${darkMode ? 'bg-gray-700 border-gray-600 text-gray-100' : 'bg-white border-gray-300 text-gray-900'} px-1 py-0.5 max-w-[140px]`}
                              >
                                <option value="">选择正确角色...</option>
                                {feedback.roles.map(r => <option key={r.idx} value={r.name}>{r.name}</option>)}
                              </select>
                              <button
                                onClick={() => handleCorrection(index, role.role, role.similarity)}
                                disabled={!correctionSelect || feedback.submitting}
                                className="px-2 py-0.5 text-xs rounded bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50"
                              >提交</button>
                              <button onClick={() => { setCorrectingRole(null); setCorrectionSelect(''); }}
                                className="px-1 py-0.5 text-xs rounded text-gray-500 hover:text-gray-700">✕</button>
                            </div>
                          ) : (
                            <button
                              onClick={() => { setCorrectingRole(index); setCorrectionSelect(''); }}
                              className="px-2 py-0.5 text-xs rounded-full bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300 hover:scale-105 transition-transform"
                              title="纠错：这个角色识别错了？"
                            >
                              <Pencil className="h-3 w-3 inline mr-1" />
                              纠错
                            </button>
                          )}
                        </div>
                        {role.role_cn && role.role_cn !== role.role && (
                          <p className="text-sm text-blue-500 mt-1">{role.role_cn}</p>
                        )}
                        {role.role_jp && (
                          <p className="text-sm text-pink-500 mt-1">{role.role_jp}</p>
                        )}
                        {role.role_anime && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{role.role_anime}</p>
                        )}
                      </div>
                      <div className="flex flex-col items-end space-y-1 flex-shrink-0">
                        <p className={`text-lg font-bold ${confidenceText(role.similarity)}`}>{(role.similarity * 100).toFixed(1)}%</p>
                        <ConfidenceBar value={role.similarity} className="w-24" />
                      </div>
                    </div>
                    {gradcamResult[index] && (
                      <GradcamBlock
                        result={gradcamResult[index]}
                        collapsed={gradcamCollapsed[index] ?? true}
                        onToggle={() => setGradcamCollapsed(prev => ({ ...prev, [index]: !(prev[index] ?? true) }))}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {debugStats && (
            <CollapsibleSection
              title={
                <span className="flex items-center gap-1.5">
                  <Bug className="h-3.5 w-3.5 text-purple-500" />
                  Debug 辅助框
                </span>
              }
              darkMode={darkMode}
              defaultCollapsed
              dotColor="bg-purple-500"
              badge={
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-600 dark:bg-purple-900/50 dark:text-purple-400 shrink-0">
                  调试视图
                </span>
              }
            >

              {message.debug!.degraded_path && (
                <div className="mt-2 mb-3 p-2 rounded-lg bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300 text-xs flex items-center space-x-1">
                  <TriangleAlert className="h-3.5 w-3.5 shrink-0" />
                  <span>降级路径（未检测到人体，使用整图分类）</span>
                </div>
              )}

              {message.debug!.annotated_image && (
                <div className="mt-3 rounded-lg overflow-hidden shadow-md border border-gray-200 dark:border-gray-600">
                  <img
                    src={message.debug!.annotated_image}
                    alt="debug annotated"
                    className="w-full h-auto object-contain"
                  />
                </div>
              )}

              <div className={`grid grid-cols-3 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">🟢 保留已知</p>
                  <p className="text-sm font-medium">{debugStats.keptKnown}</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">🟡 保留未知</p>
                  <p className="text-sm font-medium">{debugStats.keptUnknown}</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">🔴 被过滤</p>
                  <p className="text-sm font-medium">{debugStats.filtered}</p>
                </div>
              </div>

              {message.debug!.boxes && message.debug!.boxes.length > 0 && (
                <div className="mt-3 space-y-1">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">逐框置信度</p>
                  <div className={`rounded-lg ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} p-2 space-y-1 max-h-48 overflow-y-auto`}>
                    {message.debug!.boxes.map((b, i) => {
                      const dot = b.kept
                        ? b.is_known_character
                          ? '🟢'
                          : '🟡'
                        : '🔴';
                      const reason =
                        !b.kept && b.discard_reason ? `（${b.discard_reason}）` : '';
                      return (
                        <div key={i} className="flex items-center space-x-2 text-xs">
                          <span className="font-mono text-gray-400">#{i + 1}</span>
                          <span>{dot}</span>
                          <span className="font-medium text-gray-700 dark:text-gray-300">
                            {(b.raw_confidence * 100).toFixed(1)}%
                          </span>
                          {reason && (
                            <span className="text-red-500 dark:text-red-400">{reason}</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>
                  YOLO 原始框总数:{' '}
                  <span className="font-medium text-gray-700 dark:text-gray-300">
                    {message.debug!.yolo_total_boxes}
                  </span>
                </span>
              </div>

              {Object.keys(debugStats.discardCounts).length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {Object.entries(debugStats.discardCounts).map(([reason, count]) => (
                    <span
                      key={reason}
                      className="px-2 py-1 text-xs bg-red-100 text-red-600 dark:bg-red-900/50 dark:text-red-400 rounded-full"
                    >
                      {reason}: {count}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400 pt-1 border-t border-gray-200 dark:border-gray-600">
                <span>🟢 保留且已知角色</span>
                <span>🟡 保留但被判未知（开集兜底）</span>
                <span>🔴 被阈值/未知过滤丢弃</span>
              </div>
            </CollapsibleSection>
          )}

          {message.attributes && message.attributes.length > 0 && (
            <CollapsibleSection title="角色属性" darkMode={darkMode} defaultCollapsed dotColor="bg-blue-500" badge={<span className={`px-2 py-0.5 text-xs rounded-full ${darkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100 text-blue-600'}`}>{message.attributes.length}</span>}>
              <div className="flex flex-wrap gap-2">
                {message.attributes.map((attr, index) => (
                  <span
                    key={index}
                    className={`px-4 py-2 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-sm font-medium transform hover:scale-105 transition-transform`}
                  >
                    {attr.tag}
                  </span>
                ))}
              </div>
            </CollapsibleSection>
          )}

          {/* 只有当消息包含图片或识别结果时才显示文本检测 */}
          {(message.image || message.classification || message.multi_roles || message.attributes) && (
            <CollapsibleSection title="文本检测" darkMode={darkMode} defaultCollapsed dotColor="bg-blue-500">
              {message.text_detections && message.text_detections.length > 0 ? (
                <div className="space-y-2">
                  {message.text_detections.map((text, index) => (
                    <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                      <p className="text-sm font-medium">{text.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg`}>
                  <p className="text-sm font-medium">图片中无文字</p>
                </div>
              )}
            </CollapsibleSection>
          )}

          {message.ai_predicted_role && (
            <CollapsibleSection title="AI 预测角色" darkMode={darkMode} defaultCollapsed dotColor="bg-green-500">
              <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                <p className="text-sm font-medium">{message.ai_predicted_role}</p>
              </div>
            </CollapsibleSection>
          )}

          {message.thoughts && !message.isThinkingFinished && (
            <CollapsibleSection title="识别过程" darkMode={darkMode} defaultCollapsed dotColor="bg-blue-500">
              <div className="space-y-2">
                {message.thoughts.map((thought, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                    <p className="text-sm">{thought}</p>
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}

          {message.nsfw && (
            <CollapsibleSection
              title="NSFW 内容检测"
              darkMode={darkMode}
              defaultCollapsed
              dotColor={message.nsfw.is_nsfw ? "bg-red-500" : "bg-green-500"}
              badge={
                <span className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 flex items-center gap-1 ${
                  message.nsfw.is_nsfw
                    ? 'bg-red-100 text-red-600 dark:bg-red-900/50 dark:text-red-400'
                    : 'bg-green-100 text-green-600 dark:bg-green-900/50 dark:text-green-400'
                }`}>
                  {message.nsfw.is_nsfw ? (
                    <>
                      <TriangleAlert className="h-3 w-3" />
                      包含敏感内容
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-3 w-3" />
                      安全内容
                    </>
                  )}
                </span>
              }
            >
              <div className={`grid grid-cols-3 gap-3 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                <div className={`p-3 ${message.nsfw.is_nsfw ? 'bg-red-900/20 border border-red-800' : darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">检测结果</p>
                  <div className="flex items-center space-x-2">
                    <p className="text-sm font-medium">
                      {message.nsfw.is_nsfw ? "NSFW" : "安全"}
                    </p>
                    <div
                      className={`w-2 h-2 rounded-full ${message.nsfw.is_nsfw ? "bg-red-500 animate-pulse" : "bg-green-500"}`}
                    />
                  </div>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">皮肤比例</p>
                  <p className="text-sm font-medium">{(message.nsfw.skin_ratio * 100).toFixed(1)}%</p>
                </div>
                <div className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">预测类别</p>
                  <p className="text-sm font-medium">
                    {getCategoryInfo(getHighestCategory(message.nsfw.details)).label}
                  </p>
                </div>
              </div>
              {message.nsfw.details && (
                <div className="mt-3 space-y-2">
                  <h5 className="text-xs text-gray-500 dark:text-gray-400">各类别概率分布</h5>
                  <div className="space-y-2">
                    {Object.entries(message.nsfw.details)
                      .sort(([, a], [, b]) => Number(b) - Number(a))
                      .map(([key, value]) => {
                        const percentage = (Number(value) * 100).toFixed(1);
                        const categoryInfo = getCategoryInfo(key);
                        const details = message.nsfw?.details;
                        const isHighest = details ? key === getHighestCategory(details) : false;
                        const isUnsafe = key === 'porn' || key === 'sexy' || key === 'hentai';

                        return (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between items-center">
                              <span className={`text-xs font-medium ${
                                isUnsafe ? categoryInfo.color : (darkMode ? 'text-gray-300' : 'text-gray-700')
                              }`}>
                                {categoryInfo.label}
                              </span>
                              <span className={`text-xs font-medium ${
                                isHighest ? 'text-blue-500 dark:text-blue-400' : ''
                              }`}>
                                {percentage}%
                              </span>
                            </div>
                            <div className={`h-2 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded-full overflow-hidden`}>
                              <div
                                className={`h-full ${categoryInfo.bgColor} rounded-full transition-all duration-500`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </CollapsibleSection>
          )}

          {message.tags && message.tags.length > 0 && (
            <CollapsibleSection title="标签" darkMode={darkMode} defaultCollapsed dotColor="bg-purple-500" badge={<span className={`px-2 py-0.5 text-xs rounded-full ${darkMode ? 'bg-purple-900/50 text-purple-300' : 'bg-purple-100 text-purple-600'}`}>{message.tags.length}</span>}>
              <div className="flex flex-wrap gap-2">
                {message.tags.map((tag, index) => (
                  <span
                    key={index}
                    className={`px-4 py-2 ${darkMode ? 'bg-purple-900/50 text-purple-400' : 'bg-purple-100 text-purple-600'} rounded-full text-sm font-medium transform hover:scale-105 transition-transform`}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </CollapsibleSection>
          )}

          {message.possible_roles && message.possible_roles.length > 0 && (
            <CollapsibleSection title="其他模型检测结果" darkMode={darkMode} defaultCollapsed dotColor="bg-blue-500">
              <div className="space-y-2">
                {message.possible_roles.map((role, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center">
                      <p className="text-sm font-medium">{role.role}</p>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm">{(role.probability * 100).toFixed(1)}%</p>
                        <div
                          className={`w-2 h-2 rounded-full ${role.probability >= 0.8 ? "bg-green-500" : role.probability >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}
          
          {message.batch_results && message.batch_results.length > 0 && (
            <CollapsibleSection title="批量识别结果" darkMode={darkMode} defaultCollapsed dotColor="bg-blue-500" badge={<span className={`px-2 py-0.5 text-xs rounded-full ${darkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100 text-blue-600'}`}>{message.batch_results.length}</span>}>
              <div className="space-y-3">
                {message.batch_results.map((result, index) => (
                  <div key={index} className={`p-3 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} rounded-lg transform hover:scale-[1.02] transition-transform`}>
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-sm font-medium">{result.filename}</p>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm">{(result.similarity * 100).toFixed(1)}%</p>
                        <div
                          className={`w-2 h-2 rounded-full ${result.similarity >= 0.8 ? "bg-green-500" : result.similarity >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`}
                        />
                      </div>
                    </div>
                    <p className="text-sm">角色: {result.role}</p>
                    {result.roles && result.roles.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">多角色识别:</p>
                        <div className="flex flex-wrap gap-2">
                          {result.roles.map((role, roleIndex) => (
                            <span
                              key={roleIndex}
                              className={`px-3 py-1 ${darkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-blue-600'} rounded-full text-xs font-medium`}
                            >
                              {role.role} ({(role.similarity * 100).toFixed(0)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}

          <div className="flex items-center justify-between mt-3 text-xs text-gray-400 dark:text-gray-500">
            <span suppressHydrationWarning={true}>{new Date(message.timestamp).toLocaleTimeString()}</span>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleCopyMessage(message.content)}
                className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="复制内容"
              >
                <Copy className="h-3 w-3" />
              </button>
              <button
                onClick={() => handleDownloadMessage(message.content, message.role)}
                className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} transition-colors transform hover:scale-110`}
                title="下载内容"
              >
                <Download className="h-3 w-3" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageItem;