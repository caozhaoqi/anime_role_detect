// 统一「相似度(0~1) → 颜色语义」逻辑，消除多文件分裂实现。
// 阈值基准：>=0.8 高(green) / >=0.5 中(yellow) / 其余 低(red)

export type SimilarityLevel = 'high' | 'mid' | 'low';

export function similarityLevel(v: number): SimilarityLevel {
  if (v >= 0.8) return 'high';
  if (v >= 0.5) return 'mid';
  return 'low';
}

/** 进度条渐变填充（与 ConfidenceBar 组件配合） */
export function confidenceGradient(v: number): string {
  const l = similarityLevel(v);
  return l === 'high' ? 'from-green-500 to-green-600'
    : l === 'mid' ? 'from-yellow-500 to-yellow-600'
    : 'from-red-500 to-red-600';
}

/** 文本颜色（含 dark 变体） */
export function confidenceText(v: number): string {
  const l = similarityLevel(v);
  return l === 'high' ? 'text-green-600 dark:text-green-400'
    : l === 'mid' ? 'text-yellow-600 dark:text-yellow-400'
    : 'text-red-600 dark:text-red-400';
}

/** 小圆点底色（用于相似度指示点） */
export function similarityDotColor(v: number): string {
  const l = similarityLevel(v);
  return l === 'high' ? 'bg-green-500' : l === 'mid' ? 'bg-yellow-500' : 'bg-red-500';
}

/** 徽章底色+文字（用于结果标签，低分统一为红，不再用灰） */
export function similarityBadgeColor(v: number): string {
  const l = similarityLevel(v);
  return l === 'high' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    : l === 'mid' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
}
