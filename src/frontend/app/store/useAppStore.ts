import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
  /** 自动消失时间（ms），默认 3000 */
  duration?: number;
}

interface AppState {
  /** 全局 Debug 模式开关：开启后请求后端生成 YOLO/分类辅助框标注图 */
  debugEnabled: boolean;
  setDebugEnabled: (value: boolean) => void;
  toggleDebug: () => void;

  /** 全局 Toast 队列 */
  toasts: ToastItem[];
  addToast: (message: string, type?: ToastType, duration?: number) => void;
  removeToast: (id: string) => void;
}

let toastSeq = 0;

/**
 * 轻量全局客户端状态（Zustand）。
 * 已落地：debug 模式开关 + 全局 Toast 通知（替代散落的 console.error/复制成功提示）。
 */
export const useAppStore = create<AppState>((set, get) => ({
  debugEnabled: false,
  setDebugEnabled: (value) => set({ debugEnabled: value }),
  toggleDebug: () => set((state) => ({ debugEnabled: !state.debugEnabled })),

  toasts: [],
  addToast: (message, type = 'info', duration = 3000) => {
    const id = `toast-${Date.now()}-${toastSeq++}`;
    set((state) => ({ toasts: [...state.toasts, { id, message, type, duration }] }));
    // 自动移除（组件卸载时由 removeToast 兜底）
    setTimeout(() => get().removeToast(id), duration);
  },
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));
