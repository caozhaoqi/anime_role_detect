import { create } from 'zustand';

interface AppState {
  /** 全局 Debug 模式开关：开启后请求后端生成 YOLO/分类辅助框标注图 */
  debugEnabled: boolean;
  setDebugEnabled: (value: boolean) => void;
  toggleDebug: () => void;
}

/**
 * 轻量全局客户端状态（Zustand）。
 * 首个落地场景：debug 模式开关，从原先 page.tsx 的局部 useState + prop drilling 下沉到全局 store。
 */
export const useAppStore = create<AppState>((set) => ({
  debugEnabled: false,
  setDebugEnabled: (value) => set({ debugEnabled: value }),
  toggleDebug: () => set((state) => ({ debugEnabled: !state.debugEnabled })),
}));
