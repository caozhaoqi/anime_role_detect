"use client";

import React from "react";
import { CheckCircle2, XCircle, Info, X } from "lucide-react";
import { useAppStore, ToastType } from "../store/useAppStore";

const typeStyles: Record<ToastType, { bg: string; border: string; icon: React.ReactNode }> = {
  success: {
    bg: "bg-green-50 dark:bg-green-900/40",
    border: "border-green-200 dark:border-green-700",
    icon: <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />,
  },
  error: {
    bg: "bg-red-50 dark:bg-red-900/40",
    border: "border-red-200 dark:border-red-700",
    icon: <XCircle className="h-4 w-4 text-red-500 shrink-0" />,
  },
  info: {
    bg: "bg-blue-50 dark:bg-blue-900/40",
    border: "border-blue-200 dark:border-blue-700",
    icon: <Info className="h-4 w-4 text-blue-500 shrink-0" />,
  },
};

/**
 * 全局 Toast 容器：固定右下角，读取 useAppStore.toasts 渲染。
 * 挂载一次于根布局（page.tsx），各组件通过 addToast() 触发。
 */
const ToastContainer: React.FC = () => {
  const toasts = useAppStore((s) => s.toasts);
  const removeToast = useAppStore((s) => s.removeToast);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[10001] flex flex-col gap-2 w-80 max-w-[calc(100vw-3rem)]">
      {toasts.map((toast) => {
        const s = typeStyles[toast.type];
        return (
          <div
            key={toast.id}
            role="status"
            className={`${s.bg} ${s.border} border rounded-xl shadow-lg px-3 py-2.5 flex items-center gap-2 animate-fade-in`}
          >
            {s.icon}
            <span className="flex-1 text-sm font-medium break-words">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="shrink-0 p-0.5 rounded-full opacity-60 hover:opacity-100 transition-opacity"
              aria-label="关闭提示"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
};

export default ToastContainer;
