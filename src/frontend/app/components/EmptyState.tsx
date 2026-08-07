"use client";

import React, { ReactNode } from "react";

interface EmptyStateProps {
  darkMode: boolean;
  icon: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  compact?: boolean;
}

/**
 * 通用空状态引导组件：虚线卡片 + 图标 + 标题 + 描述 + 可选操作按钮
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  darkMode,
  icon,
  title,
  description,
  action,
  compact = false,
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center rounded-xl border-2 border-dashed animate-fade-in transition-colors ${
        compact ? "py-8 px-4" : "py-12 px-6"
      } ${darkMode ? "border-gray-600 bg-gray-800/40" : "border-gray-200 bg-gray-50/60 hover:border-blue-300"}`}
    >
      <div
        className={`flex items-center justify-center rounded-2xl mb-4 transition-transform transform hover:scale-110 ${
          compact ? "w-12 h-12" : "w-16 h-16"
        } ${darkMode ? "bg-gray-700 text-blue-400" : "bg-blue-50 text-blue-500"}`}
      >
        {icon}
      </div>
      <h3 className={`font-semibold ${compact ? "text-sm" : "text-base"}`}>{title}</h3>
      {description && (
        <p className={`text-sm mt-1 max-w-md ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
};

export default EmptyState;
