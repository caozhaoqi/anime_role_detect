"use client";

import React, { useRef, useState, useLayoutEffect, ReactNode } from "react";
import { ChevronDown } from "lucide-react";

interface AutoCollapseProps {
  children: ReactNode;
  /** 折叠时最大高度（px），默认 180 */
  maxHeight?: number;
  /** 渐变遮罩起始颜色类（跟随气泡/卡片背景色），如 "from-gray-100 dark:from-gray-700" */
  overlayFromClass?: string;
  className?: string;
}

/**
 * 长内容自动折叠：内容超过 maxHeight 时折叠 + 底部渐变 + "展开全部"按钮。
 * 折叠状态由 scrollHeight 实测判定，与字符数无关，适配任意内容类型。
 */
const AutoCollapse: React.FC<AutoCollapseProps> = ({
  children,
  maxHeight = 180,
  overlayFromClass = "from-gray-100 dark:from-gray-700",
  className = "",
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isOverflow, setIsOverflow] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // 先折叠再测量，确保判定的是折叠态是否溢出
    setIsOverflow(el.scrollHeight > el.clientHeight);
  }, [children, maxHeight]);

  return (
    <div className={className}>
      <div
        ref={ref}
        className="relative overflow-hidden transition-[max-height] duration-300 ease-in-out"
        style={{ maxHeight: expanded ? undefined : maxHeight }}
      >
        {children}
        {!expanded && isOverflow && (
          <div
            className={`pointer-events-none absolute bottom-0 left-0 right-0 h-14 bg-gradient-to-t ${overlayFromClass} to-transparent`}
          />
        )}
      </div>
      {isOverflow && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="mt-1 flex items-center space-x-1 text-xs font-medium text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
        >
          <span>{expanded ? "收起" : "展开全部"}</span>
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
          />
        </button>
      )}
    </div>
  );
};

export default AutoCollapse;
