"use client";

import React, { useState, ReactNode, useId } from "react";
import { ChevronDown } from "lucide-react";

interface CollapsibleSectionProps {
  title: ReactNode;
  darkMode: boolean;
  defaultCollapsed?: boolean;
  dotColor?: string;
  badge?: ReactNode;
  children: ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  darkMode,
  defaultCollapsed = false,
  dotColor = "bg-blue-500",
  badge,
  children,
}) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const contentId = useId();

  return (
    <div className="mt-4 animate-fade-in">
      <button
        type="button"
        aria-expanded={!collapsed}
        aria-controls={contentId}
        onClick={() => setCollapsed(!collapsed)}
        className={`group flex items-center justify-between w-full rounded-lg px-3 py-2 transition-colors ${
          darkMode
            ? "bg-gray-700/60 hover:bg-gray-700 text-gray-200"
            : "bg-gray-100 hover:bg-gray-200 text-gray-800"
        }`}
      >
        <div className="flex items-center space-x-2 min-w-0">
          <div
            className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColor} ${
              !collapsed ? "animate-pulse" : "opacity-60"
            }`}
          />
          <h4 className="font-semibold text-sm truncate">{title}</h4>
          {badge}
        </div>
        <ChevronDown
          className={`h-4 w-4 flex-shrink-0 transition-transform duration-200 ${
            collapsed ? "-rotate-90" : "rotate-0"
          }`}
        />
      </button>
      {!collapsed && (
        <div
          id={contentId}
          role="region"
          className="mt-3 space-y-3 animate-fade-in"
        >
          {children}
        </div>
      )}
    </div>
  );
};

export default CollapsibleSection;
