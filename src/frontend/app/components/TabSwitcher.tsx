"use client";

import { ImageIcon, Search, Video, Sparkles } from "lucide-react";

interface TabSwitcherProps {
  darkMode: boolean;
  activePanel: 'classify' | 'search' | 'video' | 'cleaning';
  onPanelChange: (panel: 'classify' | 'search' | 'video' | 'cleaning') => void;
}

type TabColor = 'blue' | 'purple' | 'green' | 'orange';

interface Tab {
  id: 'classify' | 'search' | 'video' | 'cleaning';
  icon: typeof ImageIcon | typeof Search | typeof Video | typeof Sparkles;
  label: string;
  color: TabColor;
}

export default function TabSwitcher({ darkMode, activePanel, onPanelChange }: TabSwitcherProps) {
  const tabs: Tab[] = [
    { id: 'classify', icon: ImageIcon, label: '角色识别', color: 'blue' },
    { id: 'search', icon: Search, label: '以图搜图', color: 'purple' },
    { id: 'video', icon: Video, label: '视频识别', color: 'green' },
    { id: 'cleaning', icon: Sparkles, label: '数据清洗', color: 'orange' },
  ];

  const getColorClasses = (color: TabColor, isActive: boolean) => {
    const colorClasses: Record<TabColor, { active: string; inactive: string }> = {
      blue: {
        active: darkMode ? 'text-blue-400 border-blue-400' : 'text-blue-600 border-blue-600',
        inactive: darkMode ? 'text-gray-400' : 'text-gray-500',
      },
      purple: {
        active: darkMode ? 'text-purple-400 border-purple-400' : 'text-purple-600 border-purple-600',
        inactive: darkMode ? 'text-gray-400' : 'text-gray-500',
      },
      green: {
        active: darkMode ? 'text-green-400 border-green-400' : 'text-green-600 border-green-600',
        inactive: darkMode ? 'text-gray-400' : 'text-gray-500',
      },
      orange: {
        active: darkMode ? 'text-orange-400 border-orange-400' : 'text-orange-600 border-orange-600',
        inactive: darkMode ? 'text-gray-400' : 'text-gray-500',
      },
    };
    return isActive ? colorClasses[color].active : colorClasses[color].inactive;
  };

  return (
    <div className={`sticky top-[6rem] z-40 ${darkMode ? 'bg-gray-900' : 'bg-white'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="container mx-auto px-4">
        <div className="flex space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activePanel === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => onPanelChange(tab.id)}
                className={`flex items-center space-x-2 px-4 py-3 rounded-t-lg text-sm font-medium transition-colors ${
                  isActive
                    ? darkMode ? 'bg-gray-800' : 'bg-gray-100'
                    : darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-50'
                } ${getColorClasses(tab.color, isActive)} ${isActive ? 'border-b-2' : ''}`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}