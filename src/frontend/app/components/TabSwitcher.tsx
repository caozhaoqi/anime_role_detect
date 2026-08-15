"use client";

import { ImageIcon, Search, Video, Wand2 } from "lucide-react";

interface TabSwitcherProps {
  darkMode: boolean;
  activePanel: 'classify' | 'search' | 'video' | 'generate';
  onPanelChange: (panel: 'classify' | 'search' | 'video' | 'generate') => void;
}

type TabAccent = 'chat' | 'search' | 'video' | 'generate';

interface Tab {
  id: 'classify' | 'search' | 'video' | 'generate';
  icon: typeof ImageIcon | typeof Search | typeof Video | typeof Wand2;
  label: string;
  accent: TabAccent;
}

export default function TabSwitcher({ darkMode, activePanel, onPanelChange }: TabSwitcherProps) {
  const tabs: Tab[] = [
    { id: 'classify', icon: ImageIcon, label: '角色识别', accent: 'chat' },
    { id: 'search', icon: Search, label: '以图搜图', accent: 'search' },
    { id: 'video', icon: Video, label: '视频识别', accent: 'video' },
    { id: 'generate', icon: Wand2, label: '图像生成', accent: 'generate' },
  ];

  // 四强调色令牌（Tailwind v4 @theme 声明）：每个 tab 对应一个 accent
  const accentClasses: Record<TabAccent, { active: string; iconBgActive: string; iconBgInactive: string }> = {
    chat: {
      active: darkMode ? 'text-chat border-chat' : 'text-chat border-chat',
      iconBgActive: darkMode ? 'bg-chat/20 text-chat' : 'bg-chat text-white',
      iconBgInactive: darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-100 text-gray-500',
    },
    search: {
      active: darkMode ? 'text-search border-search' : 'text-search border-search',
      iconBgActive: darkMode ? 'bg-search/20 text-search' : 'bg-search text-white',
      iconBgInactive: darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-100 text-gray-500',
    },
    video: {
      active: darkMode ? 'text-video border-video' : 'text-video border-video',
      iconBgActive: darkMode ? 'bg-video/20 text-video' : 'bg-video text-white',
      iconBgInactive: darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-100 text-gray-500',
    },
    generate: {
      active: darkMode ? 'text-generate border-generate' : 'text-generate border-generate',
      iconBgActive: darkMode ? 'bg-generate/20 text-generate' : 'bg-generate text-white',
      iconBgInactive: darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-100 text-gray-500',
    },
  };

  const getClasses = (accent: TabAccent, isActive: boolean) => ({
    active: accentClasses[accent].active,
    inactive: darkMode ? 'text-gray-400' : 'text-gray-500',
    iconBg: isActive ? accentClasses[accent].iconBgActive : accentClasses[accent].iconBgInactive,
  });

  return (
    <div className={`sticky top-[6rem] z-40 ${darkMode ? 'bg-gray-900' : 'bg-white'} border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="container mx-auto px-4">
        <div className="flex space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activePanel === tab.id;
            const colors = getClasses(tab.accent, isActive);

            return (
              <button
                type="button"
                key={tab.id}
                onClick={() => onPanelChange(tab.id)}
                className={`flex items-center space-x-2 px-3 md:px-4 py-2.5 md:py-3 rounded-t-lg text-sm font-medium transition-all ${
                  isActive
                    ? darkMode ? 'bg-gray-800' : 'bg-gray-100'
                    : darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-50'
                } ${colors.active} ${isActive ? 'border-b-2' : ''}`}
              >
                <span className={`w-7 h-7 rounded-lg flex items-center justify-center transition-all transform ${isActive ? 'scale-110' : ''} ${colors.iconBg}`}>
                  <Icon className="h-4 w-4" />
                </span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}