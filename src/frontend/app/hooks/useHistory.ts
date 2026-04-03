import { useState, useCallback } from 'react';

export interface HistoryItem {
  id: string;
  image?: string;
  result: {
    role: string;
    similarity: number;
    confidence: "high" | "medium" | "low";
  };
  timestamp: number;
}

export const useHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const loadHistory = useCallback(() => {
    try {
      const savedHistory = localStorage.getItem('animeRoleDetectHistory');
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  }, []);

  const saveHistory = useCallback((newHistory: HistoryItem[]) => {
    try {
      localStorage.setItem('animeRoleDetectHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Failed to save history:', error);
    }
  }, []);

  const addToHistory = useCallback((item: { image?: File; result: { role: string; similarity: number; confidence: string }; timestamp: number }) => {
    const historyItem: HistoryItem = {
      id: Date.now().toString(),
      result: {
        role: item.result.role,
        similarity: item.result.similarity,
        confidence: item.result.confidence as "high" | "medium" | "low",
      },
      timestamp: item.timestamp,
    };

    const newHistory = [historyItem, ...history].slice(0, 20); // 只保留最近20条记录
    setHistory(newHistory);
    saveHistory(newHistory);
  }, [history, saveHistory]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
  }, [saveHistory]);

  return {
    history,
    loadHistory,
    addToHistory,
    clearHistory,
  };
};
