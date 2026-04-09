import { useState, useCallback } from 'react';

export interface HistoryItem {
  id: string;
  image?: string;
  result: {
    role: string;
    similarity: number;
    confidence: "high" | "medium" | "low";
    ai_predicted_role?: string;
    predicted_role?: string;
  };
  timestamp: number;
}

export const useHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>([]);
  const [filters, setFilters] = useState({
    role: '',
    timeRange: 'all' as 'all' | 'today' | 'week' | 'month',
  });

  const loadHistory = useCallback(() => {
    try {
      const savedHistory = localStorage.getItem('animeRoleDetectHistory');
      if (savedHistory) {
        const parsedHistory = JSON.parse(savedHistory);
        setHistory(parsedHistory);
        setFilteredHistory(parsedHistory);
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
    setFilteredHistory([]);
    saveHistory([]);
  }, [saveHistory]);

  const applyFilters = useCallback((newFilters: typeof filters) => {
    setFilters(newFilters);
    let filtered = [...history];

    // 按角色筛选
    if (newFilters.role) {
      filtered = filtered.filter(item => 
        item.result.role.toLowerCase().includes(newFilters.role.toLowerCase())
      );
    }

    // 按时间范围筛选
    const now = Date.now();
    const oneDay = 24 * 60 * 60 * 1000;
    const oneWeek = 7 * oneDay;
    const oneMonth = 30 * oneDay;

    switch (newFilters.timeRange) {
      case 'today':
        filtered = filtered.filter(item => now - item.timestamp <= oneDay);
        break;
      case 'week':
        filtered = filtered.filter(item => now - item.timestamp <= oneWeek);
        break;
      case 'month':
        filtered = filtered.filter(item => now - item.timestamp <= oneMonth);
        break;
      default:
        break;
    }

    setFilteredHistory(filtered);
  }, [history]);

  return {
    history,
    filteredHistory,
    filters,
    loadHistory,
    addToHistory,
    clearHistory,
    applyFilters,
  };
};
