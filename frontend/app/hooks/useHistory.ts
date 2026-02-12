import { useState, useCallback } from 'react';
import { Message } from '../types';

export const useHistory = () => {
  const [history, setHistory] = useState<Message[]>([]);

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

  const saveHistory = useCallback((newHistory: Message[]) => {
    try {
      localStorage.setItem('animeRoleDetectHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.error('Failed to save history:', error);
    }
  }, []);

  const addToHistory = useCallback((message: Message) => {
    if (message.role === 'assistant' && message.classification) {
      const newHistory = [message, ...history].slice(0, 20); // 只保留最近20条记录
      setHistory(newHistory);
      saveHistory(newHistory);
    }
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
