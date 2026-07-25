import { apiClient } from '../client';

export interface HistoryRecord {
  id: string;
  image_filename: string;
  recognition_result: {
    tags?: string[];
    attributes?: Array<{ tag: string; confidence: number }>;
    text_detections?: Array<{ text: string; confidence: number; bbox: number[] }>;
    nsfw?: { is_nsfw: boolean; skin_ratio: number };
    role_info?: any;
  };
  model_used: string;
  timestamp: number;
}

export class HistoryService {
  static async getHistory(): Promise<HistoryRecord[]> {
    try {
      const response = await apiClient.get<{ success: boolean; data?: HistoryRecord[] }>('/history');
      if (response.data.success) {
        return response.data.data ?? [];
      }
      return [];
    } catch {
      return [];
    }
  }

  static async getRecord(id: string): Promise<HistoryRecord | null> {
    try {
      const response = await apiClient.get<{ success: boolean; data?: HistoryRecord }>(
        `/history/${id}`
      );
      if (response.data.success) {
        return response.data.data ?? null;
      }
      return null;
    } catch {
      return null;
    }
  }

  static async deleteRecord(id: string): Promise<boolean> {
    try {
      const response = await apiClient.delete<{ success: boolean }>(`/history/${id}`);
      return response.data.success;
    } catch {
      return false;
    }
  }
}