import { apiClient } from '../client';

interface RecognitionOptions {
  useCoreML?: boolean;
  useModel?: boolean;
  useAttributes?: boolean;
  modelName?: string;
  multiRole?: boolean;
  threshold?: number;
  useYolo?: boolean;
}

interface RecognitionResponse {
  success: boolean;
  data: {
    role?: string;
    role_cn?: string;
    role_jp?: string;
    role_anime?: string;
    similarity?: number;
    confidence?: string;
    attributes?: Array<{ tag: string; confidence: number }>;
    tags?: string[];
    text_detections?: Array<{ text: string; confidence: number; bbox: number[] }>;
    ai_predicted_role?: string;
    nsfw?: { is_nsfw: boolean; skin_ratio: number; details?: Record<string, number> };
    possible_roles?: Array<{ role: string; probability: number }>;
    summary?: string;
    mode?: string;
    roles?: Array<{
      id?: number;
      role?: string;
      role_cn?: string;
      role_jp?: string;
      role_anime?: string;
      similarity?: number;
      confidence?: number;
      box?: { x1: number; y1: number; x2: number; y2: number };
      attributes?: Array<{ tag: string; confidence: number }>;
      decision?: string;
      is_unknown?: boolean;
      is_fuzzy?: boolean;
      bbox?: number[];
    }>;
    count?: number;
    detector?: string;
  };
  message?: string;
}

interface BatchRecognitionResponse {
  success: boolean;
  results?: Array<{
    filename?: string;
    role?: string;
    similarity?: number;
    attributes?: Array<{ tag: string; confidence: number }>;
    roles?: Array<{ role: string; similarity: number }>;
  }>;
  message?: string;
}

export class RecognitionService {
  static async classify(
    file: File,
    options: RecognitionOptions = {}
  ): Promise<RecognitionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_coreml', (options.useCoreML ?? false).toString());
    formData.append('use_model', (options.useModel ?? true).toString());
    formData.append('use_attributes', (options.useAttributes ?? true).toString());
    formData.append('model_name', options.modelName ?? 'efficientnet_b3');
    formData.append('multi_role', (options.multiRole ?? false).toString());
    formData.append('use_deepdanbooru', 'true');

    let endpoint = '/classify';
    if (options.useYolo) {
      endpoint = '/classify/yolo-detect';
    }

    const response = await apiClient.post<RecognitionResponse>(endpoint, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  static async batchClassify(
    files: File[],
    options: RecognitionOptions = {}
  ): Promise<BatchRecognitionResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file, file.name);
    });
    formData.append('model_name', options.modelName ?? 'default');
    formData.append('use_attributes', (options.useAttributes ?? true).toString());
    formData.append('batch_size', '8');
    formData.append('multilabel', (options.multiRole ?? false).toString());
    formData.append('threshold', (options.threshold ?? 0.4).toString());

    const response = await apiClient.post<BatchRecognitionResponse>('/model/batch-predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  static async getModels(): Promise<string[]> {
    try {
      const response = await apiClient.get<{ success: boolean; models?: string[] }>('/models');
      if (response.data.success) {
        return ['default', ...(response.data.models ?? [])];
      }
      return ['default'];
    } catch {
      return ['default'];
    }
  }
}