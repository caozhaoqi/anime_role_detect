export interface YoloDetection {
  id: number;
  role: string;
  role_cn?: string;
  role_jp?: string;
  role_anime?: string;
  confidence: number;
  person_confidence: number;
  bbox: number[];
  class_id: number;
}

export interface YoloDetectionResponse {
  roles: YoloDetection[];
  count: number;
  image_size: [number, number];
  detector: string;
  model: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: string | null;
  classification?: {
    role: string;
    role_cn?: string;
    role_jp?: string;
    role_anime?: string;
    similarity: number;
    confidence: "high" | "medium" | "low";
  };
  attributes?: Array<{
    tag: string;
    confidence: number;
  }>;
  text_detections?: Array<{
    text: string;
    confidence: number;
    bbox: number[];
  }>;
  ai_predicted_role?: string;
  nsfw?: {
    is_nsfw: boolean;
    skin_ratio: number;
    details?: {
      [key: string]: number;
    };
  };
  possible_roles?: Array<{
    role: string;
    probability: number;
  }>;
  tags?: string[];
  multi_roles?: Array<{
    role: string;
    role_cn?: string;
    role_jp?: string;
    role_anime?: string;
    similarity: number;
    attributes?: Array<{
      tag: string;
      confidence: number;
    }>;
    bbox: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
    };
    confidence: number;
    decision?: string;
    is_unknown?: boolean;
    is_fuzzy?: boolean;
  }>;
  role_info?: any;
  model_name?: string;
  summary?: string;
  thoughts?: string[];
  isThinking?: boolean;
  isThinkingFinished?: boolean;
  error?: string;
  batch_results?: Array<{
    id: number;
    filename: string;
    role: string;
    similarity: number;
    attributes?: Array<{
      tag: string;
      confidence: number;
    }>;
    roles?: Array<{
      role: string;
      similarity: number;
    }>;
  }>;
  timestamp: number;
}

export interface Model {
  name: string;
  path: string;
  files: string[];
  description?: string;
  available?: boolean;
}

export interface User {
  username: string;
  role: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  success: boolean;
  message: string;
  data?: {
    access_token: string;
    refresh_token: string;
    username: string;
    role: string;
  };
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
}

// 数据清洗相关类型
export interface CleaningConfig {
  enable_deduplication: boolean;
  enable_consistency_filter: boolean;
  enable_cluster_filter: boolean;
  enable_mislabeled_detector: boolean;
  enable_danbooru_enrichment: boolean;
  similarity_threshold: number;
  consistency_threshold: number;
  outlier_threshold: number;
  text_threshold: number;
  confusion_gap: number;
  dry_run: boolean;
  min_images_per_character: number;
  max_workers: number;
}

export interface CleaningResponse {
  success: boolean;
  message: string;
  data?: CleaningResult;
  task_id?: string;
}

export interface CleaningResult {
  duration_seconds?: number;
  total_characters?: number;
  total_original_images?: number;
  total_cleaned_images?: number;
  total_removed_images?: number;
  overall_keep_rate?: number;
  dedup_removed?: number;
  consistency_removed?: number;
  cluster_removed?: number;
  mislabeled_removed?: number;
  character_results?: Record<string, CharacterCleanResult>;
  report_path?: string;
  status?: string;
  error?: string;
}

export interface CharacterCleanResult {
  name: string;
  original_count: number;
  after_dedup: number;
  after_consistency: number;
  after_cluster: number;
  after_mislabeled: number;
  final_count: number;
  removed_count: number;
  removed_files?: string[];
  filtered_files?: string[];
}

export interface CleaningTask {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input_dir: string;
  output_dir: string;
  start_time?: number;
  end_time?: number;
  duration_seconds?: number;
  result?: CleaningResult;
  error?: string;
}

// 数据清理进度相关类型
export interface CleaningTaskProgress {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  total: number;
  completed: number;
  failed: number;
  progress: number;
  start_time?: string;
  end_time?: string;
  message: string;
}

export interface CleaningSummary {
  total_processed: number;
  total_valid: number;
  total_rejected: number;
  total_duplicates: number;
  avg_confidence: number;
  avg_quality_score: number;
}

export interface CleaningProgress {
  last_updated: string;
  total_samples: number;
  tasks: {
    annotation: CleaningTaskProgress;
    deduplication: CleaningTaskProgress;
    quality_filter: CleaningTaskProgress;
    character_matching: CleaningTaskProgress;
    data_export: CleaningTaskProgress;
  };
  summary: CleaningSummary;
}
