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
  used_model?: boolean;
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
    used_model?: boolean;
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
    used_model?: boolean;
  }>;
  role_info?: any;
  // 模型覆盖度透明度标注（后端 annotate_coverage 注入，标识预测角色是否在模型已知类别集合内）
  model_coverage?: {
    model_name?: string;
    known_class_count?: number;
    predicted_role?: string | null;
    is_known?: boolean | null;
  };
  model_name?: string;
  summary?: string;
  fallback?: boolean;
  debug?: {
    enabled: boolean;
    degraded_path: boolean;
    yolo_total_boxes: number;
    annotated_image: string;
    boxes: Array<{
      bbox: number[];
      class_id: number;
      raw_confidence: number;
      passed_conf_threshold: boolean;
      cropped_role: boolean;
      is_known_character: boolean;
      kept: boolean;
      discard_reason: string | null;
      candidates: Array<{ role: string; prob: number }>;
    }>;
  };
  // Phase1: Grad-CAM 热力图（懒加载，按角色生成）
  gradcam?: {
    target_label: string;
    confidence: number;
    cam_heatmap_base64: string;
  };
  // Phase1: 纠错反馈状态
  feedbackSubmitted?: boolean;
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

// Phase1: Grad-CAM / 角色列表 / 反馈类型
export interface GradCAMResult {
  target_class: number;
  target_label: string;
  confidence: number;
  cam_heatmap_base64: string;
}

export interface RoleInfo {
  idx: number;
  name: string;
  cn?: string;
  jp?: string;
  anime?: string;
}

export interface FeedbackPayload {
  recognition_id: string;
  endpoint: string;
  original_prediction: string;
  original_confidence: number;
  corrected_label: string;
  image_ref?: string;
  image_data?: string;
  timestamp: string;
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

