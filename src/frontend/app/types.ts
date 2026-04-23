export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: string | null;
  classification?: {
    role: string;
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
  }>;
  thoughts?: string[];
  isThinking?: boolean;
  isThinkingFinished?: boolean;
  error?: string;
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
