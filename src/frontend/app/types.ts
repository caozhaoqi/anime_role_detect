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
