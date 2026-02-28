export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: string;
  classification?: {
    role: string;
    similarity: number;
    confidence: "high" | "medium" | "low";
  };
  thoughts?: string[];
  isThinkingFinished?: boolean;
  timestamp: number;
}

export interface Model {
  name: string;
  path: string;
  files: string[];
  description?: string;
  available?: boolean;
}
