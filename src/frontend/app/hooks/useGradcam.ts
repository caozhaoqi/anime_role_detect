import { useState } from 'react';
import { GradCAMResult } from '../types';

export function useGradcam() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GradCAMResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = async (file: File | Blob, targetClass?: number) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (targetClass !== undefined) {
        formData.append('target_class', String(targetClass));
      }
      const res = await fetch('/api/model/gradcam', {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      if (json.code === 0 && json.data) {
        setResult(json.data);
        return json.data as GradCAMResult;
      } else {
        setError(json.message || json.error || '生成失败');
        return null;
      }
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { loading, result, error, generate, setResult };
}
