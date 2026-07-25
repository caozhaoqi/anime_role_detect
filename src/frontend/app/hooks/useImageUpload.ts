import { useState, useCallback, useRef } from 'react';
import { compressImage, compressImages, formatFileSize } from '../utils/imageCompression';
import { Message } from '../types';

interface UseImageUploadOptions {
  onMessageAdd?: (message: Message) => void;
}

export const useImageUpload = (options: UseImageUploadOptions = {}) => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);
  const isMountedRef = useRef(true);

  const createCompressingMessage = useCallback(
    (filename: string, size: number, count?: number): Message => {
      const content = count
        ? `正在压缩 ${count} 张图片 (共 ${formatFileSize(size)})...`
        : `正在压缩图片 ${filename} (${formatFileSize(size)})...`;
      return {
        id: `compress_${Date.now()}`,
        role: 'assistant',
        content,
        timestamp: Date.now(),
      };
    },
    []
  );

  const createCompressedMessage = useCallback(
    (originalSize: number, compressedSize: number, count?: number): Message | null => {
      if (compressedSize >= originalSize) return null;
      const content = count
        ? `批量压缩完成: ${formatFileSize(originalSize)} → ${formatFileSize(compressedSize)} (${((compressedSize / originalSize) * 100).toFixed(1)}%)`
        : `图片已压缩: ${formatFileSize(originalSize)} → ${formatFileSize(compressedSize)}`;
      return {
        id: `compressed_${Date.now()}`,
        role: 'assistant',
        content,
        timestamp: Date.now(),
      };
    },
    []
  );

  const generatePreviews = useCallback((files: File[]): Promise<string[]> => {
    return new Promise((resolve) => {
      const previews: string[] = [];
      if (files.length === 0) {
        resolve(previews);
        return;
      }

      files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          if (isMountedRef.current) {
            previews[index] = reader.result as string;
            if (previews.filter(Boolean).length === files.length) {
              resolve(previews);
            }
          }
        };
        reader.readAsDataURL(file);
      });
    });
  }, []);

  const handleSingleImageSelect = useCallback(
    async (file: File) => {
      const compressingMsg = createCompressingMessage(file.name, file.size);
      options.onMessageAdd?.(compressingMsg);

      try {
        const compressedFile = await compressImage(file, {
          maxWidth: 1920,
          maxHeight: 1920,
          quality: 0.85,
          maxSizeMB: 5,
        });

        const compressedMsg = createCompressedMessage(file.size, compressedFile.size);
        if (compressedMsg) {
          options.onMessageAdd?.(compressedMsg);
        }

        setSelectedImage(compressedFile);
        const preview = await generatePreviews([compressedFile]);
        setImagePreview(preview[0] || null);
      } catch {
        setSelectedImage(file);
        const preview = await generatePreviews([file]);
        setImagePreview(preview[0] || null);
      }
    },
    [options.onMessageAdd, createCompressingMessage, createCompressedMessage, generatePreviews]
  );

  const handleBatchImageSelect = useCallback(
    async (files: File[]) => {
      const totalSize = files.reduce((sum, f) => sum + f.size, 0);
      const compressingMsg = createCompressingMessage('', totalSize, files.length);
      options.onMessageAdd?.(compressingMsg);

      try {
        const compressedFiles = await compressImages(files, {
          maxWidth: 1920,
          maxHeight: 1920,
          quality: 0.85,
          maxSizeMB: 5,
        });

        const compressedTotalSize = compressedFiles.reduce((sum, f) => sum + f.size, 0);
        const compressedMsg = createCompressedMessage(totalSize, compressedTotalSize, files.length);
        if (compressedMsg) {
          options.onMessageAdd?.(compressedMsg);
        }

        setSelectedImages(compressedFiles);
        const previews = await generatePreviews(compressedFiles);
        setImagePreviews(previews);
      } catch {
        setSelectedImages(files);
        const previews = await generatePreviews(files);
        setImagePreviews(previews);
      }
    },
    [options.onMessageAdd, createCompressingMessage, createCompressedMessage, generatePreviews]
  );

  const handleImageSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>, isBatchUpload: boolean) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      if (isBatchUpload) {
        await handleBatchImageSelect(Array.from(files));
      } else {
        await handleSingleImageSelect(files[0]);
      }
    },
    [handleSingleImageSelect, handleBatchImageSelect]
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragCounter(0);
      setIsDragging(false);

      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith('image/')) {
        await handleSingleImageSelect(file);
      }
    },
    [handleSingleImageSelect]
  );

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragCounter((prev) => prev + 1);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragCounter((prev) => prev - 1);
  }, []);

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
  }, []);

  const removeBatchImage = useCallback((index: number) => {
    setSelectedImages((prev) => {
      const newImages = [...prev];
      newImages.splice(index, 1);
      return newImages;
    });
    setImagePreviews((prev) => {
      const newPreviews = [...prev];
      newPreviews.splice(index, 1);
      return newPreviews;
    });
  }, []);

  const clearBatchImages = useCallback(() => {
    setSelectedImages([]);
    setImagePreviews([]);
  }, []);

  const reset = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    setSelectedImages([]);
    setImagePreviews([]);
  }, []);

  return {
    selectedImage,
    imagePreview,
    selectedImages,
    imagePreviews,
    isDragging: dragCounter > 0,
    handleImageSelect,
    handleDrop,
    handleDragEnter,
    handleDragOver,
    handleDragLeave,
    removeImage,
    removeBatchImage,
    clearBatchImages,
    reset,
  };
};