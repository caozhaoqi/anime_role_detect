/**
 * 图片压缩工具
 * 用于在上传前压缩大图片，减少上传时间和带宽消耗
 */

export interface CompressionOptions {
  maxWidth?: number;      // 最大宽度
  maxHeight?: number;     // 最大高度
  quality?: number;       // 压缩质量 (0-1)
  maxSizeMB?: number;     // 最大文件大小 (MB)
  fileType?: string;      // 输出文件类型
}

const DEFAULT_OPTIONS: CompressionOptions = {
  maxWidth: 1920,
  maxHeight: 1920,
  quality: 0.85,
  maxSizeMB: 5,
  fileType: 'image/jpeg',
};

/**
 * 压缩图片文件
 * @param file 原始图片文件
 * @param options 压缩选项
 * @returns 压缩后的图片文件
 */
export async function compressImage(
  file: File,
  options: CompressionOptions = {}
): Promise<File> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // 如果文件已经小于限制，直接返回
  if (file.size <= opts.maxSizeMB! * 1024 * 1024) {
    console.log(`文件大小 ${(file.size / 1024 / 1024).toFixed(2)}MB 在限制内，无需压缩`);
    return file;
  }
  
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const img = new Image();
      
      img.onload = () => {
        try {
          // 计算新的尺寸
          let { width, height } = calculateNewDimensions(
            img.width,
            img.height,
            opts.maxWidth!,
            opts.maxHeight!
          );
          
          // 创建canvas进行压缩
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            reject(new Error('无法创建 canvas context'));
            return;
          }
          
          // 使用更好的图像质量
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';
          
          // 绘制图片
          ctx.drawImage(img, 0, 0, width, height);
          
          // 转换为blob
          canvas.toBlob(
            (blob) => {
              if (!blob) {
                reject(new Error('压缩失败'));
                return;
              }
              
              // 创建新的文件对象
              const compressedFile = new File([blob], file.name, {
                type: opts.fileType!,
                lastModified: Date.now(),
              });
              
              console.log(
                `图片压缩完成: ${(file.size / 1024 / 1024).toFixed(2)}MB -> ${(
                  compressedFile.size / 1024 / 1024
                ).toFixed(2)}MB (${((compressedFile.size / file.size) * 100).toFixed(1)}%)`
              );
              
              resolve(compressedFile);
            },
            opts.fileType,
            opts.quality
          );
        } catch (error) {
          reject(error);
        }
      };
      
      img.onerror = () => {
        reject(new Error('图片加载失败'));
      };
      
      img.src = e.target?.result as string;
    };
    
    reader.onerror = () => {
      reject(new Error('文件读取失败'));
    };
    
    reader.readAsDataURL(file);
  });
}

/**
 * 计算新的图片尺寸，保持宽高比
 */
function calculateNewDimensions(
  originalWidth: number,
  originalHeight: number,
  maxWidth: number,
  maxHeight: number
): { width: number; height: number } {
  let width = originalWidth;
  let height = originalHeight;
  
  // 如果图片尺寸已经在限制内，返回原尺寸
  if (width <= maxWidth && height <= maxHeight) {
    return { width, height };
  }
  
  // 计算缩放比例
  const widthRatio = maxWidth / width;
  const heightRatio = maxHeight / height;
  const ratio = Math.min(widthRatio, heightRatio);
  
  width = Math.floor(width * ratio);
  height = Math.floor(height * ratio);
  
  return { width, height };
}

/**
 * 批量压缩图片（P2-4: 并行压缩，带并发限制避免内存溢出）
 * @param files 图片文件数组
 * @param options 压缩选项
 * @param concurrency 最大并发数，默认 5
 * @returns 压缩后的图片文件数组
 */
export async function compressImages(
  files: File[],
  options?: CompressionOptions,
  concurrency: number = 5
): Promise<File[]> {
  const compressedFiles: File[] = new Array(files.length);

  // 并发控制：分批处理，每批最多 concurrency 个文件
  for (let i = 0; i < files.length; i += concurrency) {
    const batch = files.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map(async (file, batchIdx) => {
        const globalIdx = i + batchIdx;
        try {
          // 只压缩图片文件
          if (!file.type.startsWith('image/')) {
            return { index: globalIdx, file };
          }
          const compressed = await compressImage(file, options);
          return { index: globalIdx, file: compressed };
        } catch (error) {
          console.warn(`压缩图片失败: ${file.name}`, error);
          // 压缩失败时使用原文件
          return { index: globalIdx, file };
        }
      })
    );

    // 按原始顺序放入结果数组
    for (const result of batchResults) {
      compressedFiles[result.index] = result.file;
    }
  }

  return compressedFiles;
}

/**
 * 获取图片尺寸信息
 */
export function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const img = new Image();
      
      img.onload = () => {
        resolve({ width: img.width, height: img.height });
      };
      
      img.onerror = () => {
        reject(new Error('图片加载失败'));
      };
      
      img.src = e.target?.result as string;
    };
    
    reader.onerror = () => {
      reject(new Error('文件读取失败'));
    };
    
    reader.readAsDataURL(file);
  });
}

/**
 * 检查文件是否为图片
 */
export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/');
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
