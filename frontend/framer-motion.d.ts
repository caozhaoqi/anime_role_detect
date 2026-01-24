import * as React from 'react';
import { motion as motionOriginal } from 'framer-motion';

// 扩展 Framer Motion 的类型定义
declare module 'framer-motion' {
  export interface HTMLMotionProps<TagName extends keyof React.ReactHTML> {
    className?: string;
    children?: React.ReactNode;
    src?: string;
    alt?: string;
    disabled?: boolean;
    onDragOver?: (e: React.DragEvent<Element>) => void;
    onDragLeave?: () => void;
    onDrop?: (e: React.DragEvent<Element>) => void;
    onClick?: () => void | Promise<void>;
  }
  
  export interface MotionProps {
    className?: string;
    children?: React.ReactNode;
    src?: string;
    alt?: string;
    disabled?: boolean;
    onDragOver?: (e: React.DragEvent<Element>) => void;
    onDragLeave?: () => void;
    onDrop?: (e: React.DragEvent<Element>) => void;
    onClick?: () => void | Promise<void>;
  }
}
