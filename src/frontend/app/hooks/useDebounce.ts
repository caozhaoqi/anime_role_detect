'use client';

import { useCallback, useRef } from 'react';

/**
 * 防抖Hook - 用于防止函数被频繁调用
 * @param fn 要防抖的函数
 * @param delay 延迟时间（毫秒）
 * @returns 防抖后的函数
 */
export function useDebounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number = 300
): T {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  return useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        fn(...args);
      }, delay);
    },
    [fn, delay]
  ) as T;
}

/**
 * 节流Hook - 用于限制函数调用频率
 * @param fn 要节流的函数
 * @param limit 限制时间（毫秒）
 * @returns 节流后的函数
 */
export function useThrottle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number = 1000
): T {
  const inThrottleRef = useRef(false);

  return useCallback(
    (...args: Parameters<T>) => {
      if (!inThrottleRef.current) {
        fn(...args);
        inThrottleRef.current = true;
        setTimeout(() => {
          inThrottleRef.current = false;
        }, limit);
      }
    },
    [fn, limit]
  ) as T;
}

/**
 * 带锁定状态的Hook - 防止重复提交
 * @returns [isLocked, lockFn, unlockFn]
 */
export function useLock() {
  const isLockedRef = useRef(false);

  const lock = useCallback(() => {
    isLockedRef.current = true;
  }, []);

  const unlock = useCallback(() => {
    isLockedRef.current = false;
  }, []);

  const isLocked = useCallback(() => {
    return isLockedRef.current;
  }, []);

  return { isLocked, lock, unlock };
}
