import React from 'react';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  label?: string;
}

const sizeMap: Record<NonNullable<SpinnerProps['size']>, string> = {
  sm: 'h-4 w-4',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
};

const Spinner: React.FC<SpinnerProps> = ({ size = 'md', className, label = '加载中' }) => (
  <div
    role="status"
    aria-label={label}
    className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-500 ${sizeMap[size]} ${className ?? ''}`}
  />
);

export default Spinner;
