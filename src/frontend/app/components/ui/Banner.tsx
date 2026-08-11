import React from 'react';

type Tone = 'success' | 'error' | 'info';

const toneStyles: Record<Tone, string> = {
  success: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700 text-green-700 dark:text-green-300',
  error: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700 text-red-700 dark:text-red-300',
  info: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-700 text-blue-700 dark:text-blue-300',
};

interface BannerProps {
  tone?: Tone;
  icon?: React.ReactNode;
  children: React.ReactNode;
  onClose?: () => void;
  className?: string;
}

const Banner: React.FC<BannerProps> = ({ tone = 'info', icon, children, onClose, className }) => (
  <div
    role="status"
    className={`mx-4 mt-4 px-4 py-3 border rounded-lg flex items-center space-x-3 ${toneStyles[tone]} ${className ?? ''}`}
  >
    {icon}
    <div className="flex-1 text-sm font-medium">{children}</div>
    {onClose && (
      <button
        type="button"
        onClick={onClose}
        aria-label="关闭"
        className="opacity-70 hover:opacity-100 transition-opacity"
      >
        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </button>
    )}
  </div>
);

export default Banner;
