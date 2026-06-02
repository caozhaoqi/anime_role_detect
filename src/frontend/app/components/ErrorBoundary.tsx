'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error: error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      hasError: true,
      error: error,
      errorInfo: errorInfo,
    });

    // 发送错误日志到服务器
    this.logErrorToServer(error, errorInfo);
  }

  logErrorToServer(error: Error, errorInfo: ErrorInfo) {
    try {
      fetch('/api/log/error', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          error: error.message,
          stack: error.stack,
          componentStack: errorInfo.componentStack,
          timestamp: new Date().toISOString(),
          userAgent: navigator.userAgent,
        }),
      });
    } catch (e) {
      console.error('Failed to log error:', e);
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    
    // 刷新页面以确保状态完全重置
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      const { fallback } = this.props;
      
      if (fallback) {
        return fallback;
      }

      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
          <div className="bg-white rounded-xl shadow-lg p-8 max-w-md w-full text-center">
            <div className="text-6xl mb-4">🚨</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              应用出错了
            </h2>
            <p className="text-gray-600 mb-6">
              很抱歉，应用遇到了一个错误。请尝试刷新页面。
            </p>
            <div className="bg-red-50 rounded-lg p-4 mb-6 text-left">
              <p className="text-sm text-red-600 font-semibold mb-2">错误信息:</p>
              <p className="text-sm text-red-700 break-all">
                {this.state.error?.message}
              </p>
            </div>
            <button
              onClick={this.handleRetry}
              className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg transition-colors duration-200 shadow-md hover:shadow-lg"
            >
              刷新页面
            </button>
            <p className="text-xs text-gray-400 mt-4">
              错误已自动报告给开发团队
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

/**
 * 全局错误处理器 Hook
 */
export function useGlobalErrorHandler() {
  React.useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error('全局错误:', event.error);
      
      // 发送错误日志
      try {
        fetch('/api/log/error', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            error: event.error?.message || 'Unknown error',
            stack: event.error?.stack,
            type: 'window.error',
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
          }),
        });
      } catch (e) {
        console.error('Failed to log error:', e);
      }
    };

    const handleRejection = (event: PromiseRejectionEvent) => {
      console.error('未处理的 Promise rejection:', event.reason);
      
      try {
        fetch('/api/log/error', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            error: event.reason?.message || JSON.stringify(event.reason),
            type: 'unhandled.rejection',
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
          }),
        });
      } catch (e) {
        console.error('Failed to log rejection:', e);
      }
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, []);
}
