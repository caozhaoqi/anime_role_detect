import type { Metadata } from 'next';
import './globals.css';
import ErrorBoundary from './components/ErrorBoundary';

export const metadata: Metadata = {
  title: '动漫角色识别',
  description: '基于深度学习的动漫角色识别系统',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
