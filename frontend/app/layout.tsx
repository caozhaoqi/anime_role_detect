import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '角色分类系统',
  description: '基于人工智能的游戏角色识别系统',
  keywords: ['角色识别', '游戏角色', '人工智能', '图片分类'],
  authors: [{ name: 'System Developer' }],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className={inter.className}>
        {children}
      </body>
    </html>
  )
}
