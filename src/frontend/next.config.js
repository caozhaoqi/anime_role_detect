/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  output: 'export', // 👈 启用静态 HTML 导出，运行 build 时会自动生成真实的 out/ 目录
  // distDir: 'out',
}

module.exports = nextConfig
