/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  output: 'standalone',
  async rewrites() {
    const gatewayUrl = process.env.API_GATEWAY_URL || 'http://localhost:8080'
    return [
      {
        source: '/api/:path*',
        destination: `${gatewayUrl}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
