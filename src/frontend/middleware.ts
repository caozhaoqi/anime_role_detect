import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const protectedPaths = ['/monitor', '/admin']
const publicPaths = ['/', '/login']

export function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname
  const token = request.cookies.get('accessToken')?.value || request.headers.get('Authorization')

  const isProtected = protectedPaths.some((p) => path.startsWith(p))
  const isPublic = publicPaths.some((p) => path === p)

  if (isProtected && !token) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  if (!isPublic && !isProtected && !token) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}