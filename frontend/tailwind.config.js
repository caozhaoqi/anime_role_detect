/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#6366f1',
        secondary: '#8b5cf6',
        accent: '#ec4899',
        danger: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6',
        success: '#10b981',
        background: '#0a0a0a',
        foreground: '#ededed',
        border: '#27272a',
        'border-light': '#3f3f46',
        'border-dark': '#18181b',
        'text-primary': '#fafafa',
        'text-secondary': '#a1a1aa',
        'text-light': '#71717a',
        'text-placeholder': '#52525b',
        'card-bg': '#18181b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}