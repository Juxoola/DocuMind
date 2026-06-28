// Настройка окружения тестов: моки framer-motion, lucide-react, react-markdown и др.
import '@testing-library/jest-dom/vitest'
import React from 'react'

// ── Mock framer-motion ──
vi.mock('framer-motion', () => ({
  motion: new Proxy({}, { get: () => 'div' }),
  AnimatePresence: ({ children }) => children,
}))

// ── Mock lucide-react icons — use createElement to avoid JSX in .js ──
vi.mock('lucide-react', () => {
  const names = [
    'X', 'Save', 'Globe', 'Key', 'Cpu', 'HardDrive', 'Server',
    'RefreshCw', 'FolderOpen', 'ChevronDown', 'ChevronRight',
    'Zap', 'Filter', 'MessageSquare', 'Database', 'Search',
    'Send', 'Trash2', 'Sparkles', 'Clock', 'FileText', 'Settings',
    'Image', 'Square', 'Plus', 'SlidersHorizontal',
    'Bookmark', 'BookmarkCheck', 'Tag', 'RotateCcw',
    'Eye', 'Pencil', 'Copy', 'Check',
    'ListChecks', 'ListOrdered', 'AlignLeft', 'Scale',
    'GraduationCap', 'Smile',
  ]
  const icons = {}
  names.forEach(name => {
    icons[name] = React.forwardRef(({ size, className, ...props }, ref) =>
      React.createElement('svg', {
        'data-testid': `icon-${name.toLowerCase()}`,
        className,
        ref,
        ...props,
      })
    )
    icons[name].displayName = name
  })
  return icons
})

// ── Mock react-markdown ──
vi.mock('react-markdown', () => ({
  default: ({ children }) => React.createElement('div', { 'data-testid': 'markdown' }, children),
}))

// ── Mock remark/rehype plugins ──
vi.mock('remark-gfm', () => ({ default: () => {} }))
vi.mock('remark-math', () => ({ default: () => {} }))
vi.mock('rehype-katex', () => ({ default: () => {} }))

// ── Mock react-syntax-highlighter ──
vi.mock('react-syntax-highlighter', () => ({
  Prism: ({ children }) => React.createElement('pre', { 'data-testid': 'syntax' }, children),
}))
vi.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({ atomDark: {} }))

// ── Mock @tanstack/react-virtual ──
vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({
        key: i, index: i, start: 0, size: 80, lane: 0,
      })),
    getTotalSize: () => count * 80,
    scrollToIndex: () => {},
    measureElement: () => {},
  }),
}))

// ── Mock axios ──
vi.mock('axios', () => ({
  default: {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    patch: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
  },
  get: vi.fn(() => Promise.resolve({ data: {} })),
  post: vi.fn(() => Promise.resolve({ data: {} })),
}))
