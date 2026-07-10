// Тесты компонента ChatArea: рендеринг, базовые сценарии.
import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

// ── Mock EventSource ──
class MockEventSource {
  constructor(url) {
    this.url = url
    this.onmessage = null
    this.onerror = null
  }
  close() {}
}
global.EventSource = MockEventSource

// ── Mock fetch ──
global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    headers: new Headers(),
    body: {
      getReader: () => ({
        read: () => Promise.resolve({ done: true, value: new Uint8Array() }),
      }),
    },
  })
)

// ── Mock Clipboard API ──
Object.assign(navigator, {
  clipboard: {
    write: vi.fn(() => Promise.resolve()),
    writeText: vi.fn(() => Promise.resolve()),
  },
})

// ── SettingsModal is lazy-loaded, mock it ──
vi.mock('../components/SettingsModal', () => ({
  default: ({ isOpen, onClose }) =>
    isOpen ? <div data-testid="settings-modal"><button onClick={onClose}>X</button></div> : null,
}))

import ChatArea from '../components/ChatArea.jsx'


describe('ChatArea', () => {
  const defaultProps = {
    notebook: { id: 'test-nb', name: 'Test Notebook' },
    selectedSources: ['doc1.pdf', 'doc2.pdf'],
    llmSettings: {
      llm_url: 'http://localhost:1234/v1',
      llm_api_key: 'lm-studio',
      llm_model: 'gpt-4o',
    },
    setLlmSettings: vi.fn(),
    onOpenSource: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing', () => {
    const { container } = render(<ChatArea {...defaultProps} />)
    expect(container.querySelector('[class*="flex"]')).toBeTruthy()
  })

  it('renders with empty sources without crashing', () => {
    const { container } = render(<ChatArea {...defaultProps} selectedSources={[]} />)
    expect(container.querySelector('[class*="flex"]')).toBeTruthy()
  })

  it('renders without crashing with selected sources', () => {
    render(<ChatArea {...defaultProps} selectedSources={['test.pdf']} />)
    // Компонент должен рендериться без ошибок
    expect(vi.mocked(global.fetch)).toBeDefined()
  })
})
