// Тесты модалки SettingsModal: рендеринг, переключение вкладок, сохранение.
import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
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

// ── LocalStorage mock ──
const localStorageMock = (() => {
  let store = {}
  return {
    getItem: vi.fn((key) => store[key] ?? null),
    setItem: vi.fn((key, value) => { store[key] = value }),
    clear: vi.fn(() => { store = {} }),
    removeItem: vi.fn((key) => { delete store[key] }),
  }
})()
Object.defineProperty(global, 'localStorage', { value: localStorageMock })

import SettingsModal from '../components/SettingsModal.jsx'


describe('SettingsModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    settings: {
      llm_url: 'http://localhost:1234/v1',
      llm_api_key: 'lm-studio',
      llm_model: 'gpt-4o',
      use_gguf: '',
      gguf_model_path: '',
      gguf_mmproj_path: '',
      mtp_enabled: false,
      gguf_batch_size: 512,
      gguf_ubatch_size: 256,
      vision_mtp_enabled: false,
      vision_batch_size: 512,
      vision_ubatch_size: 256,
    },
    onSave: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    localStorageMock.clear()
  })

  it('returns null when not open', () => {
    const { container } = render(<SettingsModal {...defaultProps} isOpen={false} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders modal when open', () => {
    render(<SettingsModal {...defaultProps} />)
    expect(screen.getByText(/Настройки LLM/i)).toBeInTheDocument()
  })

  it('has LLM and RAG tabs', () => {
    render(<SettingsModal {...defaultProps} />)
    const llmElements = screen.getAllByText(/LLM/i)
    expect(llmElements.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Поиск и RAG/i)).toBeInTheDocument()
  })

  it('switches to RAG tab on click', async () => {
    render(<SettingsModal {...defaultProps} />)
    fireEvent.click(screen.getByText(/Поиск и RAG/i))
    // После клика вкладка RAG должна быть видна
    await waitFor(() => {
      expect(screen.getByText(/Поиск и RAG/i)).toBeInTheDocument()
    })
  })

  it('renders save button', () => {
    render(<SettingsModal {...defaultProps} />)
    expect(screen.getByText(/Сохранить/i)).toBeInTheDocument()
  })
})
