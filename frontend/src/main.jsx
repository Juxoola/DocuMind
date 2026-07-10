// Точка входа: рендер App в StrictMode.
// ── Импорты ──
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
// ── Точка входа: рендер приложения ──
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
