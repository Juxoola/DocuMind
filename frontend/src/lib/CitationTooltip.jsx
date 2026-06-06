import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText } from 'lucide-react';

/**
 * Единый компонент кнопки-цитаты [N].
 *
 * Используется И в чате (ChatArea), И в модале закладки (LlmMarkdown).
 * Один и тот же визуал, одно и то же поведение hover/click.
 *
 * Props:
 *   n      — номер источника
 *   src    — объект источника {file_name, page, time, text, snippet}
 *   onClick — (src) => void
 *   onHover — (src, {x, y}) => void
 *   onLeave — () => void
 */
export function CitationButton({ n, src, onClick, onHover, onLeave }) {
  const btnRef = useRef(null);

  if (!src) {
    return <span data-copy-skip="1" className="text-muted-foreground opacity-50 whitespace-nowrap">[{n}]</span>;
  }

  const handleMouseEnter = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      onHover && onHover(src, { x: rect.left + rect.width / 2, y: rect.top });
    }
  };

  return (
    <span data-copy-skip="1" className="relative inline-flex items-center align-baseline">
      <button
        ref={btnRef}
        type="button"
        onClick={(e) => { e.preventDefault(); onClick && onClick(src); }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={onLeave}
        title={`${src.file_name}${src.page != null ? ` · стр. ${src.page}` : ''}`}
        className="inline-flex items-center justify-center w-4 h-4 bg-primary/20 text-primary text-[9px] font-bold rounded-full hover:bg-primary hover:text-white transition-all shadow-sm border border-primary/20 relative -top-[1px]"
      >
        {n}
      </button>
    </span>
  );
}

/**
 * Единый портал-тултип источника.
 *
 * Рендерится в document.body, привязан к координатам кнопки снизу.
 * Стилистика идентична чату: тёмная карточка, иконка FileText, мета, сниппет.
 *
 * Props:
 *   hoveredSource — { src, x, y } | null
 *   onClose       — () => void
 *   onCancelClose — () => void  (вызывается при mouseenter на тултип)
 *   onResumeClose — () => void  (вызывается при mouseleave с тултипа)
 */
export function CitationTooltipPortal({ hoveredSource, onClose, onCancelClose, onResumeClose }) {
  // Закрытие по Escape
  useEffect(() => {
    if (!hoveredSource) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose && onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [hoveredSource, onClose]);

  if (!hoveredSource) return null;

  const { src, x, y } = hoveredSource;
  const preview = src.snippet || src.text || '';
  const n = (src._n != null) ? src._n : null; // опционально — если хотим показать [N] в углу

  return createPortal(
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.12 }}
        onMouseEnter={onCancelClose}
        onMouseLeave={onResumeClose}
        className="fixed w-80 p-4 bg-card/98 border border-border/80 shadow-[0_20px_50px_rgba(0,0,0,0.5)] rounded-2xl z-[9999] pointer-events-auto backdrop-blur-2xl text-left"
        style={{
          left: Math.max(20, Math.min(window.innerWidth - 340, x - 160)),
          bottom: window.innerHeight - y + 12,
        }}
      >
        <div className="flex items-center gap-3 mb-3 pb-2 border-b border-border/40">
          <div className="p-1.5 bg-primary/15 rounded-xl text-primary shadow-inner shrink-0">
            <FileText size={14} />
          </div>
          <div className="flex flex-col min-w-0 flex-1">
            <span className="text-[11px] font-bold truncate text-primary uppercase tracking-wider">
              {src.file_name}
            </span>
            <div className="flex gap-2 mt-0.5">
              {src.page != null && (
                <span className="text-[9px] text-muted-foreground font-medium">Стр {src.page}</span>
              )}
              {src.time != null && (
                <span className="text-[9px] text-muted-foreground font-medium">
                  • {Math.floor(src.time / 60)}:{(Math.floor(src.time) % 60).toString().padStart(2, '0')}
                </span>
              )}
            </div>
          </div>
        </div>
        {preview && (
          <div className="max-h-60 overflow-y-auto pr-2 custom-scrollbar">
            <p className="text-[12px] leading-relaxed text-foreground/90 italic font-medium whitespace-pre-wrap">
              «{preview.replace(/^Source \d+:\s*/im, '').slice(0, 300)}{preview.length > 300 ? '…' : ''}»
            </p>
          </div>
        )}
        <div className="mt-3 pt-2 border-t border-border/40 flex justify-between items-center">
          <span className="text-[9px] font-medium text-muted-foreground uppercase tracking-wider">
            Нажмите для перехода
          </span>
          {n != null && (
            <span className="text-[9px] font-black text-primary">[{n}]</span>
          )}
        </div>
        <div
          className="absolute top-full border-[8px] border-transparent border-t-card/98"
          style={{
            left: Math.max(
              15,
              Math.min(305, x - (Math.max(20, Math.min(window.innerWidth - 340, x - 160))) - 8)
            ),
          }}
        />
      </motion.div>
    </AnimatePresence>,
    document.body
  );
}
