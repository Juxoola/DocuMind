import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Trash2, Sparkles, Clock, Zap, Cpu, FileText, Settings as SettingsIcon, HardDrive, Square, Image as ImageIcon, Plus, X as XIcon, ChevronRight, SlidersHorizontal, RefreshCw, Bookmark, BookmarkCheck, Tag as TagIcon, RotateCcw, Eye, Pencil, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// import rehypeRaw from 'rehype-raw'; // XSS fix: rehype-raw позволял LLM-выводу выполнять произвольный HTML
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '../lib/utils';
import SettingsModal from './SettingsModal';
import axios from 'axios';
import { CitationButton, CitationTooltipPortal } from '../lib/CitationTooltip';
import { extractCleanContent, copyAsRichText } from '../lib/copyToClipboard';

// ── Collapsible Thinking Block ────────────────────────────────────────────────
const ThinkingBlock = ({ content, isStreaming }) => {
  const [open, setOpen] = useState(true);
  const bodyRef = useRef(null);

  // Автоскролл тела пока стримится
  useEffect(() => {
    if (isStreaming && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [content, isStreaming]);

  // Автосворачивание когда стриминг завершён
  useEffect(() => {
    if (!isStreaming && content) {
      setOpen(false);
    }
  }, [isStreaming]);

  if (!content) return null;

  return (
    <div className="mb-3 rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-2 text-left hover:bg-purple-500/10 transition-colors"
      >
        <ChevronRight
          size={12}
          className={cn("text-purple-400 transition-transform duration-200 flex-shrink-0", open && "rotate-90")}
        />
        <span className="text-[10px] font-black uppercase tracking-widest text-purple-400/80 flex-1">
          {isStreaming ? 'Рассуждения модели...' : 'Рассуждения'}
        </span>
        {isStreaming && (
          <span className="flex gap-0.5">
            {[0, 0.15, 0.3].map((d, i) => (
              <span
                key={i}
                className="w-1 h-1 rounded-full bg-purple-400"
                style={{ animation: `pulse 1s ease-in-out ${d}s infinite` }}
              />
            ))}
          </span>
        )}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div
              ref={bodyRef}
              className="px-4 pb-4 pt-2 text-[11px] text-muted-foreground/90 italic border-t border-purple-500/10 whitespace-pre-wrap max-h-96 overflow-y-auto leading-relaxed custom-scrollbar bg-purple-500/5"
            >
              {content}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Citation = ({ n, sources, onClick, onHover, onLeave }) => {
  // Делегируем в общий компонент. Старая локальная реализация удалена,
  // API идентично — props один-в-один.
  return (
    <CitationButton
      n={n}
      src={sources?.[n - 1]}
      onClick={onClick}
      onHover={onHover}
      onLeave={onLeave}
    />
  );
};

// ── Sleek Custom Slider Component ───────────────────────────────────────────
const SleekSlider = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  icon: Icon,
  presets = [],
  suffix = "",
  description,
  accentColor = "var(--color-primary)",
  disabled = false,
  showUnlimited = false,
  unlimitedChecked = false,
  onUnlimitedChange = null
}) => {
  const [inputValue, setInputValue] = useState(value);

  // Sync internal state when prop changes
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleSliderChange = (e) => {
    if (disabled) return;
    const val = parseInt(e.target.value);
    setInputValue(val);
    onChange(val);
  };

  const handleInputChange = (e) => {
    if (disabled) return;
    let val = e.target.value === '' ? '' : parseInt(e.target.value);
    setInputValue(val);
  };

  const handleInputBlur = () => {
    if (disabled) return;
    let val = parseInt(inputValue);
    if (isNaN(val)) val = min;
    if (val < min) val = min;
    if (val > max) val = max;
    setInputValue(val);
    onChange(val);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleInputBlur();
      e.target.blur();
    }
  };

  // Calculate percentage for gradient track filling
  const pct = disabled ? 0 : ((value - min) / (max - min)) * 100;

  return (
    <div className={cn(
      "flex flex-col gap-2.5 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all duration-300",
      disabled && "opacity-60"
    )}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 flex-wrap sm:flex-nowrap min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            {Icon && <Icon className="text-muted-foreground shrink-0" size={14} />}
            <span className="text-xs font-bold text-foreground/90 leading-tight">{label}</span>
          </div>
          {showUnlimited && (
            <label className={cn(
              "flex items-center gap-1.5 cursor-pointer select-none text-[9px] font-black uppercase px-2 py-0.5 rounded-md border transition-all pointer-events-auto whitespace-nowrap shrink-0",
              unlimitedChecked
                ? "bg-purple-500/10 border-purple-500/30 text-purple-400"
                : "bg-muted/40 border-border/40 text-muted-foreground hover:border-border/60 hover:text-foreground"
            )}>
              <input
                type="checkbox"
                checked={unlimitedChecked}
                onChange={(e) => onUnlimitedChange(e.target.checked)}
                className="hidden"
              />
              <span className="whitespace-nowrap">Без лимита</span>
            </label>
          )}
        </div>
        
        {/* Sleek inline number input */}
        <div className="flex items-center gap-1.5 bg-muted/60 border border-border/40 rounded-lg px-2 py-0.5">
          <input
            type="number"
            value={disabled ? "" : inputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            onKeyDown={handleKeyDown}
            min={min}
            max={max}
            disabled={disabled}
            className="w-11 bg-transparent border-none text-[11px] font-black focus:ring-0 p-0 text-right appearance-none tuning-number-input"
            style={{ color: accentColor }}
          />
          <span className="text-[9px] font-bold text-muted-foreground uppercase">{disabled ? "∞" : suffix}</span>
        </div>
      </div>

      {description && (
        <p className="text-[9.5px] text-muted-foreground/80 leading-relaxed font-medium">
          {description}
        </p>
      )}

      {/* Slide range with custom visual background fill */}
      <div className="relative mt-2.5 flex items-center">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={disabled ? max : value}
          onChange={handleSliderChange}
          disabled={disabled}
          className="tuning-slider"
          style={{
            background: `linear-gradient(to right, ${accentColor} 0%, ${accentColor} ${pct}%, var(--color-border) ${pct}%, var(--color-border) 100%)`
          }}
        />
      </div>

      {/* Preset markers */}
      {presets.length > 0 && (
        <div className="flex items-center justify-between mt-2.5 flex-wrap gap-1">
          {presets.map((p) => {
            const isSelected = !disabled && value === p.value;
            return (
              <button
                key={p.value}
                onClick={() => {
                  if (disabled) return;
                  setInputValue(p.value);
                  onChange(p.value);
                }}
                disabled={disabled}
                className={cn(
                  "px-2 py-0.5 text-[9px] font-extrabold rounded-md transition-all border",
                  isSelected
                    ? "bg-primary/10 border-primary/20 shadow-sm"
                    : "bg-transparent text-muted-foreground border-transparent hover:border-border/60 hover:text-foreground"
                )}
                style={isSelected ? { color: accentColor, backgroundColor: `${accentColor}12`, borderColor: `${accentColor}25` } : {}}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Memoized Message Item Component for High Performance ──────────────────────
const MessageItem = React.memo(({
  msg,
  index,
  messagesLength,
  onOpenSource,
  setHoveredSource,
  setTooltipCoords,
  tooltipTimeoutRef,
  notebook,
  question, // текст вопроса пользователя над этим AI-сообщением (для закладки)
  onBookmarkSaved, // (index, ts) => void
}) => {
  const [bmPopover, setBmPopover] = React.useState(false);
  const [bmTitle, setBmTitle] = React.useState('');
  const [bmTags, setBmTags] = React.useState('');
  const [bmSaving, setBmSaving] = React.useState(false);
  const [bmError, setBmError] = React.useState('');
  const bmBtnRef = React.useRef(null);
  const bmPopoverRef = React.useRef(null);
  const [bmCoords, setBmCoords] = React.useState({ top: 0, left: 0 });
  const [copied, setCopied] = React.useState(false);
  const copyTimeoutRef = React.useRef(null);
  const bubbleRef = React.useRef(null);

  const handleCopy = async () => {
    const extracted = extractCleanContent(bubbleRef.current);
    if (!extracted || (!extracted.html && !extracted.text)) return;
    const ok = await copyAsRichText(extracted);
    if (ok) {
      setCopied(true);
      clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
    }
  };

  React.useEffect(() => () => clearTimeout(copyTimeoutRef.current), []);

  const openBmPopover = () => {
    setBmTitle(''); // По умолчанию пусто — на карточке возьмётся question
    setBmTags('');
    setBmError('');
    // Позиционируем относительно кнопки через getBoundingClientRect
    if (bmBtnRef.current) {
      const rect = bmBtnRef.current.getBoundingClientRect();
      // Поповер шириной 320px — выровняем по левому краю кнопки,
      // но если упрётся в правый край viewport — сдвинем влево
      const popWidth = 320;
      const margin = 8;
      let left = rect.left;
      if (left + popWidth + margin > window.innerWidth) {
        left = window.innerWidth - popWidth - margin;
      }
      if (left < margin) left = margin;
      const top = rect.bottom + 6;
      setBmCoords({ top, left });
    }
    setBmPopover(true);
  };

  // Закрытие по клику вне и по Escape
  React.useEffect(() => {
    if (!bmPopover) return;
    const onDown = (e) => {
      if (bmPopoverRef.current && !bmPopoverRef.current.contains(e.target)) {
        // Игнорируем клик по самой кнопке (она сама переключает)
        if (bmBtnRef.current && bmBtnRef.current.contains(e.target)) return;
        setBmPopover(false);
      }
    };
    const onKey = (e) => { if (e.key === 'Escape') setBmPopover(false); };
    // mousedown чтобы успеть до onClick
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [bmPopover]);

  const saveBookmark = async () => {
    if (!notebook || !question || !msg.content) return;
    setBmSaving(true);
    setBmError('');
    try {
      // Снэпшот источников (только метаданные, без полного текста — текст устаревает)
      const sourcesSnap = (msg.sources || []).map(s => ({
        file_name: s.file_name,
        page: s.page ?? null,
        time: s.time ?? null,
        snippet: (s.text || '').slice(0, 200),
      }));
      const payload = {
        notebook_id: notebook.id,
        question: question,
        answer: msg.content,
        sources: sourcesSnap,
        model: msg._meta?.model || '',
        answer_mode: msg._meta?.answer_mode || 'concise',
        thinking_mode: !!msg._meta?.thinking_mode,
        title: bmTitle.trim(),
        tags: bmTags.split(',').map(s => s.trim()).filter(Boolean),
      };
      await axios.post('/api/bookmarks', payload);
      // Сообщаем ChatArea, чтобы пометить сообщение как сохранённое
      if (onBookmarkSaved) onBookmarkSaved(msg.id || index, Date.now());
      window.dispatchEvent(new CustomEvent('bookmark:added'));
      setBmPopover(false);
    } catch (e) {
      setBmError(e?.response?.data?.detail || 'Не удалось сохранить');
    } finally {
      setBmSaving(false);
    }
  };
  const preProcessMessage = (text) => {
    if (!text) return "";
    
    // Сначала обрабатываем формулы LaTeX
    let processed = text
      .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$')
      .replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');

    // Разделяем текст на части: обычный текст и блоки кода (```...```)
    const parts = processed.split(/(```[\s\S]*?```)/g);
    
    const finalParts = parts.map(part => {
      // Если это блок кода, возвращаем его как есть (не трогаем цитаты внутри)
      if (part.startsWith('```')) return part;
      
      // В обычном тексте защищаем LaTeX от ссылок
      const mathBlocks = [];
      let subPart = part.replace(/(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)/g, (m) => {
        mathBlocks.push(m);
        return `%%MATH_${mathBlocks.length - 1}%%`;
      });

      // Заменяем [N] на ссылки только в обычном тексте
      subPart = subPart.replace(/(?<!\\in|\\subset|\\subseteq|\\supset)\[(\d+(?:,\s*\d+)*)\]/g, (match, nums) => {
        return nums.split(',').map(n => {
          const num = n.trim();
          return `[${num}](#cite:${num})`;
        }).join('');
      });

      // Возвращаем LaTeX
      return subPart.replace(/%%MATH_(\d+)%%/g, (_, idx) => mathBlocks[parseInt(idx)]);
    });

    processed = finalParts.join('');

    // Обработка всех форматов thinking-тегов:
    const thinkFormats = [
      { open: '<|channel|>', close: '<channel|>' },   // Gemma 4 (рус)
      { open: '<think>', close: '</think>' },     // Qwen/DeepSeek (рус)
      { open: '<|think|>', close: '<|/think|>' },  // запасной
    ];
    for (const { open, close } of thinkFormats) {
      if (!processed.includes(open)) continue;
      const parts = processed.split(open);
      const beforeThink = parts[0];
      const thinkContent = parts[1];
      if (thinkContent && thinkContent.includes(close)) {
        const thinkParts = thinkContent.split(close);
        processed = `${beforeThink}<details class="mb-4 rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden"><summary class="cursor-pointer px-4 py-2 text-[11px] font-bold text-purple-500 hover:bg-purple-500/10 transition-colors select-none">✨ Рассуждения</summary><div class="p-4 text-xs text-muted-foreground/80 italic border-t border-purple-500/10 whitespace-pre-wrap">${thinkParts[0]}</div></details>\n${thinkParts.slice(1).join(close)}`;
      } else if (thinkContent) {
        processed = `${beforeThink}<div class="mb-4 rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden"><div class="px-4 py-2 text-[11px] font-bold text-purple-500 flex items-center gap-2"><span class="animate-pulse">✨ Модель рассуждает...</span></div><div class="p-4 text-xs text-muted-foreground/80 italic border-t border-purple-500/10 whitespace-pre-wrap">${thinkContent}</div></div>`;
      }
      break; // обработали — выходим
    }
    return processed;
  };

  const renderContent = () => {
    if (msg.loading) return (
      <div className="flex gap-1 py-2">
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.2s]" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.4s]" />
      </div>
    );

    return (
      <div className="flex flex-col gap-2">
        {msg.image && (
          <div className="mb-2 rounded-lg overflow-hidden border border-border/40 max-w-sm">
            <img src={msg.image} alt="User upload" className="w-full h-auto object-cover" />
          </div>
        )}

        {/* Блок рассуждений — рендерится отдельно, до ответа */}
        {msg.thinkingContent && (
          <ThinkingBlock
            content={msg.thinkingContent}
            isStreaming={!msg.thinkingDone}
          />
        )}
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith('#cite:')) {
                  const num = href.split(':')[1];
                  return (
                    <span data-copy-skip="1" className="inline-block ml-1 no-underline">
                      <Citation
                        n={parseInt(num)}
                        sources={msg.sources}
                        onClick={(src) => {
                          setHoveredSource(null);
                          onOpenSource(src);
                        }}
                        onHover={(src, coords) => {
                          clearTimeout(tooltipTimeoutRef.current);
                          setHoveredSource(src);
                          setTooltipCoords(coords);
                        }}
                        onLeave={() => {
                          tooltipTimeoutRef.current = setTimeout(() => setHoveredSource(null), 100);
                        }}
                      />
                    </span>
                  );
                }
                return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
              },
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const lang = match ? match[1] : '';
                return !inline && match ? (
                  <div className="relative group my-4 rounded-xl overflow-hidden bg-[#0d1117] border border-white/5">
                    {/* Тонкий индикатор языка и кнопка COPY в одной строке, встроенные в блок */}
                    <div data-copy-skip="1" className="flex items-center justify-between px-4 py-2 opacity-40 group-hover:opacity-100 transition-opacity">
                      <span className="text-[9px] font-bold text-white/50 uppercase tracking-[0.2em]">{lang}</span>
                      <button
                        onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                        className="flex items-center gap-1 text-[9px] font-bold text-white/50 hover:text-white transition-colors"
                      >
                        <Zap size={10} /> COPY
                      </button>
                    </div>
                    <div className="overflow-x-auto custom-scrollbar">
                      <SyntaxHighlighter
                        style={atomDark}
                        language={lang}
                        PreTag="div"
                        className="!bg-transparent !pt-0 !p-4 font-mono text-sm leading-relaxed"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    </div>
                  </div>
                ) : (
                  <code className={cn("bg-white/10 px-1.5 py-0.5 rounded text-indigo-300 font-mono text-xs", className)} {...props}>
                    {children}
                  </code>
                );
              }
            }}
          >
            {preProcessMessage(msg.content)}
          </ReactMarkdown>

          {msg.sources && msg.sources.length > 0 && (
            <div data-copy-skip="1" className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-border/20">
              {msg.sources.map((src, idx) => (
                <button
                  key={idx}
                  title={`${src.file_name} ${src.page ? '(стр. ' + src.page + ')' : ''}`}
                  onClick={() => {
                    setHoveredSource(null);
                    onOpenSource(src);
                  }}
                  onMouseEnter={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    clearTimeout(tooltipTimeoutRef.current);
                    setHoveredSource(src);
                    setTooltipCoords({ x: rect.left + rect.width / 2, y: rect.top });
                  }}
                  onMouseLeave={() => {
                    tooltipTimeoutRef.current = setTimeout(() => setHoveredSource(null), 100);
                  }}
                  className="flex items-center gap-2 bg-muted/30 hover:bg-primary/10 border border-border/40 hover:border-primary/30 px-2.5 py-1 rounded-full text-[9px] transition-all"
                >
                  <span className="w-3.5 h-3.5 flex items-center justify-center bg-primary/20 text-primary rounded-full text-[8px] font-bold">
                    {idx + 1}
                  </span>
                  <span className="truncate max-w-[140px] opacity-70 hover:opacity-100">{src.file_name}</span>
                </button>
              ))}
            </div>
          )}

          {/* Кнопка «Сохранить в закладки» (только для AI и только когда стрим завершён) */}
          {msg.role === 'ai' && !msg.loading && msg.content && notebook && (
            <div data-copy-skip="1" className="mt-3 pt-2 border-t border-border/20 flex items-center gap-2">
              {msg._savedAt ? (
                <span className="flex items-center gap-1.5 text-[10px] font-bold text-amber-400/80">
                  <BookmarkCheck size={12} />
                  Сохранено в закладки
                </span>
              ) : (
                <button
                  ref={bmBtnRef}
                  onClick={openBmPopover}
                  className="flex items-center gap-1.5 text-[10px] font-bold text-muted-foreground hover:text-amber-400 transition-colors"
                >
                  <Bookmark size={12} />
                  Сохранить в закладки
                </button>
              )}
              {msg._savedAt && (
                <span className="text-[9px] text-muted-foreground/50">
                  {new Date(msg._savedAt).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}
                </span>
              )}
              <button
                onClick={handleCopy}
                title="Скопировать ответ"
                className={cn(
                  "ml-auto flex items-center gap-1.5 text-[10px] font-bold transition-colors",
                  copied ? "text-emerald-400" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {copied ? <Check size={12} /> : <Copy size={12} />}
                {copied ? "Скопировано" : "Копировать"}
              </button>
            </div>
          )}

          {/* Кнопка «Копировать» для сообщений пользователя — снаружи prose,
              чтобы не ломать типографику текстового блока */}
          {msg.role === 'user' && !msg.loading && msg.content && (
            <div data-copy-skip="1" className="pt-2 flex items-center gap-2">
              <button
                onClick={handleCopy}
                title="Скопировать сообщение"
                className={cn(
                  "ml-auto flex items-center gap-1.5 text-[10px] font-bold transition-colors",
                  copied ? "text-emerald-400" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {copied ? <Check size={12} /> : <Copy size={12} />}
                {copied ? "Скопировано" : "Копировать"}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex w-full",
        msg.role === 'user' ? "justify-end" : "justify-start"
      )}
    >
      <div ref={bubbleRef} className={cn(
        "max-w-[85%] group relative min-w-0",
        msg.role === 'user' ? "chat-bubble-user" : "chat-bubble-ai"
      )} style={{ zIndex: messagesLength - index }}>
        {renderContent()}
      </div>

      {/* Поповер «Сохранить в закладки» — рендерим в портал, чтобы overflow-hidden
          родительского chat-bubble-ai не обрезал форму. */}
      {bmPopover && msg.role === 'ai' && createPortal(
        <AnimatePresence>
          <motion.div
            ref={bmPopoverRef}
            initial={{ opacity: 0, y: -4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.96 }}
            transition={{ duration: 0.12 }}
            style={{ position: 'fixed', top: bmCoords.top, left: bmCoords.left, zIndex: 9999 }}
            className="w-80 p-3 bg-card border border-border shadow-2xl rounded-2xl"
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground flex items-center gap-1.5">
                <Bookmark size={11} className="text-amber-400" />
                Новая закладка
              </span>
              <button
                onClick={() => setBmPopover(false)}
                className="p-1 hover:bg-muted rounded-md text-muted-foreground"
              >
                <XIcon size={12} />
              </button>
            </div>
            <input
              autoFocus
              value={bmTitle}
              onChange={(e) => setBmTitle(e.target.value)}
              placeholder="Заголовок (опц.)"
              className="w-full bg-muted/30 border border-border/50 rounded-lg px-2.5 py-1.5 text-xs mb-2 focus:outline-none focus:ring-1 focus:ring-amber-500/50"
            />
            <input
              value={bmTags}
              onChange={(e) => setBmTags(e.target.value)}
              placeholder="Теги через запятую"
              className="w-full bg-muted/30 border border-border/50 rounded-lg px-2.5 py-1.5 text-xs mb-2 focus:outline-none focus:ring-1 focus:ring-amber-500/50"
            />
            {bmError && (
              <p className="text-[10px] text-destructive mb-2">{bmError}</p>
            )}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setBmPopover(false)}
                className="flex-1 px-2.5 py-1.5 hover:bg-muted text-muted-foreground rounded-lg text-[10px] font-bold transition-all"
              >
                Отмена
              </button>
              <button
                onClick={saveBookmark}
                disabled={bmSaving}
                className="flex-1 px-2.5 py-1.5 bg-amber-500/15 hover:bg-amber-500/25 text-amber-400 disabled:opacity-50 rounded-lg text-[10px] font-black transition-all"
              >
                {bmSaving ? 'Сохранение…' : 'Сохранить'}
              </button>
            </div>
          </motion.div>
        </AnimatePresence>,
        document.body
      )}
    </motion.div>
  );
}, (prev, next) => {
  return (
    prev.msg.content === next.msg.content &&
    prev.msg.loading === next.msg.loading &&
    prev.msg.image === next.msg.image &&
    prev.msg.thinkingContent === next.msg.thinkingContent &&
    prev.msg.thinkingDone === next.msg.thinkingDone &&
    prev.messagesLength === next.messagesLength &&
    prev.index === next.index &&
    prev.msg.sources === next.msg.sources &&
    prev.msg._savedAt === next.msg._savedAt &&
    prev.notebook === next.notebook &&
    prev.question === next.question
  );
});

export default function ChatArea({ notebook, selectedSources, onOpenSource, llmSettings, setLlmSettings }) {
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Привет! Я проанализировал ваши источники и готов ответить на любые вопросы. Что вас интересует?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [maxTokens, setMaxTokens] = useState(() => parseInt(localStorage.getItem('chat_max_tokens')) || 1024);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isTuningOpen, setIsTuningOpen] = useState(false);
  const [contextStrategy, setContextStrategy] = useState(() => localStorage.getItem('chat_context_strategy') || 'sliding');
  const [thinkingMode, setThinkingMode] = useState(() => localStorage.getItem('chat_thinking_mode') === 'true');
  const [thinkingBudget, setThinkingBudget] = useState(() => parseInt(localStorage.getItem('chat_thinking_budget')) || 1024); // -1 = no limit
  // Режим ответа: "concise" (сначала кратко, потом объяснение) / "detailed" (сразу развёрнуто)
  const [answerMode, setAnswerMode] = useState(() => {
    const stored = localStorage.getItem('chat_answer_mode');
    return stored === 'detailed' ? 'detailed' : 'concise';
  });
  const [hoveredSource, setHoveredSource] = useState(null);
  const [tooltipCoords, setTooltipCoords] = useState({ x: 0, y: 0 });
  const [abortController, setAbortController] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const tooltipTimeoutRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Коллбэк: MessageItem сообщает, что Q&A сохранён — помечаем сообщение
  const markMessageSaved = React.useCallback((msgIndex, ts) => {
    setMessages(prev => prev.map((m, i) => i === msgIndex ? { ...m, _savedAt: ts } : m));
  }, []);

  // Слушаем событие «заполни input» от закладки (Sidebar.askAgain)
  React.useEffect(() => {
    const onFill = (e) => {
      const text = e.detail?.text || '';
      if (text) setInput(text);
    };
    window.addEventListener('chat:fill-input', onFill);
    return () => window.removeEventListener('chat:fill-input', onFill);
  }, []);

  // Сохранение настроек при изменении
  useEffect(() => {
    localStorage.setItem('chat_max_tokens', maxTokens.toString());
    localStorage.setItem('chat_thinking_mode', thinkingMode.toString());
    localStorage.setItem('chat_thinking_budget', thinkingBudget.toString());
    localStorage.setItem('chat_context_strategy', contextStrategy);
    localStorage.setItem('chat_answer_mode', answerMode);
  }, [maxTokens, thinkingMode, thinkingBudget, contextStrategy, answerMode]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Подписка на статус LLM (для блокировки Send пока грузится модель)
  const [llmStatus, setLlmStatus] = useState({ state: 'ready', phase: null, model: null, error: null });
  useEffect(() => {
    let es;
    let reconnectTimer;
    const connect = () => {
      es = new EventSource('/api/llm-status/stream');
      es.onmessage = (e) => {
        try { setLlmStatus(JSON.parse(e.data)); } catch {}
      };
      es.onerror = () => {
        if (es) { es.close(); es = null; }
        reconnectTimer = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      if (es) es.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        handleFile(file);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleSend = async () => {
    if (!input.trim() && !selectedImage || isLoading) return;
    if (llmStatus.state === 'loading') return;  // блокируем Send пока грузится модель
    if (selectedSources.length === 0) {
      alert('Выберите хотя бы один источник в боковой панели!');
      return;
    }

    const userMsg = { role: 'user', content: input, image: imagePreview };
    setMessages(prev => [...prev, userMsg]);

    const currentInput = input;
    const currentImage = imagePreview?.split(',')[1];

    setInput('');
    removeImage();
    setIsLoading(true);
    setStats(null);

    const aiMsgIndex = messages.length + 1;
    setMessages(prev => [...prev, {
      role: 'ai',
      content: '',
      thinkingContent: '',
      thinkingDone: false,
      loading: true,
      // Снимок настроек на момент вопроса — для закладки
      _meta: {
        model: llmSettings?.gguf_model_path || llmSettings?.llm_model || '',
        answer_mode: answerMode,
        thinking_mode: thinkingMode,
      },
    }]);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          query: currentInput,
          allowed_files: selectedSources,
          max_tokens: maxTokens,
          notebook_id: notebook.id,
          thinking_mode: thinkingMode,
          thinking_budget: thinkingBudget,
          context_strategy: contextStrategy,
          answer_mode: answerMode,
          image_base64: currentImage,
          ...llmSettings
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';
      let sources = [];
      let buffer = ''; // Буфер для склейки разорванных строк

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Оставляем последний (возможно неполный) кусок в буфере
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;

          const payload = trimmedLine.slice(6);
          if (payload === '[DONE]') break;

          try {
            const data = JSON.parse(payload);
            if (data.type === 'sources') {
              sources = data.sources;
            } else if (data.type === 'thinking_start') {
              // Начало рассуждений — убираем loading-спиннер, инициируем thinking
              setMessages(prev => {
                const next = [...prev];
                if (next[aiMsgIndex]) next[aiMsgIndex] = { ...next[aiMsgIndex], loading: false, thinkingContent: '', thinkingDone: false };
                return next;
              });
            } else if (data.type === 'thinking_chunk') {
              // Стримим рассуждения
              setMessages(prev => {
                const next = [...prev];
                if (next[aiMsgIndex]) next[aiMsgIndex] = {
                  ...next[aiMsgIndex],
                  thinkingContent: (next[aiMsgIndex].thinkingContent || '') + data.text
                };
                return next;
              });
            } else if (data.type === 'thinking_done') {
              // Рассуждения завершены
              setMessages(prev => {
                const next = [...prev];
                if (next[aiMsgIndex]) next[aiMsgIndex] = { ...next[aiMsgIndex], thinkingDone: true };
                return next;
              });
            } else if (data.type === 'chunk') {
              fullContent += data.text;
              updateAiMessage(aiMsgIndex, fullContent, sources);
            } else if (data.type === 'stats') {
              setStats(data);
            } else if (data.type === 'error') {
              fullContent = '⚠️ Ошибка: ' + data.text;
              updateAiMessage(aiMsgIndex, fullContent, []);
            }
          } catch (e) {
            console.error('Ошибка парсинга SSE:', e, payload);
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        updateAiMessage(aiMsgIndex, '*(Генерация остановлена)*', []);
      } else {
        updateAiMessage(aiMsgIndex, '⚠️ Ошибка связи с сервером.', []);
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  };

  const updateAiMessage = (index, content, sources) => {
    setMessages(prev => {
      if (index >= prev.length + 2) return prev;
      const newMessages = [...prev];
      if (!newMessages[index]) return prev;
      newMessages[index] = { ...newMessages[index], content, sources, loading: false };
      return newMessages;
    });
  };

  const clearChat = () => {
    if (abortController) abortController.abort();
    setMessages([{ role: 'ai', content: 'Чат очищен. Какой новый вопрос?' }]);
    setStats(null);
    setIsLoading(false);
  };



  return (
    <div
      className="flex flex-col h-full w-full max-w-[1400px] mx-auto relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-primary/10 backdrop-blur-[2px] border-2 border-dashed border-primary m-4 rounded-3xl pointer-events-none"
          >
            <div className="flex flex-col items-center gap-3 text-primary">
              <Plus size={48} className="animate-pulse" />
              <p className="font-bold text-lg">Отпустите, чтобы прикрепить фото</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Шапка */}
      <div className="flex items-center justify-between gap-3 p-4 border-b bg-background/80 backdrop-blur-md sticky top-0 z-10 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="text-muted-foreground shrink-0" size={18} />
          <h2 className="font-medium text-sm whitespace-nowrap">Ассистент по документам</h2>
          {llmSettings.use_gguf === 'true' && (
            <span className="flex items-center gap-1 px-2 py-0.5 bg-green-500/10 text-green-500 rounded-full text-[9px] font-bold border border-green-500/20 whitespace-nowrap">
              <HardDrive size={10} /> GGUF
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <div className="flex items-center gap-3 bg-muted/30 p-1.5 rounded-xl border border-white/5 px-2">
            <button
              onClick={() => setIsTuningOpen(!isTuningOpen)}
              title="Открыть параметры генерации"
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all text-[10px] font-black uppercase tracking-tight",
                isTuningOpen
                  ? "bg-primary/20 text-primary shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <SlidersHorizontal size={11} />
              <span>Параметры</span>
            </button>

            <div className="w-px h-4 bg-white/10 self-center" />

            <button
              onClick={() => setIsTuningOpen(!isTuningOpen)}
              className="px-2.5 py-1 text-[10px] font-black text-muted-foreground/80 hover:text-foreground transition-all flex items-center gap-1.5"
            >
              <span>Ответ:</span>
              <span className="text-primary font-black">{maxTokens} t</span>
            </button>

            {thinkingMode && (
              <>
                <div className="w-px h-4 bg-white/10 self-center" />
                <button
                  onClick={() => setIsTuningOpen(!isTuningOpen)}
                  className="px-2.5 py-1 text-[10px] font-black text-muted-foreground/80 hover:text-foreground transition-all flex items-center gap-1.5"
                >
                  <span>Бюджет:</span>
                  <span className="text-purple-400 font-black">{thinkingBudget === -1 ? "∞" : `${thinkingBudget} t`}</span>
                </button>
              </>
            )}
          </div>

          <button
            onClick={() => setThinkingMode(!thinkingMode)}
            className={cn(
              "px-3 py-1.5 text-[10px] font-black uppercase tracking-wider rounded-xl transition-all flex items-center gap-1.5 border",
              thinkingMode
                ? "bg-purple-500/10 text-purple-400 border-purple-500/20 shadow-sm"
                : "bg-muted/40 text-muted-foreground hover:text-foreground border-transparent hover:border-border/60"
            )}
          >
            <Sparkles size={11} />
            {thinkingMode ? "Думает" : "Без рассуждений"}
          </button>

          <button
            onClick={() => setAnswerMode(answerMode === 'concise' ? 'detailed' : 'concise')}
            title={answerMode === 'concise'
              ? 'Сейчас: сначала краткий ответ, потом объяснение. Кликните, чтобы переключить на развёрнутый режим.'
              : 'Сейчас: сразу развёрнутый ответ. Кликните, чтобы вернуть режим «сначала кратко».'}
            className={cn(
              "px-3 py-1.5 text-[10px] font-black uppercase tracking-wider rounded-xl transition-all flex items-center gap-1.5 border",
              answerMode === 'detailed'
                ? "bg-amber-500/10 text-amber-400 border-amber-500/20 shadow-sm"
                : "bg-muted/40 text-muted-foreground hover:text-foreground border-transparent hover:border-border/60"
            )}
          >
            {answerMode === 'detailed' ? <FileText size={11} /> : <Zap size={11} />}
            {answerMode === 'detailed' ? 'Развёрнуто' : 'Кратко → пояснение'}
          </button>
          
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors"
          >
            <SettingsIcon size={16} />
          </button>
          <button onClick={clearChat} className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors">
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <AnimatePresence>
        {isTuningOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 220 }}
            className="border-b border-border bg-card/25 backdrop-blur-md overflow-hidden z-10"
          >
            <div className="p-5 max-w-[1400px] mx-auto flex flex-col gap-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* 1. Response length slider */}
                <SleekSlider
                  label="Максимальная длина ответа"
                  value={maxTokens}
                  onChange={setMaxTokens}
                  min={100}
                  max={32000}
                  step={100}
                  icon={Cpu}
                  suffix="t"
                  accentColor="var(--color-primary)"
                  description="Регулирует максимальный лимит токенов, генерируемых моделью за один ответ. Большие значения подходят для развернутых эссе."
                  presets={[
                    { label: "512", value: 512 },
                    { label: "1k", value: 1024 },
                    { label: "2k", value: 2048 },
                    { label: "4k", value: 4096 },
                    { label: "8k", value: 8192 },
                    { label: "16k", value: 16384 },
                    { label: "32k", value: 32000 }
                  ]}
                />

                {/* 2. Thinking budget slider */}
                <SleekSlider
                  label="Бюджет рассуждений"
                  value={thinkingBudget === -1 ? 32000 : thinkingBudget}
                  onChange={(val) => {
                    if (thinkingBudget !== -1) {
                      setThinkingBudget(val);
                    }
                  }}
                  min={100}
                  max={32000}
                  step={100}
                  icon={Sparkles}
                  suffix="t"
                  accentColor="rgb(168, 85, 247)"
                  disabled={!thinkingMode || thinkingBudget === -1}
                  showUnlimited={thinkingMode}
                  unlimitedChecked={thinkingBudget === -1}
                  onUnlimitedChange={(checked) => {
                    if (checked) {
                      setThinkingBudget(-1);
                    } else {
                      setThinkingBudget(1024);
                    }
                  }}
                  description="Сколько токенов думающая модель (например, DeepSeek-R1) может потратить на внутренний процесс рассуждений. Отключение лимита дает модели свободу мысли."
                  presets={[
                    { label: "512", value: 512 },
                    { label: "1k", value: 1024 },
                    { label: "2k", value: 2048 },
                    { label: "4k", value: 4096 },
                    { label: "8k", value: 8192 },
                    { label: "16k", value: 16384 }
                  ]}
                />

                {/* 3. Overflow strategy selector */}
                <div className="flex flex-col gap-2.5 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all duration-300 md:col-span-2">
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal className="text-primary" size={14} />
                    <span className="text-xs font-bold text-foreground/90">Стратегия при переполнении контекста</span>
                  </div>
                  <p className="text-[9.5px] text-muted-foreground/80 leading-relaxed font-medium">
                    Определяет поведение системы, когда размер входного контекста приближается к границам модели.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1.5">
                    <button
                      onClick={() => setContextStrategy('sliding')}
                      className={cn(
                        "flex flex-col text-left p-3.5 rounded-xl border transition-all duration-300",
                        contextStrategy === 'sliding'
                          ? "bg-primary/5 border-primary text-foreground shadow-sm"
                          : "bg-transparent border-border/40 text-muted-foreground hover:border-border/80"
                      )}
                    >
                      <span className="text-xs font-black flex items-center gap-1.5">
                        <span className={cn("w-1.5 h-1.5 rounded-full", contextStrategy === 'sliding' ? "bg-primary animate-pulse" : "bg-muted-foreground")} />
                        «Скользящее окно» (Sliding Window)
                      </span>
                      <span className="text-[9px] font-medium mt-1.5 opacity-80 leading-normal">
                        Автоматически вытесняет старые сообщения из истории чата, освобождая место для новых ответов. Держит RAG-источники в приоритете.
                      </span>
                    </button>

                    <button
                      onClick={() => setContextStrategy('rag_priority')}
                      className={cn(
                        "flex flex-col text-left p-3.5 rounded-xl border transition-all duration-300",
                        contextStrategy === 'rag_priority'
                          ? "bg-purple-500/5 border-purple-500 text-purple-300 shadow-sm"
                          : "bg-transparent border-border/40 text-muted-foreground hover:border-border/80"
                      )}
                    >
                      <span className="text-xs font-black flex items-center gap-1.5">
                        <span className={cn("w-1.5 h-1.5 rounded-full", contextStrategy === 'rag_priority' ? "bg-purple-500 animate-pulse" : "bg-muted-foreground")} />
                        «Приоритет документов» (Document Priority)
                      </span>
                      <span className="text-[9px] font-medium mt-1.5 opacity-80 leading-normal">
                        Жестко удерживает все прикрепленные фрагменты документов в контексте. При нехватке памяти история чата мгновенно очищается.
                      </span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Область сообщений */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <MessageItem
              key={i}
              msg={msg}
              index={i}
              messagesLength={messages.length}
              onOpenSource={onOpenSource}
              setHoveredSource={setHoveredSource}
              setTooltipCoords={setTooltipCoords}
              tooltipTimeoutRef={tooltipTimeoutRef}
              notebook={notebook}
              question={msg.role === 'ai' && i > 0 ? messages[i - 1]?.content : ''}
              onBookmarkSaved={markMessageSaved}
            />
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Область ввода */}
      <div className="p-6 pt-0">
        {imagePreview && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-3 relative inline-block"
          >
            <img src={imagePreview} alt="Preview" className="h-20 w-auto rounded-xl border border-primary/30 shadow-lg" />
            <button
              onClick={removeImage}
              className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full shadow-md hover:scale-110 transition-transform"
            >
              <XIcon size={12} />
            </button>
          </motion.div>
        )}
        <div className="relative">
          <div className="flex items-end gap-2 bg-muted/20 border border-border/50 rounded-xl p-2 pl-4">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageChange}
              accept="image/*"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-3 text-muted-foreground hover:text-primary transition-colors"
              title="Прикрепить фото"
            >
              <ImageIcon size={20} />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={handlePaste}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Спросите что-нибудь об источниках..."
              className="flex-1 bg-transparent border-none outline-none resize-none py-3 text-sm focus:ring-0 focus:outline-none overflow-y-auto"
              rows={1}
            />
            {isLoading ? (
              <button
                onClick={() => abortController?.abort()}
                className="p-3 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-all shadow-lg shadow-red-500/20"
                title="Остановить генерацию"
              >
                <Square size={18} className="fill-current" />
              </button>
            ) : llmStatus.state === 'loading' ? (
              <button
                disabled
                className="p-3 bg-blue-500/50 text-white rounded-xl cursor-not-allowed"
                title="Дождитесь загрузки модели"
              >
                <RefreshCw size={18} className="animate-spin" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim() && !imagePreview}
                className={cn(
                  "p-3 rounded-xl transition-all",
                  (input.trim() || imagePreview) ? "bg-primary text-white shadow-lg shadow-primary/20" : "bg-muted text-muted-foreground"
                )}
                title="Отправить (Enter)"
              >
                <Send size={18} />
              </button>
            )}
          </div>
        </div>

        {/* Статистика */}
        <AnimatePresence>
          {stats && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center justify-center gap-2 mt-3 flex-wrap"
            >
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-muted/30 border border-border/30 text-[10px] font-medium text-muted-foreground">
                <Clock size={10} className="text-blue-400" />
                <span className="text-blue-400">{stats.elapsed_sec}с</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-muted/30 border border-border/30 text-[10px] font-medium text-muted-foreground">
                <FileText size={10} className="text-emerald-400" />
                <span className="text-emerald-400">~{stats.total_tokens}</span>
                <span>токенов</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-muted/30 border border-border/30 text-[10px] font-medium text-muted-foreground">
                <Zap size={10} className={stats.tokens_per_sec >= 20 ? "text-yellow-400" : stats.tokens_per_sec >= 10 ? "text-orange-400" : "text-red-400"} />
                <span className={stats.tokens_per_sec >= 20 ? "text-yellow-400" : stats.tokens_per_sec >= 10 ? "text-orange-400" : "text-red-400"}>
                  {stats.tokens_per_sec}
                </span>
                <span>т/с</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {/* Модальное окно настроек */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={llmSettings}
        onSave={(newSettings) => {
          setLlmSettings(newSettings);
          // F-fix #27: API-ключ выносим в sessionStorage (живёт только в текущей вкладке).
          // В localStorage — всё остальное (модели, температуры, URL).
          // При компрометации скрипта через XSS злоумышленник получит доступ
          // к localStorage, но НЕ к ключу (если вкладка закрыта — ключа уже нет).
          const { llm_api_key, ...rest } = newSettings;
          try { sessionStorage.setItem('llm_api_key', llm_api_key || ''); } catch (e) { /* sessionStorage disabled */ }
          localStorage.setItem('llm_settings', JSON.stringify(rest));
        }}
      />

      {/* Глобальный портал для тултипов — единый компонент, общий с модалом закладки */}
      <CitationTooltipPortal
        hoveredSource={hoveredSource && tooltipCoords.x ? { src: hoveredSource, ...tooltipCoords } : null}
        onClose={() => setHoveredSource(null)}
        onCancelClose={() => clearTimeout(tooltipTimeoutRef.current)}
        onResumeClose={() => {
          tooltipTimeoutRef.current = setTimeout(() => setHoveredSource(null), 100);
        }}
      />
    </div>
  );
}
