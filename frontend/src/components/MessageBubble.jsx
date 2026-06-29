// Пузырь сообщения: рендер пользовательских и AI-ответов с источниками, закладками.
import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { ChevronRight, Zap, Bookmark, BookmarkCheck, Copy, Check, X as XIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import atomDark from 'react-syntax-highlighter/dist/esm/styles/prism/atom-dark';
import js from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import ts from 'react-syntax-highlighter/dist/esm/languages/prism/typescript';
import py from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash';
import jsonLang from 'react-syntax-highlighter/dist/esm/languages/prism/json';
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml';
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql';
import cssLang from 'react-syntax-highlighter/dist/esm/languages/prism/css';
import rustLang from 'react-syntax-highlighter/dist/esm/languages/prism/rust';
import cppLang from 'react-syntax-highlighter/dist/esm/languages/prism/cpp';

// Регистрируем только нужные языки — вместо ~2.5MB полного Prism
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('js', js);
SyntaxHighlighter.registerLanguage('jsx', js);
SyntaxHighlighter.registerLanguage('typescript', ts);
SyntaxHighlighter.registerLanguage('ts', ts);
SyntaxHighlighter.registerLanguage('tsx', ts);
SyntaxHighlighter.registerLanguage('python', py);
SyntaxHighlighter.registerLanguage('py', py);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('sh', bash);
SyntaxHighlighter.registerLanguage('shell', bash);
SyntaxHighlighter.registerLanguage('json', jsonLang);
SyntaxHighlighter.registerLanguage('yaml', yaml);
SyntaxHighlighter.registerLanguage('yml', yaml);
SyntaxHighlighter.registerLanguage('sql', sql);
SyntaxHighlighter.registerLanguage('css', cssLang);
SyntaxHighlighter.registerLanguage('rust', rustLang);
SyntaxHighlighter.registerLanguage('rs', rustLang);
SyntaxHighlighter.registerLanguage('cpp', cppLang);
SyntaxHighlighter.registerLanguage('c', cppLang);
SyntaxHighlighter.registerLanguage('html', cppLang);
SyntaxHighlighter.registerLanguage('xml', cppLang);
import { cn } from '../lib/utils';
import { preProcessMessage, loadMathPlugins } from '../lib/markdownRender';
import { CitationButton } from '../lib/CitationTooltip';
import { extractCleanContent, copyAsRichText } from '../lib/copyToClipboard';
import axios from 'axios';

// Блок рассуждений модели (thinking/CoT)
const ThinkingBlock = ({ content, isStreaming }) => {
  const [open, setOpen] = useState(true);
  const bodyRef = useRef(null);

  useEffect(() => {
    if (isStreaming && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [content, isStreaming]);

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
      <div className={cn(
        "grid transition-all duration-200 ease-out",
        open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
      )}>
        <div className="overflow-hidden">
          <div
            ref={bodyRef}
            className="px-4 pb-4 pt-2 text-[11px] text-muted-foreground/90 italic border-t border-purple-500/10 whitespace-pre-wrap max-h-96 overflow-y-auto leading-relaxed custom-scrollbar bg-purple-500/5"
          >
            {content}
          </div>
        </div>
      </div>
    </div>
  );
};

// Кнопка-сноска в тексте
const Citation = ({ n, sources, onClick, onHover, onLeave }) => {
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

// Основной компонент пузыря сообщения
const MessageItem = React.memo(({
  msg,
  index,
  messagesLength,
  onOpenSource,
  setHoveredSource,
  setTooltipCoords,
  tooltipTimeoutRef,
  notebook,
  question,
  onBookmarkSaved,
}) => {
  const [bmPopover, setBmPopover] = useState(false);
  const [bmTitle, setBmTitle] = useState('');
  const [bmTags, setBmTags] = useState('');
  const [bmSaving, setBmSaving] = useState(false);
  const [bmError, setBmError] = useState('');
  const bmBtnRef = useRef(null);
  const bmPopoverRef = useRef(null);
  const [bmCoords, setBmCoords] = useState({ top: 0, left: 0 });
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef(null);
  const bubbleRef = useRef(null);
  const [mathPlugins, setMathPlugins] = useState(null);

  // ── Lazy-загрузка math-плагинов: только при наличии $ в тексте ──
  useEffect(() => {
    if (mathPlugins) return;
    if (msg.content && /\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\[[\s\S]*?\\]/.test(msg.content)) {
      loadMathPlugins().then(setMathPlugins);
    }
  }, [msg.content, mathPlugins]);

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

  useEffect(() => () => clearTimeout(copyTimeoutRef.current), []);

  const openBmPopover = () => {
    setBmTitle('');
    setBmTags('');
    setBmError('');
    if (bmBtnRef.current) {
      const rect = bmBtnRef.current.getBoundingClientRect();
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

  useEffect(() => {
    if (!bmPopover) return;
    const onDown = (e) => {
      if (bmPopoverRef.current && !bmPopoverRef.current.contains(e.target)) {
        if (bmBtnRef.current && bmBtnRef.current.contains(e.target)) return;
        setBmPopover(false);
      }
    };
    const onKey = (e) => { if (e.key === 'Escape') setBmPopover(false); };
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
      if (onBookmarkSaved) onBookmarkSaved(msg.id || index, Date.now());
      window.dispatchEvent(new CustomEvent('bookmark:added'));
      setBmPopover(false);
    } catch (e) {
      setBmError(e?.response?.data?.detail || 'Не удалось сохранить');
    } finally {
      setBmSaving(false);
    }
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

        {msg.thinkingContent && (
          <ThinkingBlock
            content={msg.thinkingContent}
            isStreaming={!msg.thinkingDone}
          />
        )}
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, ...(mathPlugins ? [mathPlugins.remarkMath] : [])]}
            rehypePlugins={mathPlugins ? [mathPlugins.rehypeKatex] : []}
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

          {/* Кнопки-источники под ответом */}
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

          {/* Действия AI-ответа: закладки + копирование */}
          {msg.role === 'ai' && !msg.loading && msg.content && notebook && !msg.system && (
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

          {/* Действия пользовательского сообщения: копирование */}
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
    <div
      className={cn(
        "flex w-full animate-fadeInUp",
        msg.role === 'user' ? "justify-end" : "justify-start"
      )}
    >
      <div ref={bubbleRef} className={cn(
        "max-w-[85%] group relative min-w-0",
        msg.role === 'user' ? "chat-bubble-user" : "chat-bubble-ai"
      )} style={{ zIndex: messagesLength - index }}>
        {renderContent()}
      </div>

      {/* Попап закладки */}
      {bmPopover && msg.role === 'ai' && createPortal(
          <div
            ref={bmPopoverRef}
            style={{ position: 'fixed', top: bmCoords.top, left: bmCoords.left, zIndex: 9999 }}
            className="w-80 p-3 bg-card border border-border shadow-2xl rounded-2xl animate-scaleIn"
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
          </div>,
        document.body
      )}
    </div>
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

export { ThinkingBlock, Citation, MessageItem };
export default MessageItem;
