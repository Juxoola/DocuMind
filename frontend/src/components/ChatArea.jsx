// Основная область чата: ввод, стриминг, рендер сообщений, закладки.
import React, { useState, useRef, useEffect, lazy, Suspense } from 'react';
import { Trash2, Sparkles, Settings as SettingsIcon, SlidersHorizontal, Cpu } from 'lucide-react';
import { cn } from '../lib/utils';
const SettingsModal = lazy(() => import('./SettingsModal'));
import { useVirtualizer } from '@tanstack/react-virtual';

import { MessageItem } from './MessageBubble';
import ChatInput from './ChatInput';
import SourcePanel from './SourcePanel';
import useStreamingHandler from './StreamingHandler';
import { AnswerModeSelect, TuningPanel, normalizeAnswerMode } from './ChatToolbar';

export default function ChatArea({ notebook, selectedSources, onOpenSource, llmSettings, setLlmSettings }) {
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Привет! Я проанализировал ваши источники и готов ответить на любые вопросы. Что вас интересует?', system: true }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [maxTokens, setMaxTokens] = useState(() => parseInt(localStorage.getItem('chat_max_tokens')) || 1024);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isTuningOpen, setIsTuningOpen] = useState(false);
  const [contextStrategy, setContextStrategy] = useState(() => localStorage.getItem('chat_context_strategy') || 'sliding');
  const [thinkingMode, setThinkingMode] = useState(() => localStorage.getItem('chat_thinking_mode') === 'true');
  const [thinkingBudget, setThinkingBudget] = useState(() => parseInt(localStorage.getItem('chat_thinking_budget')) || 1024);
  const [answerMode, setAnswerMode] = useState(() => normalizeAnswerMode(localStorage.getItem('chat_answer_mode')));
  const [hoveredSource, setHoveredSource] = useState(null);
  const [tooltipCoords, setTooltipCoords] = useState({ x: 0, y: 0 });
  const [abortController, setAbortController] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const tooltipTimeoutRef = useRef(null);
  const messagesScrollRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);

  const rowVirtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => messagesScrollRef.current,
    estimateSize: () => 240,
    overscan: 8,
  });

  const prevMessagesLength = React.useRef(messages.length);
  React.useEffect(() => {
    if (messages.length > prevMessagesLength.current) {
      rowVirtualizer.scrollToIndex(messages.length - 1, { align: 'end' });
    }
    prevMessagesLength.current = messages.length;
  }, [messages.length, rowVirtualizer]);

  const markMessageSaved = React.useCallback((msgIndex, ts) => {
    setMessages(prev => prev.map((m, i) => i === msgIndex ? { ...m, _savedAt: ts } : m));
  }, []);

  const measureElement = React.useCallback((el) => {
    if (el) rowVirtualizer.measureElement(el);
  }, [rowVirtualizer]);

  React.useEffect(() => {
    const onFill = (e) => {
      const text = e.detail?.text || '';
      if (text) setInput(text);
    };
    window.addEventListener('chat:fill-input', onFill);
    return () => window.removeEventListener('chat:fill-input', onFill);
  }, []);

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

  useEffect(() => {
    if (messages.length > 0) {
      rowVirtualizer.scrollToIndex(messages.length - 1, { align: 'end', behavior: 'smooth' });
    }
  }, [messages, rowVirtualizer]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
    };
  }, []);

  const [llmStatus, setLlmStatus] = useState({ state: 'ready', phase: null, model: null, error: null });
  const [contextUsage, setContextUsage] = useState(null);
  const contextIntervalRef = useRef(null);
  useEffect(() => {
    let es;
    let reconnectTimer;
    const connect = () => {
      es = new EventSource('/api/llm-status/stream');
      es.onmessage = (e) => {
        try { setLlmStatus(JSON.parse(e.data)); } catch {}
      };
      es.onerror = () => {
        es?.close();
        reconnectTimer = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      es?.close();
      clearTimeout(reconnectTimer);
    };
  }, []);

  const calcContextUsage = React.useCallback(() => {
    const ctxSize = llmSettings.use_gguf === 'true'
      ? parseInt(llmSettings.gguf_ctx_size) || 32768
      : parseInt(llmSettings.llm_ctx_size) || 8192;
    const totalChars = messages.reduce((sum, m) => sum + (m.content?.length || 0) + (m.thinkingContent?.length || 0), 0);
    const estimatedTokens = Math.round(totalChars * 0.25) + 500;
    setContextUsage({
      used: Math.min(estimatedTokens, ctxSize),
      total: ctxSize,
      pct: Math.round(Math.min(estimatedTokens, ctxSize) / ctxSize * 100),
    });
  }, [messages, llmSettings]);

  useEffect(() => {
    calcContextUsage();
    contextIntervalRef.current = setInterval(calcContextUsage, 5000);
    return () => clearInterval(contextIntervalRef.current);
  }, [calcContextUsage]);

  // Обработчики файлов/изображений
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file?.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
  };

  const handleFile = (file) => {
    if (file?.type.startsWith('image/')) {
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
      if (items[i].type.indexOf('image') !== -1) handleFile(items[i].getAsFile());
    }
  };

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); };

  // SSE-стриминг через хук
  const { handleSend: sendStreaming } = useStreamingHandler({
    messages, setMessages, llmSettings, setIsLoading, setStats,
    setAbortController, selectedSources, notebook, answerMode,
    thinkingMode, thinkingBudget, contextStrategy, maxTokens, input, setInput,
  });

  const handleSend = async () => {
    if ((!input.trim() && !selectedImage) || isLoading) return;
    if (llmStatus.state === 'loading') return;
    await sendStreaming();
  };

  const clearChat = () => {
    abortController?.abort();
    abortControllerRef.current = null;
    setMessages([{ role: 'ai', content: 'Чат очищен. Какой новый вопрос?', system: true }]);
    setStats(null);
    setIsLoading(false);
  };

  return (
    <div
      className="flex flex-col h-full w-full relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Панель инструментов */}
      <div className="flex items-center justify-center gap-3 p-4 border-b bg-background/80 backdrop-blur-md sticky top-0 z-10 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <div className="flex items-center gap-3 bg-muted/30 p-1.5 rounded-xl border border-white/5 px-2">
            <button onClick={() => setIsTuningOpen(!isTuningOpen)} title="Открыть параметры генерации" className={cn("flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all text-[10px] font-black uppercase tracking-tight", isTuningOpen ? "bg-primary/20 text-primary shadow-sm" : "text-muted-foreground hover:text-foreground")}>
              <SlidersHorizontal size={11} /><span>Параметры</span>
            </button>
            <div className="w-px h-4 bg-white/10 self-center" />
            <button onClick={() => setIsTuningOpen(!isTuningOpen)} className="px-2.5 py-1 text-[10px] font-black text-muted-foreground/80 hover:text-foreground transition-all flex items-center gap-1.5">
              <span>Ответ:</span><span className="text-primary font-black">{maxTokens} t</span>
            </button>
            {thinkingMode && (<>
              <div className="w-px h-4 bg-white/10 self-center" />
              <button onClick={() => setIsTuningOpen(!isTuningOpen)} className="px-2.5 py-1 text-[10px] font-black text-muted-foreground/80 hover:text-foreground transition-all flex items-center gap-1.5">
                <span>Бюджет:</span><span className="text-purple-400 font-black">{thinkingBudget === -1 ? "∞" : `${thinkingBudget} t`}</span>
              </button>
            </>)}
          </div>
          <button onClick={() => setThinkingMode(!thinkingMode)} className={cn("px-3 py-1.5 text-[10px] font-black uppercase tracking-wider rounded-xl transition-all flex items-center gap-1.5 border", thinkingMode ? "bg-purple-500/10 text-purple-400 border-purple-500/20 shadow-sm" : "bg-muted/40 text-muted-foreground hover:text-foreground border-transparent hover:border-border/60")}>
            <Sparkles size={11} />{thinkingMode ? "Думает" : "Без рассуждений"}
          </button>
          <AnswerModeSelect value={answerMode} onChange={setAnswerMode} />
          <button onClick={() => setIsSettingsOpen(true)} className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors"><SettingsIcon size={16} /></button>
          <button onClick={clearChat} className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors"><Trash2 size={16} /></button>
        </div>
      </div>

      <TuningPanel isTuningOpen={isTuningOpen} maxTokens={maxTokens} setMaxTokens={setMaxTokens} thinkingMode={thinkingMode} thinkingBudget={thinkingBudget} setThinkingBudget={setThinkingBudget} contextStrategy={contextStrategy} setContextStrategy={setContextStrategy} />

      {/* Виртуализированный список сообщений */}
      <div ref={messagesScrollRef} className="flex-1 overflow-y-auto p-6" style={{ contain: 'strict' }}>
        <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const msg = messages[virtualRow.index];
            const i = virtualRow.index;
            return (
              <div key={virtualRow.key} data-index={i} ref={measureElement} style={{ position: 'absolute', top: 0, left: 0, width: '100%', transform: `translateY(${virtualRow.start}px)`, paddingBottom: '2rem' }}>
                <MessageItem msg={msg} index={i} messagesLength={messages.length} onOpenSource={onOpenSource} setHoveredSource={setHoveredSource} setTooltipCoords={setTooltipCoords} tooltipTimeoutRef={tooltipTimeoutRef} notebook={notebook} question={msg.role === 'ai' && i > 0 ? messages[i - 1]?.content : ''} onBookmarkSaved={markMessageSaved} />
              </div>
            );
          })}
        </div>
      </div>

      <SourcePanel stats={stats} contextUsage={contextUsage} hoveredSource={hoveredSource} tooltipCoords={tooltipCoords} tooltipTimeoutRef={tooltipTimeoutRef} setHoveredSource={setHoveredSource} />

      <ChatInput input={input} setInput={setInput} isLoading={isLoading} imagePreview={imagePreview} removeImage={removeImage} handleSend={handleSend} handleImageChange={handleImageChange} handlePaste={handlePaste} handleDragOver={handleDragOver} handleDragLeave={handleDragLeave} isDragging={isDragging} abortController={abortController} abortControllerRef={abortControllerRef} llmStatus={llmStatus} textareaRef={textareaRef} />

      <Suspense fallback={null}>
        {isSettingsOpen && (
          <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={llmSettings} onSave={(newSettings) => {
            setLlmSettings(newSettings);
            const { llm_api_key, ...rest } = newSettings;
            try { sessionStorage.setItem('llm_api_key', llm_api_key || ''); } catch (e) { /* sessionStorage disabled */ }
            localStorage.setItem('llm_settings', JSON.stringify(rest));
          }} />
        )}
      </Suspense>
    </div>
  );
}
