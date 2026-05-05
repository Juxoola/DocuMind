import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Trash2, Sparkles, Clock, Zap, Cpu, FileText, Settings as SettingsIcon, HardDrive, Square, Image as ImageIcon, Plus, X as XIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '../lib/utils';
import SettingsModal from './SettingsModal';

const Citation = ({ n, sources, onClick, onHover, onLeave }) => {
  const src = sources?.[n - 1];
  const btnRef = useRef(null);

  if (!src) {
    return <span className="text-muted-foreground opacity-50 whitespace-nowrap">[{n}]</span>;
  }

  const handleMouseEnter = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      onHover(src, { x: rect.left + rect.width / 2, y: rect.top });
    }
  };

  return (
    <span className="relative inline-flex items-center align-baseline">
      <button 
        ref={btnRef}
        onClick={(e) => { e.preventDefault(); onClick(src); }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={onLeave}
        className="inline-flex items-center justify-center w-4 h-4 bg-primary/20 text-primary text-[9px] font-bold rounded-full hover:bg-primary hover:text-white transition-all shadow-sm border border-primary/20 relative -top-[1px]"
      >
        {n}
      </button>
    </span>
  );
};

export default function ChatArea({ notebook, selectedSources, onOpenSource, llmSettings, setLlmSettings }) {
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Привет! Я проанализировал ваши источники и готов ответить на любые вопросы. Что вас интересует?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [thinkingMode, setThinkingMode] = useState(false);
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
    setMessages(prev => [...prev, { role: 'ai', content: '', loading: true }]);
    
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

  const renderMessageContent = (msg) => {
    if (msg.loading) return (
      <div className="flex gap-1 py-2">
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.2s]" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.4s]" />
      </div>
    );

    const preProcessMessage = (text) => {
      if (!text) return "";
      let processed = text
        .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$')
        .replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');

      processed = processed.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (match, nums) => {
        return nums.split(',').map(n => {
          const num = n.trim();
          return `[${num}](#cite:${num})`;
        }).join('');
      });

      if (processed.includes('<think>')) {
        let parts = processed.split('<think>');
        let beforeThink = parts[0];
        let thinkContent = parts[1];
        if (thinkContent.includes('</think>')) {
            let thinkParts = thinkContent.split('</think>');
            processed = `${beforeThink}<details class="mb-4 rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden"><summary class="cursor-pointer px-4 py-2 text-[11px] font-bold text-purple-500 hover:bg-purple-500/10 transition-colors select-none">✨ Рассуждения</summary><div class="p-4 text-xs text-muted-foreground/80 italic border-t border-purple-500/10 whitespace-pre-wrap">${thinkParts[0]}</div></details>\n${thinkParts.slice(1).join('</think>')}`;
        } else {
            processed = `${beforeThink}<div class="mb-4 rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden"><div class="px-4 py-2 text-[11px] font-bold text-purple-500 flex items-center gap-2"><span class="animate-pulse">✨ Модель рассуждает...</span></div><div class="p-4 text-xs text-muted-foreground/80 italic border-t border-purple-500/10 whitespace-pre-wrap">${thinkContent}</div></div>`;
        }
      }
      return processed;
    };

    return (
      <div className="flex flex-col gap-2">
        {msg.image && (
          <div className="mb-2 rounded-lg overflow-hidden border border-border/40 max-w-sm">
            <img src={msg.image} alt="User upload" className="w-full h-auto object-cover" />
          </div>
        )}
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeKatex]}
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith('#cite:')) {
                  const num = href.split(':')[1];
                  return (
                    <span className="inline-block ml-1 no-underline">
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
                return !inline && match ? (
                  <div className="relative group my-4">
                    <div className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button 
                        onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                        className="p-1.5 bg-white/10 hover:bg-white/20 rounded text-xs text-white/70 flex items-center gap-1"
                      >
                        <Zap size={12} /> Copy
                      </button>
                    </div>
                    <SyntaxHighlighter
                      style={atomDark}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-lg !bg-slate-900/50 !p-4 border border-white/5"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
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
            <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-border/20">
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
        </div>
      </div>
    );
  };

  return (
    <div 
      className="flex flex-col h-full w-full max-w-4xl mx-auto relative"
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

      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-background/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <FileText className="text-muted-foreground" size={18} />
          <h2 className="font-medium text-sm">Ассистент по документам</h2>
          {llmSettings.use_gguf === 'true' && (
            <span className="flex items-center gap-1 px-2 py-0.5 bg-green-500/10 text-green-500 rounded-full text-[9px] font-bold border border-green-500/20">
              <HardDrive size={10} /> GGUF
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div className="flex bg-muted p-1 rounded-lg">
            {[512, 1024, 2048].map(tokens => (
              <button 
                key={tokens}
                onClick={() => setMaxTokens(tokens)}
                className={cn(
                  "px-3 py-1 text-[10px] font-bold rounded-md transition-all",
                  maxTokens === tokens ? "bg-background text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {tokens === 512 ? 'Короткий' : tokens === 1024 ? 'Средний' : 'Длинный'}
              </button>
            ))}
          </div>
          <button
            onClick={() => setThinkingMode(!thinkingMode)}
            className={cn(
              "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all flex items-center gap-1.5",
              thinkingMode ? "bg-purple-500/10 text-purple-500 border border-purple-500/20 shadow-sm" : "bg-muted text-muted-foreground hover:text-foreground border border-transparent"
            )}
          >
            <Sparkles size={12} />
            {thinkingMode ? "Думает" : "Без рассуждений"}
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

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "flex w-full",
                msg.role === 'user' ? "justify-end" : "justify-start"
              )}
            >
              <div className={cn(
                "max-w-[85%] group relative",
                msg.role === 'user' ? "chat-bubble-user" : "chat-bubble-ai"
              )} style={{ zIndex: messages.length - i }}>
                {renderMessageContent(msg)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
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

        {/* Stats */}
        <AnimatePresence>
          {stats && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-center gap-6 mt-4 text-[10px] text-muted-foreground/60 font-medium"
            >
              <div className="flex items-center gap-1.5"><Clock size={12}/> {stats.elapsed_sec}s</div>
              <div className="flex items-center gap-1.5"><Zap size={12}/> {stats.total_tokens} tokens</div>
              <div className="flex items-center gap-1.5"><Cpu size={12}/> {stats.tokens_per_sec} t/s</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {/* Settings Modal */}
      <SettingsModal 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={llmSettings}
        onSave={(newSettings) => {
          setLlmSettings(newSettings);
          localStorage.setItem('llm_settings', JSON.stringify(newSettings));
        }}
      />

      {/* Global Tooltip Portal (Stacking Context Fix) */}
      <AnimatePresence>
        {hoveredSource && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            onMouseEnter={() => clearTimeout(tooltipTimeoutRef.current)}
            onMouseLeave={() => setHoveredSource(null)}
            className="fixed w-80 p-4 bg-card/98 border border-border/80 shadow-[0_20px_50px_rgba(0,0,0,0.5)] rounded-2xl z-[9999] pointer-events-auto backdrop-blur-2xl text-left"
            style={{ 
              left: Math.max(20, Math.min(window.innerWidth - 340, tooltipCoords.x - 160)),
              bottom: window.innerHeight - tooltipCoords.y + 12
            }}
          >
            <div className="flex items-center gap-3 mb-3 pb-2 border-b border-border/40">
              <div className="p-1.5 bg-primary/15 rounded-xl text-primary shadow-inner">
                <FileText size={14} />
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-[11px] font-bold truncate text-primary uppercase tracking-wider">{hoveredSource.file_name}</span>
                <div className="flex gap-2 mt-0.5">
                   {hoveredSource.page && <span className="text-[9px] text-muted-foreground font-medium">Стр {hoveredSource.page}</span>}
                   {hoveredSource.time !== undefined && <span className="text-[9px] text-muted-foreground font-medium">• {Math.floor(hoveredSource.time / 60)}:{(hoveredSource.time % 60).toString().padStart(2, '0')}</span>}
                </div>
              </div>
            </div>
            <div className="max-h-48 overflow-y-auto pr-1 custom-scrollbar">
              <p className="text-[12px] leading-relaxed text-foreground/90 italic font-medium whitespace-pre-wrap">
                "{hoveredSource.text.replace(/^Source \d+:\s*/im, '').substring(0, 600)}..."
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-border/40 flex justify-between items-center">
              <span className="text-[9px] font-medium text-muted-foreground uppercase tracking-wider">Нажмите для перехода</span>
            </div>
            <div 
              className="absolute top-full border-[8px] border-transparent border-t-card/98" 
              style={{ left: Math.max(15, Math.min(305, tooltipCoords.x - (Math.max(20, Math.min(window.innerWidth - 340, tooltipCoords.x - 160))) - 8)) }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
