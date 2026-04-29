import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Trash2, Sparkles, Clock, Zap, Cpu, FileText, Settings as SettingsIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '../lib/utils';
import SettingsModal from './SettingsModal';

const Citation = ({ n, sources, onClick, onHover, onLeave }) => {
  const src = sources?.[n - 1];
  const btnRef = useRef(null);

  if (!src) return <span className="text-muted-foreground opacity-50 whitespace-nowrap">[{n}]</span>;

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

export default function ChatArea({ notebook, selectedSources, onOpenSource }) {
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Привет! Я проанализировал ваши источники и готов ответить на любые вопросы. Что вас интересует?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [llmSettings, setLlmSettings] = useState(() => {
    const saved = localStorage.getItem('llm_settings');
    return saved ? JSON.parse(saved) : {
      llm_url: 'http://localhost:1234/v1',
      llm_api_key: 'lm-studio',
      llm_model: 'gpt-4o'
    };
  });
  const [hoveredSource, setHoveredSource] = useState(null);
  const [tooltipCoords, setTooltipCoords] = useState({ x: 0, y: 0 });
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

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    if (selectedSources.length === 0) {
      alert('Выберите хотя бы один источник в боковой панели!');
      return;
    }

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setStats(null);

    const aiMsgIndex = messages.length + 1;
    setMessages(prev => [...prev, { role: 'ai', content: '', loading: true }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          allowed_files: selectedSources,
          max_tokens: maxTokens,
          notebook_id: notebook.id,
          ...llmSettings
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';
      let sources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
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
          } catch (e) {}
        }
      }
    } catch (err) {
      updateAiMessage(aiMsgIndex, '⚠️ Ошибка связи с сервером.', []);
    } finally {
      setIsLoading(false);
    }
  };

  const updateAiMessage = (index, content, sources) => {
    setMessages(prev => {
      const newMessages = [...prev];
      newMessages[index] = { role: 'ai', content, sources, loading: false };
      return newMessages;
    });
  };

  const clearChat = () => {
    setMessages([{ role: 'ai', content: 'Чат очищен. Какой новый вопрос?' }]);
    setStats(null);
  };

  const renderMessageContent = (msg) => {
    if (msg.loading) return (
      <div className="flex gap-1 py-2">
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.2s]" />
        <span className="w-1.5 h-1.5 bg-primary/40 rounded-full animate-bounce [animation-delay:0.4s]" />
      </div>
    );

    const processText = (child) => {
      if (child === null || child === undefined) return null;
      if (typeof child !== 'string') return child;
      const parts = child.split(/(\S*\s*\[[\d,\s]+\])/g);
      return parts.map((part, i) => {
        const match = part.match(/^(\S*)(\s*)\[([\d,\s]+)\]$/);
        if (match) {
          const word = match[1];
          const space = match[2];
          const nums = match[3].split(',').map(n => n.trim());
          return (
            <span key={i} className="whitespace-nowrap inline-flex items-center">
              {word}
              <span className="inline-flex ml-0.5">
                {nums.map((num, idx) => (
                  <Citation 
                    key={idx} 
                    n={parseInt(num)} 
                    sources={msg.sources} 
                    onClick={onOpenSource} 
                    onHover={(src, coords) => {
                      clearTimeout(tooltipTimeoutRef.current);
                      setHoveredSource(src);
                      setTooltipCoords(coords);
                    }}
                    onLeave={() => {
                      tooltipTimeoutRef.current = setTimeout(() => setHoveredSource(null), 200);
                    }}
                  />
                ))}
              </span>
            </span>
          );
        }
        return part;
      });
    };

    return (
      <div className="prose prose-invert prose-sm max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p>{React.Children.map(children, processText)}</p>,
            li: ({ children }) => <li>{React.Children.map(children, processText)}</li>,
            td: ({ children }) => <td>{React.Children.map(children, processText)}</td>,
            th: ({ children }) => <th>{React.Children.map(children, processText)}</th>,
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <div className="rounded-xl overflow-hidden my-4 border border-border/50">
                  <div className="bg-muted/50 px-4 py-1.5 border-b border-border/50 flex items-center justify-between">
                    <span className="text-[10px] font-bold text-muted-foreground uppercase">{match[1]}</span>
                  </div>
                  <SyntaxHighlighter
                    style={atomDark}
                    language={match[1]}
                    PreTag="div"
                    className="!m-0 !bg-[#1e1e1e]"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              ) : (
                <code className={cn("bg-muted px-1.5 py-0.5 rounded text-primary font-mono text-[11px]", className)} {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {msg.content}
        </ReactMarkdown>
        
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-border/20">
            {msg.sources.map((src, idx) => (
              <button 
                key={idx}
                onClick={() => onOpenSource(src)}
                className="flex items-center gap-2 bg-muted/30 hover:bg-primary/10 border border-border/40 hover:border-primary/30 px-2.5 py-1 rounded-full text-[9px] transition-all"
              >
                <span className="w-3.5 h-3.5 flex items-center justify-center bg-primary/20 text-primary rounded-full text-[8px] font-bold">
                  {idx + 1}
                </span>
                <span className="truncate max-w-[140px] opacity-70 group-hover:opacity-100">{src.file_name}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-background/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <Sparkles className="text-primary" size={18} />
          <h2 className="font-semibold text-sm">Интеллектуальный поиск</h2>
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
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/50 to-accent/50 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-1000"></div>
          <div className="relative flex items-end gap-2 bg-card border border-border rounded-2xl p-2 pl-4 shadow-2xl">
            <textarea 
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
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
            <button 
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={cn(
                "p-3 rounded-xl transition-all",
                input.trim() && !isLoading ? "bg-primary text-white shadow-lg shadow-primary/20" : "bg-muted text-muted-foreground"
              )}
            >
              <Send size={18} />
            </button>
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
              <span className="text-[9px] font-bold text-primary animate-pulse tracking-tight">Нажмите для перехода</span>
              <Sparkles size={10} className="text-primary/50" />
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
