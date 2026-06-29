// Боковая панель: список файлов, загрузка, состояние ingestion.
import React, { useState } from 'react';
import {
  FileText,
  Upload,
  X,
  LogOut,
  ChevronLeft,
  CheckCircle2,
  AlertCircle,
  Database,
  Trash2,
  PowerOff,
  Video,
  Music,
  FileCode,
  FileSpreadsheet,
  File,
  Image as ImageIcon,
  Bookmark,
  Search,
  Tag as TagIcon,
  Copy,
  RotateCcw,
  Eye,
  Pencil,
  ExternalLink,
  Check,
} from 'lucide-react';

function getFileIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  switch (ext) {
    case 'pdf':
      return <FileText size={14} className="text-red-400" />;
    case 'docx':
    case 'doc':
      return <FileText size={14} className="text-blue-400" />;
    case 'pptx':
    case 'ppt':
      return <FileText size={14} className="text-orange-400" />;
    case 'xlsx':
    case 'xls':
    case 'csv':
      return <FileSpreadsheet size={14} className="text-green-400" />;
    case 'mp4':
    case 'avi':
    case 'mov':
    case 'mkv':
      return <Video size={14} className="text-purple-400" />;
    case 'mp3':
    case 'wav':
    case 'm4a':
    case 'ogg':
      return <Music size={14} className="text-yellow-400" />;
    case 'py':
    case 'js':
    case 'ts':
    case 'jsx':
    case 'tsx':
    case 'json':
      return <FileCode size={14} className="text-cyan-400" />;
    case 'txt':
    case 'md':
      return <FileText size={14} className="text-muted-foreground" />;
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'webp':
    case 'gif':
    case 'bmp':
      return <ImageIcon size={14} className="text-pink-400" />;
    default:
      return <File size={14} className="text-muted-foreground" />;
  }
}
import { cn } from '../lib/utils';
import axios from 'axios';
import { LlmMarkdown } from '../lib/markdownRender';
import { extractCleanContent, copyAsRichText } from '../lib/copyToClipboard';

export default function Sidebar({ 
  notebook, 
  sources, 
  selectedSources, 
  onSelectSources, 
  onRefresh, 
  onExit,
  onOpenFile,
  width,
  onToggle,
  uploadState,
  onUpload
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [llamaCount, setLlamaCount] = useState(0);

  React.useEffect(() => {
    const checkLlama = async () => {
      try {
        const res = await axios.get('/api/gguf-status');
        setLlamaCount(res.data.running_count);
      } catch (err) {}
    };
    checkLlama();
    const timer = setInterval(checkLlama, 10000);
    return () => clearInterval(timer);
  }, []);

  const killAllLlama = async () => {
    if (!confirm('Принудительно завершить ВСЕ процессы llama-server? Это освободит VRAM, но может прервать текущую генерацию.')) return;
    try {
      await axios.post('/api/gguf-kill-all');
      const res = await axios.get('/api/gguf-status');
      setLlamaCount(res.data.running_count);
    } catch (err) {
      alert('Ошибка при завершении процессов');
    }
  };

  const handleUpload = (files) => {
    onUpload(files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files);
    }
  };

  const deleteFile = async (e, filename) => {
    e.stopPropagation();
    if (!confirm(`Удалить источник "${filename}"?`)) return;
    try {
      await axios.delete(`/api/files/${encodeURIComponent(filename)}?notebook_id=${notebook.id}`);
      onRefresh();
    } catch (err) {
      alert('Ошибка удаления');
    }
  };

  const clearDatabase = async () => {
    if (!confirm('Это удалит все проиндексированные данные. Продолжить?')) return;
    try {
      await axios.delete(`/api/clear?notebook_id=${notebook.id}`);
      alert('База очищена');
      onRefresh();
    } catch (err) {
      alert('Ошибка');
    }
  };

  // Закладки — полный CRUD с фильтрацией по тегам, поиском и модалами просмотра/редактирования
  const [activeTab, setActiveTab] = useState('files');
  const [bookmarks, setBookmarks] = useState([]);
  const [bmSearch, setBmSearch] = useState('');
  const [bmTagFilter, setBmTagFilter] = useState(null);
  const [bmLoading, setBmLoading] = useState(false);
  const [viewingBm, setViewingBm] = useState(null);
  const [editingBm, setEditingBm] = useState(null);
  const bmAnswerRef = React.useRef(null);
  const [bmCopied, setBmCopied] = useState(false);
  const bmCopyTimeoutRef = React.useRef(null);

  const fetchBookmarks = async () => {
    setBmLoading(true);
    try {
      const r = await axios.get('/api/bookmarks', { params: { notebook_id: notebook.id } });
      setBookmarks(r.data.bookmarks || []);
    } catch (e) {
      console.error('bookmarks fetch failed', e);
    } finally {
      setBmLoading(false);
    }
  };

  React.useEffect(() => {
    if (activeTab === 'bookmarks') fetchBookmarks();
  }, [activeTab, notebook.id]);

  // Обновление списка закладок через кастомные события — избегаем prop drilling между ChatArea и Sidebar
  React.useEffect(() => {
    const onAdded = () => { if (activeTab === 'bookmarks') fetchBookmarks(); };
    const onDeleted = () => { if (activeTab === 'bookmarks') fetchBookmarks(); };
    const onStale = () => { if (activeTab === 'bookmarks') fetchBookmarks(); };
    window.addEventListener('bookmark:added', onAdded);
    window.addEventListener('bookmark:deleted', onDeleted);
    window.addEventListener('bookmark:stale', onStale);
    return () => {
      window.removeEventListener('bookmark:added', onAdded);
      window.removeEventListener('bookmark:deleted', onDeleted);
      window.removeEventListener('bookmark:stale', onStale);
    };
  }, [activeTab, notebook.id]);

  const prevSourcesCount = React.useRef(sources.length);
  React.useEffect(() => {
    if (sources.length < prevSourcesCount.current && activeTab === 'bookmarks') {
      fetchBookmarks();
    }
    prevSourcesCount.current = sources.length;
  }, [sources.length, activeTab]);

  const deleteBookmark = async (id) => {
    if (!confirm('Удалить закладку?')) return;
    try {
      await axios.delete(`/api/bookmarks/${id}`, { params: { notebook_id: notebook.id } });
      setBookmarks(prev => prev.filter(b => b.id !== id));
      window.dispatchEvent(new CustomEvent('bookmark:deleted', { detail: { id } }));
    } catch (e) {
      alert('Не удалось удалить закладку');
    }
  };

  const saveBookmarkEdit = async () => {
    if (!editingBm) return;
    try {
      const tagsArr = typeof editingBm.tags === 'string'
        ? editingBm.tags.split(',').map(t => t.trim()).filter(Boolean)
        : (editingBm.tags || []);
      const r = await axios.patch(`/api/bookmarks/${editingBm.id}`, {
        notebook_id: notebook.id,
        title: editingBm.title,
        tags: tagsArr,
      });
      setBookmarks(prev => prev.map(b => b.id === editingBm.id ? r.data : b));
      setEditingBm(null);
    } catch (e) {
      alert('Не удалось обновить закладку');
    }
  };

  const askAgain = (question) => {
    // askAgain: копирует вопрос в чат через CustomEvent для ChatArea
    if (navigator.clipboard) navigator.clipboard.writeText(question);
    window.dispatchEvent(new CustomEvent('chat:fill-input', { detail: { text: question } }));
    setViewingBm(null);
  };

  const copyText = async (text) => {
    const root = bmAnswerRef.current;
    const extracted = extractCleanContent(root) || { html: '', text: text || '' };
    if (!extracted.html && !extracted.text) return;
    const ok = await copyAsRichText(extracted);
    if (ok) {
      setBmCopied(true);
      clearTimeout(bmCopyTimeoutRef.current);
      bmCopyTimeoutRef.current = setTimeout(() => setBmCopied(false), 1500);
    }
  };

  const allTags = React.useMemo(() => {
    const set = new Set();
    bookmarks.forEach(b => (b.tags || []).forEach(t => set.add(t)));
    return Array.from(set).sort();
  }, [bookmarks]);

  const filteredBookmarks = React.useMemo(() => {
    const q = bmSearch.trim().toLowerCase();
    return bookmarks.filter(b => {
      if (bmTagFilter && !(b.tags || []).includes(bmTagFilter)) return false;
      if (!q) return true;
      const hay = `${b.title || ''}\n${b.question}\n${b.answer}\n${(b.tags || []).join(' ')}`.toLowerCase();
      return hay.includes(q);
    });
  }, [bookmarks, bmSearch, bmTagFilter]);

  const formatRel = (ts) => {
    if (!ts) return '';
    const diff = Date.now() / 1000 - ts;
    if (diff < 60) return 'только что';
    if (diff < 3600) return `${Math.floor(diff / 60)} мин назад`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} ч назад`;
    if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} д назад`;
    return new Date(ts * 1000).toLocaleDateString('ru-RU');
  };

  const modelShort = (m) => {
    if (!m) return '';
    return m.split(/[\\/]/).filter(Boolean).slice(-1)[0] || m;
  };

  // Проверка: источник удалён — кнопка «открыть» блокируется, но карточка остаётся
  const bmSourceStale = (fileName) => {
    return viewingBm?.status === 'stale' || !sources.includes(fileName);
  };

  return (
    <div style={{ width }} className="h-full border-r glass flex flex-col z-10 overflow-hidden">
      <div className="p-4 border-b flex items-center justify-between flex-shrink-0 bg-background/50 backdrop-blur-sm">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 bg-primary/10 rounded-xl flex-shrink-0 flex items-center justify-center text-primary shadow-sm">
            <Database size={16} />
          </div>
          <div className="flex flex-col min-w-0">
            <span className="font-bold text-[10px] uppercase tracking-widest text-muted-foreground/50">Блокнот</span>
            <span className="font-bold text-sm truncate">{notebook.name}</span>
          </div>
        </div>
        <button 
          onClick={onToggle}
          className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors"
          title="Свернуть панель"
        >
          <ChevronLeft size={18} />
        </button>
      </div>

      <div className="p-4 flex-shrink-0">
        <label 
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={cn(
            "relative flex flex-col items-center justify-center p-6 border-2 border-dashed border-border rounded-2xl cursor-pointer hover:border-primary/50 hover:bg-primary/5 transition-all group",
            isDragging && "border-primary bg-primary/10 scale-[1.02]",
            uploadState?.isUploading && "pointer-events-none opacity-50"
          )}
        >
          <input type="file" className="hidden" multiple onChange={(e) => handleUpload(e.target.files)} />
          <Upload className={cn(
            "mb-2 transition-colors",
            isDragging ? "text-primary scale-110" : "text-muted-foreground group-hover:text-primary"
          )} size={24} />
          <span className="text-xs font-bold text-muted-foreground group-hover:text-primary text-center">
            {isDragging ? 'Отпустите для загрузки' : 'Добавить источники'}
          </span>
          <span className="text-[10px] text-muted-foreground/60 mt-1">PDF, DOCX, Video, Audio</span>

          {uploadState?.isUploading && (
            <div className="absolute inset-0 bg-background/90 backdrop-blur-md rounded-2xl flex flex-col items-center justify-center p-6 border border-primary/20 shadow-2xl">
              <div className="w-full mb-4">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-[9px] font-black uppercase tracking-widest text-primary">Файл</span>
                  <span className="text-[9px] font-bold text-primary">{Math.round(uploadState.progress)}%</span>
                </div>
                <div className="w-full bg-primary/10 h-1.5 rounded-full overflow-hidden shadow-inner">
                  <div 
                    className="bg-primary h-full shadow-[0_0_10px_rgba(var(--primary),0.5)] transition-[width] duration-500 ease-out" 
                    style={{ width: `${uploadState.progress}%` }}
                  />
                </div>
              </div>

              <div className="w-full mb-6">
                <div className="flex justify-between items-center mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">Всего</span>
                    <span className="text-[9px] font-bold text-muted-foreground/60 bg-muted px-1.5 py-0.5 rounded">
                      {uploadState.currentFile} из {uploadState.totalFiles}
                    </span>
                  </div>
                  <span className="text-[9px] font-bold text-muted-foreground">{Math.round(uploadState.batchProgress)}%</span>
                </div>
                <div className="w-full bg-muted h-1 rounded-full overflow-hidden">
                  <div 
                    className="bg-muted-foreground/40 h-full transition-[width] duration-300 ease-out" 
                    style={{ width: `${uploadState.batchProgress}%` }}
                  />
                </div>
              </div>

              <div className="absolute bottom-0 left-0 right-0 p-2 bg-primary/10 border-t border-primary/20 backdrop-blur-sm">
                <p className="text-[9px] font-bold text-primary text-center break-words leading-tight">
                  {uploadState.status || 'Подготовка...'}
                </p>
              </div>
            </div>
          )}
        </label>
      </div>

      <div className="px-4 pt-3 flex-shrink-0">
        <div className="flex gap-1 p-1 bg-muted/30 rounded-xl border border-white/5">
          <button
            onClick={() => setActiveTab('files')}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider transition-all",
              activeTab === 'files'
                ? "bg-primary/15 text-primary shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <FileText size={11} />
            <span>Файлы</span>
            <span className="text-[9px] opacity-60">({sources.length})</span>
          </button>
          <button
            onClick={() => setActiveTab('bookmarks')}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wider transition-all",
              activeTab === 'bookmarks'
                ? "bg-amber-500/15 text-amber-400 shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Bookmark size={11} />
            <span>Закладки</span>
            <span className="text-[9px] opacity-60">({bookmarks.length})</span>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 custom-scrollbar">
        {activeTab === 'files' && (() => {
          const filtered = sources.filter(s => {
            const low = s.toLowerCase();
            if (low.endsWith('.pdf')) {
              const base = s.slice(0, -4);
              return !sources.some(other => {
                const olow = other.toLowerCase();
                return olow === base.toLowerCase() + '.pptx' || olow === base.toLowerCase() + '.ppt';
              });
            }
            return true;
          });
          
          return (
            <>
              <div className="flex items-center justify-between px-2 mb-2">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Источники ({filtered.length})</span>
                <button onClick={clearDatabase} className="text-destructive hover:bg-destructive/10 p-1 rounded-md transition-colors" title="Очистить базу">
                   <Database size={12} />
                </button>
              </div>
              {filtered.map((file) => (
                <div 
                  key={file}
                  className={cn(
                    "group flex items-center justify-between gap-3 p-3 rounded-xl border transition-all cursor-pointer",
                    selectedSources.includes(file) 
                      ? "bg-primary/5 border-primary/20" 
                      : "bg-transparent border-transparent hover:bg-muted"
                  )}
                  onClick={() => onOpenFile(file)}
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <div className="p-1.5 bg-muted rounded-lg group-hover:bg-primary/10 transition-colors flex-shrink-0">
                      {getFileIcon(file)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-medium truncate group-hover:text-primary transition-colors">{file}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button 
                      onClick={(e) => deleteFile(e, file)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded-lg transition-all"
                    >
                      <Trash2 size={16} />
                    </button>
                    
                    <div 
                      className={cn(
                        "transition-all duration-300 transform cursor-pointer",
                        selectedSources.includes(file) 
                          ? "text-primary scale-125 opacity-100" 
                          : "text-muted-foreground/20 scale-100 opacity-0 group-hover:opacity-30"
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (selectedSources.includes(file)) {
                          onSelectSources(selectedSources.filter(s => s !== file));
                        } else {
                          onSelectSources([...selectedSources, file]);
                        }
                      }}
                    >
                      <CheckCircle2 size={24} strokeWidth={2.5} />
                    </div>
                  </div>
                </div>
              ))}
              {filtered.length === 0 && !uploadState?.isUploading && (
                <div className="text-center py-8 opacity-40">
                  <FileText size={32} className="mx-auto mb-2" />
                  <p className="text-[10px]">Нет источников</p>
                </div>
              )}
            </>
          );
        })()}

        {activeTab === 'bookmarks' && (
          <div className="space-y-2">
            <div className="relative px-1">
              <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                value={bmSearch}
                onChange={(e) => setBmSearch(e.target.value)}
                placeholder="Поиск по закладкам…"
                className="w-full bg-muted/30 border border-border/40 rounded-lg pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-primary/50"
              />
            </div>

            {allTags.length > 0 && (
              <div className="flex flex-wrap gap-1 px-1">
                <button
                  onClick={() => setBmTagFilter(null)}
                  className={cn(
                    "px-2 py-0.5 rounded-full text-[9px] font-bold transition-all border",
                    !bmTagFilter
                      ? "bg-primary/10 text-primary border-primary/30"
                      : "bg-muted/30 text-muted-foreground border-transparent hover:border-border/60"
                  )}
                >
                  все
                </button>
                {allTags.map(t => (
                  <button
                    key={t}
                    onClick={() => setBmTagFilter(bmTagFilter === t ? null : t)}
                    className={cn(
                      "px-2 py-0.5 rounded-full text-[9px] font-bold transition-all border flex items-center gap-1",
                      bmTagFilter === t
                        ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
                        : "bg-muted/30 text-muted-foreground border-transparent hover:border-border/60"
                    )}
                  >
                    <TagIcon size={9} />
                    {t}
                  </button>
                ))}
              </div>
            )}

            {bmLoading ? (
              <div className="text-center py-8 opacity-40 text-[10px]">Загрузка…</div>
            ) : filteredBookmarks.length === 0 ? (
              <div className="text-center py-8 opacity-40">
                <Bookmark size={32} className="mx-auto mb-2" />
                <p className="text-[10px]">
                  {bookmarks.length === 0
                    ? 'Нет закладок. Нажмите «☆» на ответе, чтобы сохранить.'
                    : 'Ничего не найдено по фильтру.'}
                </p>
              </div>
            ) : (
              filteredBookmarks.map(bm => (
                <div
                  key={bm.id}
                  className={cn(
                    "group p-3 rounded-xl border bg-transparent hover:bg-muted/40 transition-all",
                    bm.status === 'stale'
                      ? "border-amber-500/30 bg-amber-500/5"
                      : "border-border/40"
                  )}
                >
                  <div className="flex items-start gap-2 mb-1.5">
                    <p
                      className="text-xs font-bold text-foreground line-clamp-2 flex-1 cursor-pointer hover:text-primary transition-colors"
                      onClick={() => setViewingBm(bm)}
                      title="Открыть"
                    >
                      {bm.title || bm.question}
                    </p>
                    {bm.status === 'stale' && (
                      <span title="Источник удалён — ответ может быть неактуальным" className="shrink-0 text-amber-400">
                        <AlertCircle size={12} />
                      </span>
                    )}
                  </div>

                  {bm.tags && bm.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-1.5">
                      {bm.tags.map(t => (
                        <span key={t} className="px-1.5 py-0 rounded-md bg-muted/60 text-[9px] font-medium text-muted-foreground">
                          #{t}
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="flex items-center gap-2 text-[9px] text-muted-foreground/70 mb-2">
                    {bm.model && (
                      <span className="truncate max-w-[120px]" title={bm.model}>
                        {modelShort(bm.model)}
                      </span>
                    )}
                    <span>·</span>
                    <span>{formatRel(bm.created_at)}</span>
                    {bm.answer_mode && bm.answer_mode !== 'concise' && (
                      <>
                        <span>·</span>
                        <span className="text-amber-400">разв.</span>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-1 opacity-70 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => setViewingBm(bm)}
                      className="flex-1 flex items-center justify-center gap-1 px-2 py-1 hover:bg-primary/10 text-muted-foreground hover:text-primary rounded-md text-[10px] font-bold transition-all"
                      title="Открыть"
                    >
                      <Eye size={11} />
                      Открыть
                    </button>
                    <button
                      onClick={() => askAgain(bm.question)}
                      className="flex items-center justify-center gap-1 px-2 py-1 hover:bg-amber-500/10 text-muted-foreground hover:text-amber-400 rounded-md text-[10px] font-bold transition-all"
                      title="Скопировать вопрос в чат"
                    >
                      <RotateCcw size={11} />
                    </button>
                    <button
                      onClick={() => setEditingBm({ id: bm.id, title: bm.title || '', tags: (bm.tags || []).join(', ') })}
                      className="flex items-center justify-center gap-1 px-2 py-1 hover:bg-muted text-muted-foreground hover:text-foreground rounded-md text-[10px] font-bold transition-all"
                      title="Редактировать заголовок и теги"
                    >
                      <Pencil size={11} />
                    </button>
                    <button
                      onClick={() => deleteBookmark(bm.id)}
                      className="flex items-center justify-center gap-1 px-2 py-1 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded-md text-[10px] font-bold transition-all"
                      title="Удалить"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <div className="mt-auto flex flex-col border-t bg-muted/5">
        <div className="px-4 py-2 flex items-center justify-between border-b border-border/40">
          <div className="flex items-center gap-2">
            <div className={cn(
              "w-2 h-2 rounded-full",
              llamaCount > 0 ? "bg-green-500 animate-pulse" : "bg-muted-foreground/30"
            )} />
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-tight">
              Llama-CPP: {llamaCount} {llamaCount === 1 ? 'активен' : llamaCount > 1 ? 'активно' : 'спит'}
            </span>
          </div>
          {llamaCount > 0 && (
            <button 
              onClick={killAllLlama}
              className="p-1.5 hover:bg-destructive/10 text-destructive rounded-lg transition-all"
              title="Убить все процессы llama-server"
            >
              <PowerOff size={14} />
            </button>
          )}
        </div>

        <div className="p-4 flex items-center justify-between">
          <div className="flex flex-col">
             <span className="text-[8px] text-muted-foreground/40 uppercase tracking-widest font-bold">Хранилище</span>
             <span className="text-[10px] font-mono text-muted-foreground/60">{notebook.id}</span>
          </div>
          <button 
            onClick={onExit}
            className="flex items-center gap-2 px-3 py-1.5 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded-lg transition-all text-[11px] font-bold"
          >
            <LogOut size={14} />
            Выйти
          </button>
        </div>
      </div>

      {viewingBm && (
          <div
            onClick={() => setViewingBm(null)}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-fadeIn"
          >
            <div
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-2xl max-h-[85vh] flex flex-col bg-card border border-border shadow-2xl rounded-3xl overflow-hidden animate-scaleIn"
            >
              <div className="flex items-start justify-between p-5 border-b border-border/50 gap-3">
                <div className="min-w-0 flex-1">
                  {viewingBm.title && (
                    <h3 className="text-lg font-bold mb-1 break-words">{viewingBm.title}</h3>
                  )}
                  <p className="text-xs text-muted-foreground line-clamp-2">{viewingBm.question}</p>
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground/70 mt-1.5">
                    {viewingBm.model && <span className="truncate max-w-[180px]">{modelShort(viewingBm.model)}</span>}
                    <span>·</span>
                    <span>{formatRel(viewingBm.created_at)}</span>
                    {viewingBm.status === 'stale' && (
                      <>
                        <span>·</span>
                        <span className="text-amber-400 flex items-center gap-1">
                          <AlertCircle size={10} /> источник удалён
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setViewingBm(null)}
                  className="p-2 hover:bg-muted rounded-full transition-colors shrink-0"
                >
                  <X size={18} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-5 space-y-4">
                <div>
                  <div className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1.5">Вопрос</div>
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">{viewingBm.question}</p>
                </div>
                <div>
                  <div className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1.5">Ответ</div>
                  <div ref={bmAnswerRef}>
                    <LlmMarkdown
                      text={viewingBm.answer}
                      sources={viewingBm.sources || []}
                      onCite={(n, src) => {
                        if (src && onOpenFile) {
                          onOpenFile(src);
                        }
                      }}
                    />
                  </div>
                </div>
                {viewingBm.sources && viewingBm.sources.length > 0 && (
                  <div>
                    <div className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1.5">
                      Источники ({viewingBm.sources.length})
                    </div>
                    <div className="space-y-1">
                      {viewingBm.sources.map((s, i) => (
                        <button
                          key={i}
                          type="button"
                          disabled={!onOpenFile || bmSourceStale(s.file_name)}
                          onClick={() => { if (onOpenFile) onOpenFile(s); }}
                          className={cn(
                            "w-full text-left text-xs text-muted-foreground flex items-start gap-2 p-2 rounded-lg bg-muted/30 hover:bg-primary/10 hover:border-primary/30 border border-transparent transition-all group",
                            (!onOpenFile || bmSourceStale(s.file_name)) && "opacity-50 cursor-not-allowed hover:bg-muted/30 hover:border-transparent"
                          )}
                          title={bmSourceStale(s.file_name) ? 'Файл удалён из блокнота' : `Открыть «${s.file_name}»`}
                        >
                          <span className="text-primary font-bold shrink-0">[{i + 1}]</span>
                          <div className="min-w-0 flex-1">
                            <span className="font-medium text-foreground/90 group-hover:text-primary transition-colors">{s.file_name}</span>
                            {s.page != null && <span className="opacity-70"> · стр. {s.page}</span>}
                            {s.time != null && <span className="opacity-70"> · {s.time}</span>}
                            {s.snippet && (
                              <p className="text-[10px] opacity-70 mt-0.5 line-clamp-2">{s.snippet}</p>
                            )}
                          </div>
                          <ExternalLink size={11} className="shrink-0 mt-0.5 opacity-30 group-hover:opacity-100 transition-opacity" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {viewingBm.tags && viewingBm.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {viewingBm.tags.map(t => (
                      <span key={t} className="px-2 py-0.5 rounded-md bg-muted text-[10px] font-medium text-muted-foreground">
                        #{t}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2 p-4 border-t border-border/50 bg-muted/10">
                <button
                  onClick={() => askAgain(viewingBm.question)}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 rounded-xl text-xs font-black transition-all"
                >
                  <RotateCcw size={13} />
                  Спросить заново
                </button>
                <button
                  onClick={() => copyText(viewingBm.answer)}
                  className={cn(
                    "flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-black transition-all",
                    bmCopied
                      ? "bg-emerald-500/10 text-emerald-400"
                      : "hover:bg-muted text-muted-foreground hover:text-foreground"
                  )}
                  title="Скопировать ответ"
                >
                  {bmCopied ? <Check size={13} /> : <Copy size={13} />}
                </button>
                <button
                  onClick={() => {
                    deleteBookmark(viewingBm.id);
                    setViewingBm(null);
                  }}
                  className="flex items-center justify-center gap-2 px-3 py-2 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded-xl text-xs font-black transition-all"
                  title="Удалить"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          </div>
        )}

      {editingBm && (
          <div
            onClick={() => setEditingBm(null)}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-fadeIn"
          >
            <div
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md bg-card border border-border shadow-2xl rounded-2xl p-5 animate-scaleIn"
            >
              <h3 className="text-base font-bold mb-4">Редактировать закладку</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                    Заголовок
                  </label>
                  <input
                    value={editingBm.title}
                    onChange={(e) => setEditingBm({ ...editingBm, title: e.target.value })}
                    placeholder="Опционально"
                    className="w-full mt-1 bg-muted/30 border border-border/50 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                    Теги (через запятую)
                  </label>
                  <input
                    value={editingBm.tags}
                    onChange={(e) => setEditingBm({ ...editingBm, tags: e.target.value })}
                    placeholder="важно, про X, тест"
                    className="w-full mt-1 bg-muted/30 border border-border/50 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
              </div>
              <div className="flex gap-2 mt-5">
                <button
                  onClick={() => setEditingBm(null)}
                  className="flex-1 px-3 py-2 hover:bg-muted text-muted-foreground rounded-lg text-xs font-bold transition-all"
                >
                  Отмена
                </button>
                <button
                  onClick={saveBookmarkEdit}
                  className="flex-1 px-3 py-2 bg-primary text-primary-foreground rounded-lg text-xs font-bold transition-all hover:opacity-90"
                >
                  Сохранить
                </button>
              </div>
            </div>
          </div>
        )}
    </div>
  );
}
