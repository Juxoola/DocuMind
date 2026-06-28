// Просмотрщик документов: PDF, изображения, видео, аудио, текст.
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import { X, FileText, Play, Image as ImageIcon, Clock, AlertCircle, Download, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '../lib/utils';

export default function DocumentViewer({ file, notebook, onClose }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pptxData, setPptxData] = useState(null);
  const [videoMeta, setVideoMeta] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [showRawText, setShowRawText] = useState(false);
  const [showImagesOnly, setShowImagesOnly] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const vidRef = useRef(null);
  const feedRef = useRef(null);
  const itemRefs = useRef({});
  const lastSoughtTime = useRef(null);

  // file — строка (имя) или объект {file_name, page, time} из чата
  const filename = typeof file === 'string' ? file : file?.file_name;
  const page = typeof file === 'object' ? file?.page : null;
  const startTime = typeof file === 'object' ? file?.time : null;
  
  const isPdf = filename?.toLowerCase().endsWith('.pdf');
  const isVideo = filename?.toLowerCase().match(/\.(mp4|avi|mov|mkv)$/i);
  const isAudio = filename?.toLowerCase().match(/\.(mp3|wav|m4a|aac|flac)$/i);
  const isPpt = filename?.toLowerCase().match(/\.(pptx|ppt)$/i);
  const isImage = filename?.toLowerCase().match(/\.(jpg|jpeg|png|webp|gif|bmp)$/i);
  const isMedia = isVideo || isAudio;
  const isSpecial = isPdf || isMedia || isPpt || isImage;
  
  let fileUrl = filename ? `/files/${notebook.id}/data/${encodeURIComponent(filename)}` : '';
  
  const viewerUrl = isPdf 
    ? `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(fileUrl)}${page ? `#page=${page}` : ''}`
    : fileUrl;

  useEffect(() => {
    if (!filename) return;

    // Для медиа и PDF — плеер/iframe рендерится мгновенно, метаданные грузятся в фоне
    if (isSpecial) {
      setLoading(false);
    } else {
      setLoading(true);
    }

    setVideoMeta(null);
    setPptxData(null);
    setContent(null);

    const metaUrl = `/api/video_metadata?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}`;
    if (isMedia || isPdf || isImage) {
      fetch(metaUrl)
        .then(r => r.json())
        .then(data => { if (!data.error) setVideoMeta(data); })
        .catch(() => {});
    } else if (isPpt) {
      fetch(metaUrl)
        .then(r => r.json())
        .then(data => { if (!data.error) setPptxData(data); })
        .catch(() => {});
    }

    if (!isMedia) {
      fetch(`/api/source_content?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}`)
        .then(r => r.json())
        .then(data => {
          setContent(data.text);
          if (!isSpecial) setLoading(false);
        })
        .catch(() => { if (!isSpecial) setLoading(false); });
    }
  }, [filename, notebook.id]);

  useEffect(() => {
    if (isMedia && startTime !== null && vidRef.current && lastSoughtTime.current !== file) {
      const vid = vidRef.current;
      const doSeek = () => {
        vid.currentTime = startTime;
        vid.play().catch(() => {});
        lastSoughtTime.current = file;
      };

      if (vid.readyState >= 1) {
        doSeek();
      } else {
        vid.addEventListener('canplay', doSeek, { once: true });
      }
    }
  }, [startTime, isMedia, file]);

  const seekMedia = (time) => {
    if (vidRef.current) {
      vidRef.current.currentTime = time;
      vidRef.current.play().catch(() => {});
    }
  };

  const fmtTime = (s) => {
    let m = Math.floor(s / 60);
    let sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const downloadText = (fmt = 'txt') => {
    setShowExportMenu(false);
    const stem = filename.replace(/\.[^.]+$/, '');
    const ext = fmt === 'md' ? 'md' : fmt;
    const url = `/api/export_text?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}&fmt=${fmt}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `${stem}.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const exportFormats = [
    { fmt: 'txt', label: 'Текст (.txt)', icon: '📄' },
    { fmt: 'md', label: 'Markdown (.md)', icon: '📝' },
    { fmt: 'pdf', label: 'PDF (.pdf)', icon: '📑' },
  ];

  // Закрытие дропдауна по клику снаружи
  useEffect(() => {
    if (!showExportMenu) return;
    const handler = (e) => {
      if (!e.target.closest('[data-export-menu]')) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showExportMenu]);

  // Объединяем транскрипт и кадры в единую хронологическую ленту для навигации
  const feedItems = useMemo(() => {
    if (!videoMeta) return [];
    const items = [
      ...(videoMeta.transcript || []).map(t => ({ ...t, type: 'text', time: t.start })),
      ...(videoMeta.frames || []).map(f => ({ ...f, type: 'image', time: f.time }))
    ];
    return items.sort((a, b) => a.time - b.time);
  }, [videoMeta]);

  const activeIndex = useMemo(() => {
    let bestIdx = -1;
    for (let i = 0; i < feedItems.length; i++) {
      if (feedItems[i].time <= currentTime + 0.5) {
        bestIdx = i;
      } else {
        break;
      }
    }
    return bestIdx;
  }, [feedItems, currentTime]);

  // Автопрокрутка к активному элементу при смене текущего времени воспроизведения
  useEffect(() => {
    if (isMedia && activeIndex !== -1 && itemRefs.current[activeIndex]) {
      itemRefs.current[activeIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
    if (isPpt && page && itemRefs.current[page]) {
      itemRefs.current[page].scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  }, [activeIndex, page, isPpt, isMedia]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-background">
      <div className="p-4 border-b flex items-center justify-between bg-card/50 backdrop-blur-md z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 text-primary rounded-xl shadow-inner">
            <FileText size={18} />
          </div>
          <div className="flex flex-col">
            <h3 className="text-sm font-bold truncate max-w-[200px] text-foreground">{filename}</h3>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-black opacity-70">Просмотр документа</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!isMedia && (
            <div className="flex bg-muted/30 p-1 rounded-xl border border-border/40">
              <button 
                onClick={() => { setShowRawText(false); setShowImagesOnly(false); }}
                className={cn(
                  "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all",
                  (!showRawText && !showImagesOnly) ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Документ
              </button>
              <button 
                onClick={() => { setShowRawText(true); setShowImagesOnly(false); }}
                className={cn(
                  "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all",
                  showRawText ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Текст
              </button>
              {(isPdf || isPpt) && videoMeta?.frames?.length > 0 && (
                <button 
                  onClick={() => { setShowRawText(false); setShowImagesOnly(true); }}
                  className={cn(
                    "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all",
                    showImagesOnly ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Картинки
                </button>
              )}
            </div>
          )}
          <button onClick={onClose} className="p-2 hover:bg-muted rounded-xl transition-all hover:rotate-90">
            <X size={18} />
          </button>
          <div className="relative">
            <button 
              onClick={() => setShowExportMenu(!showExportMenu)}
              title="Скачать текст"
              className="flex items-center gap-1 p-2 hover:bg-muted rounded-xl transition-all text-muted-foreground hover:text-foreground"
              data-export-menu
            >
              <Download size={18} />
              <ChevronDown size={12} />
            </button>
            {showExportMenu && (
              <div data-export-menu className="absolute right-0 top-full mt-1 bg-card border border-border/60 rounded-xl shadow-xl z-50 py-1 min-w-[160px]">
                {exportFormats.map(({ fmt, label, icon }) => (
                  <button
                    key={fmt}
                    onClick={() => downloadText(fmt)}
                    className="w-full px-3 py-2 text-left text-xs font-medium hover:bg-muted/60 flex items-center gap-2 transition-colors"
                  >
                    <span>{icon}</span>
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden relative flex flex-col">
        {loading ? (
          <div className="h-full flex flex-col items-center justify-center gap-4 text-muted-foreground">
             <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin shadow-[0_0_15px_rgba(var(--primary),0.3)]" />
             <span className="text-[10px] font-bold uppercase tracking-tighter">Загрузка контента...</span>
          </div>
        ) : showImagesOnly ? (
          <div className="h-full overflow-y-auto p-6 space-y-6 custom-scrollbar bg-muted/5">
            {videoMeta?.frames?.map((f, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card border border-border/50 rounded-2xl overflow-hidden shadow-md"
              >
                <div className="p-3 border-b bg-muted/20 flex items-center justify-between">
                  <span className="text-[10px] font-black uppercase tracking-widest text-primary">Стр {f.page}</span>
                  <ImageIcon size={14} className="text-muted-foreground" />
                </div>
                <div className="p-4 flex flex-col md:flex-row gap-6">
                  <div className="w-full md:w-1/2 rounded-xl overflow-hidden border border-border/40 bg-black">
                    <img 
                      src={`/files/${notebook.id}/images/${f.image_path.split(/[\\/]/).pop()}`} 
                      className="w-full h-auto object-contain cursor-zoom-in hover:scale-105 transition-transform duration-500" 
                      onClick={() => window.open(`/files/${notebook.id}/images/${f.image_path.split(/[\\/]/).pop()}`, '_blank')}
                    />
                  </div>
                  <div className="w-full md:w-1/2 space-y-3">
                    <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Описание</h4>
                    <div className="text-xs leading-relaxed text-foreground/80 font-medium md-content max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {f.description || ''}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : showRawText ? (
          <div className="flex-1 p-8 overflow-y-auto custom-scrollbar min-h-0">
            <div className="md-content max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content || 'Текст пуст или извлекается...'}
              </ReactMarkdown>
            </div>
          </div>
        ) : isPdf ? (
          <iframe 
            key={`${filename}__p${page ?? 'all'}`}
            src={viewerUrl}
            className="w-full h-full border-none"
          />
        ) : isImage ? (
          <div className="h-full overflow-y-auto">
            <div className="p-4">
              <img
                src={`/files/${notebook.id}/data/${encodeURIComponent(filename)}`}
                alt={filename}
                className="max-w-full h-auto rounded-lg shadow-lg cursor-zoom-in hover:scale-105 transition-transform duration-300"
                onClick={() => window.open(`/files/${notebook.id}/data/${encodeURIComponent(filename)}`, '_blank')}
              />
            </div>
            {videoMeta?.frames?.[0]?.description && (
              <div className="p-4 border-t border-border/30 bg-muted/20">
                <p className="text-xs leading-relaxed text-foreground/80 font-medium whitespace-pre-wrap">
                  {videoMeta.frames[0].description}
                </p>
              </div>
            )}
          </div>
        ) : isMedia ? (
          <div className="h-full flex flex-col overflow-hidden">
            <div className="relative p-6 pb-0 group">
              <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none opacity-50" />
              <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-full h-40 bg-primary/10 blur-[100px] rounded-full pointer-events-none" />
              
              <div className={cn(
                "relative z-10 rounded-2xl overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-white/10 bg-black",
                isVideo ? "aspect-video" : "py-8 px-6 flex flex-col items-center justify-center bg-gradient-to-br from-card to-muted/20"
              )}>
                {isVideo ? (
                  <video 
                    ref={vidRef}
                    src={fileUrl} 
                    controls 
                    onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full max-w-md flex flex-col items-center gap-6">
                    <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center text-primary animate-pulse shadow-[0_0_30px_rgba(var(--primary),0.2)]">
                      <Play size={32} fill="currentColor" />
                    </div>
                    <div className="text-center">
                       <p className="text-xs font-black uppercase tracking-widest text-primary mb-1">Воспроизведение аудио</p>
                       <p className="text-[10px] text-muted-foreground truncate max-w-[300px]">{filename}</p>
                    </div>
                    <audio 
                      ref={vidRef}
                      src={fileUrl} 
                      controls 
                      onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                      className="w-full"
                    />
                  </div>
                )}
              </div>
            </div>
            
            <div className="flex-1 overflow-hidden flex flex-col mt-6">
               <div className="px-6 mb-3 flex items-center justify-between">
                  <h4 className="text-[10px] font-black uppercase text-primary tracking-[0.2em]">Интеллектуальная лента</h4>
                  {videoMeta && (
                    <span className="text-[9px] font-bold text-muted-foreground bg-muted/50 px-2 py-0.5 rounded-full uppercase">
                      {feedItems.length} событий
                    </span>
                  )}
               </div>
               
               <div 
                 ref={feedRef}
                 className="flex-1 overflow-y-auto px-6 pb-12 space-y-3 custom-scrollbar"
               >
                {videoMeta ? (
                  feedItems.map((ev, i) => {
                    const isActive = activeIndex === i;
                    return (
                      <motion.div 
                        key={i}
                        ref={el => itemRefs.current[i] = el}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ 
                          opacity: 1, 
                          x: 0,
                          scale: isActive ? 1.02 : 1,
                        }}
                        className={cn(
                          "group p-4 rounded-2xl transition-all duration-300 cursor-pointer border",
                          isActive 
                            ? "bg-primary/10 border-primary/40 shadow-[0_8px_20px_rgba(var(--primary),0.1)] ring-1 ring-primary/20" 
                            : "bg-muted/30 border-border/40 hover:border-primary/30 hover:bg-muted/50"
                        )}
                        onClick={() => seekMedia(ev.time)}
                      >
                        <div className="flex items-center gap-3 mb-2">
                           <div className={cn(
                             "flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-black transition-colors",
                             isActive ? "bg-primary text-white" : "bg-primary/10 text-primary"
                           )}>
                             <Clock size={10} className={isActive ? "animate-pulse" : ""} />
                             {fmtTime(ev.time)}
                           </div>
                           {ev.type === 'image' && (
                             <div className="text-[9px] font-black text-accent uppercase flex items-center gap-1 opacity-80">
                               <ImageIcon size={10} /> Слайд
                             </div>
                           )}
                        </div>
                        
                        {ev.type === 'image' && (
                          <div className="mb-3 overflow-hidden rounded-xl border border-border/50 shadow-lg">
                             <img 
                                src={`/files/${notebook.id}/images/${ev.image_path.split(/[\\/]/).pop()}`} 
                                className="w-full h-auto group-hover:scale-105 transition-transform duration-700 ease-out"
                             />
                          </div>
                        )}
                        
                        <div className={cn(
                          "text-xs leading-relaxed transition-colors md-content max-w-none",
                          isActive ? "text-foreground font-medium" : "text-foreground/70"
                        )}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {ev.text || ev.description || ''}
                          </ReactMarkdown>
                        </div>
                      </motion.div>
                    );
                  })
                ) : (
                  <div className="h-full flex flex-col items-center justify-center py-20 opacity-30 text-center">
                    <AlertCircle size={32} className="mb-4" />
                    <p className="text-xs font-bold uppercase tracking-widest">Метаданные не найдены</p>
                  </div>
                )}
               </div>
            </div>
          </div>
        ) : isPpt && pptxData?.has_pdf ? (
          <iframe 
            key={`${filename}__p${page ?? 'all'}_ppt`}
            src={`/static/pdfjs/web/viewer.html?file=${encodeURIComponent(`/files/${notebook.id}/data/${pptxData.pdf_name}`)}${page ? `#page=${page}` : ''}`}
            className="w-full h-full border-none"
          />
        ) : isPpt ? (
          <div className="h-full overflow-y-auto p-6 space-y-8 custom-scrollbar bg-muted/10">
            {!pptxData ? (
              <div className="space-y-6">
                {[1,2,3].map(i => (
                  <div key={i} className="bg-card border border-border/30 rounded-3xl overflow-hidden animate-pulse">
                    <div className="bg-muted/40 h-12" />
                    <div className="p-8 space-y-4">
                      <div className="h-48 bg-muted/30 rounded-2xl" />
                      <div className="h-3 bg-muted/30 rounded-full w-3/4" />
                      <div className="h-3 bg-muted/30 rounded-full w-1/2" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              pptxData.slides?.map((slide, i) => (
                <motion.div 
                  key={i}
                  ref={el => itemRefs.current[slide.number] = el}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "bg-card border rounded-3xl overflow-hidden shadow-xl transition-all",
                    page === slide.number ? "ring-2 ring-primary border-primary/50" : "border-border/50"
                  )}
                >
                  <div className="bg-muted/30 p-4 border-b flex items-center justify-between">
                    <span className="text-[10px] font-black uppercase tracking-widest text-primary">Слайд {slide.number}</span>
                    {slide.title && <span className="text-xs font-bold truncate max-w-[70%]">{slide.title}</span>}
                  </div>
                  
                  <div className="p-8 space-y-6">
                    {slide.images && slide.images.length > 0 && (
                      <div className="grid grid-cols-1 gap-4">
                        {slide.images.map((img, imgIdx) => (
                          <div key={imgIdx} className="group relative rounded-2xl overflow-hidden border border-border shadow-md bg-black">
                             <img 
                              src={`/files/${notebook.id}/images/${img.path}`} 
                              className="w-full h-auto max-h-[400px] object-contain transition-transform duration-500 group-hover:scale-[1.02]"
                             />
                             {img.description && (
                               <div className="absolute bottom-0 left-0 right-0 p-3 bg-black/60 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity">
                                 <p className="text-[10px] text-white/80 leading-relaxed italic">{img.description}</p>
                               </div>
                             )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {slide.text && (
                      <div className="prose prose-invert prose-sm max-w-none">
                        <p className="text-sm leading-relaxed text-foreground/80 whitespace-pre-wrap font-medium">
                          {slide.text}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        ) : (
          <div className="h-full p-8 overflow-y-auto prose prose-invert prose-sm max-w-none">
            <pre className="whitespace-pre-wrap font-sans text-xs leading-loose text-foreground/80">
              {content || 'Контент пуст или не может быть отображен.'}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
