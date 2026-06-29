// Просмотрщик документов: PDF, изображения, видео, аудио, текст.
import React, { useState, useEffect, useRef, useMemo, useCallback, startTransition } from 'react';
import { X, FileText, Play, Image as ImageIcon, Clock, AlertCircle, Download, ChevronDown } from 'lucide-react';
import { marked } from 'marked';
import { useVirtualizer } from '@tanstack/react-virtual';
import { cn } from '../lib/utils';

const FrameCard = React.memo(({ frame, notebookId }) => {
  const [imgLoaded, setImgLoaded] = useState(false);
  const imgSrc = `/files/${notebookId}/images/${frame.image_path.split(/[\\/]/).pop()}`;
  return (
    <div className="bg-card border border-border/50 rounded-2xl overflow-hidden shadow-md animate-fadeInUp" style={{ contentVisibility: 'auto', containIntrinsicSize: '1px 400px' }}>
      <div className="p-3 border-b bg-muted/20 flex items-center justify-between">
        <span className="text-[10px] font-black uppercase tracking-widest text-primary">Стр {frame.page}</span>
        <ImageIcon size={14} className="text-muted-foreground" />
      </div>
      <div className="p-4 flex flex-col md:flex-row gap-6">
        <div className="w-full md:w-1/2 rounded-xl overflow-hidden border border-border/40 bg-black relative">
          {!imgLoaded && <div className="absolute inset-0 skeleton-shimmer" style={{ aspectRatio: '3/4' }} />}
          <img 
            src={imgSrc}
            loading="lazy"
            className={`w-full h-auto object-contain cursor-zoom-in hover:scale-105 transition-transform duration-500 ${imgLoaded ? 'opacity-100' : 'opacity-0'}`}
            onLoad={() => setImgLoaded(true)}
            onClick={() => window.open(imgSrc, '_blank')}
          />
        </div>
        <div className="w-full md:w-1/2 space-y-3">
          <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Описание</h4>
          <div className="text-xs leading-relaxed text-foreground/80 font-medium md-content max-w-none" dangerouslySetInnerHTML={{ __html: marked.parse(frame.description || '', { gfm: true, breaks: true }) }} />
        </div>
      </div>
    </div>
  );
});
FrameCard.displayName = 'FrameCard';

const TextSection = React.memo(({ text }) => {
  const html = useMemo(() => marked.parse(text, { gfm: true, breaks: true }), [text]);
  if (!text) return null;
  return <div className="md-content max-w-none" dangerouslySetInnerHTML={{ __html: html }} />;
});
TextSection.displayName = 'TextSection';

const PdfPageCard = React.memo(({ frame, notebookId }) => {
  const [loaded, setLoaded] = useState(false);
  const imgSrc = `/files/${notebookId}/images/${frame.image_path.split(/[\\/]/).pop()}`;
  return (
    <div className="mx-auto max-w-5xl px-4 pb-6" style={{ contentVisibility: 'auto', containIntrinsicSize: '1px 800px' }}>
      <div className="p-1.5 text-center text-[10px] font-bold text-muted-foreground bg-muted/5 border-b border-border/10 rounded-t-lg">
        Стр. {frame.page}
      </div>
      <div className="relative bg-black rounded-b-lg overflow-hidden">
        {!loaded && <div className="skeleton-shimmer" style={{ width: '100%', paddingBottom: '130%' }} />}
        <img 
          src={imgSrc}
          loading="lazy"
          className={`w-full h-auto mx-auto ${loaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
          onLoad={() => setLoaded(true)}
          onClick={() => window.open(imgSrc, '_blank')}
        />
      </div>
    </div>
  );
});
PdfPageCard.displayName = 'PdfPageCard';

export default function DocumentViewer({ file, notebook, onClose }) {
  const [content, setContent] = useState(null);
  const [contentLoading, setContentLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [videoMeta, setVideoMeta] = useState(null);
  const [pptxData, setPptxData] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [showRawText, setShowRawText] = useState(false);
  const [showImagesOnly, setShowImagesOnly] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const vidRef = useRef(null);
  const feedRef = useRef(null);
  const itemRefs = useRef({});
  const lastSoughtTime = useRef(null);
  const textScrollRef = useRef(null);
  const imageScrollRef = useRef(null);
  const pdfImageScrollRef = useRef(null);

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
    ? `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(fileUrl)}&textlayer=off&disablefontface=true${page ? `#page=${page}` : ''}`
    : fileUrl;

  useEffect(() => {
    if (!filename) return;

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
      setContentLoading(true);
      fetch(`/api/source_content?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}`)
        .then(r => r.json())
        .then(data => {
          setContent(data.text);
          setContentLoading(false);
          if (!isSpecial) setLoading(false);
        })
        .catch(() => { setContentLoading(false); if (!isSpecial) setLoading(false); });
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

  // ── Предзагрузка первого изображения фрейма ──
  useEffect(() => {
    const links = [];
    if (videoMeta?.frames?.length > 0 && !showImagesOnly) {
      const firstImg = videoMeta.frames[0];
      const imgUrl = `/files/${notebook.id}/images/${firstImg.image_path.split(/[\\/]/).pop()}`;
      const link = document.createElement('link');
      link.rel = 'preload';
      link.as = 'image';
      link.href = imgUrl;
      document.head.appendChild(link);
      links.push(link);
    }
    return () => links.forEach(l => document.head.removeChild(l));
  }, [videoMeta, notebook.id]);

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

  // ── Разбиение контента на секции для виртуального скролла ──
  const sections = useMemo(() => {
    if (!content) return [];
    const parts = content.split(/(?=^#{1,3}\s)/m);
    const result = [];
    for (let part of parts) {
      part = part.trim();
      if (!part) continue;
      if (part.length > 4000) {
        const paras = part.split(/\n\s*\n/);
        let buf = '', cnt = 0;
        for (const para of paras) {
          const p = para.trim();
          if (!p) continue;
          if (buf && (cnt > 25 || (buf + '\n\n' + p).length > 4000)) {
            result.push(buf);
            buf = p; cnt = p.split('\n').length;
          } else {
            buf = buf ? buf + '\n\n' + p : p;
            cnt += p.split('\n').length;
          }
        }
        if (buf) result.push(buf);
      } else {
        result.push(part);
      }
    }
    return result;
  }, [content]);
  
  const rowVirtualizer = useVirtualizer({
    count: sections.length,
    getScrollElement: () => textScrollRef.current,
    estimateSize: (i) => Math.max(60, Math.ceil((sections[i]?.length || 100) / 80) * 22),
    overscan: 3,
  });
  
  
  const measureTextRow = useCallback((el) => {
    if (el) rowVirtualizer.measureElement(el);
  }, [rowVirtualizer]);
  
  const imageVirtualizer = useVirtualizer({
    count: videoMeta?.frames?.length || 0,
    getScrollElement: () => imageScrollRef.current,
    estimateSize: () => 350,
    overscan: 2,
  });
  
  const measureImageRow = useCallback((el) => {
    if (el) imageVirtualizer.measureElement(el);
  }, [imageVirtualizer]);
  
  const pdfImageVirtualizer = useVirtualizer({
    count: videoMeta?.frames?.length || 0,
    getScrollElement: () => pdfImageScrollRef.current,
    estimateSize: () => 900,
    overscan: 1,
  });
  
  const measurePdfImageRow = useCallback((el) => {
    if (el) pdfImageVirtualizer.measureElement(el);
  }, [pdfImageVirtualizer]);

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
                onClick={() => startTransition(() => { setShowRawText(false); setShowImagesOnly(false); })}
                className={cn(
                  "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all",
                  (!showRawText && !showImagesOnly) ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Документ
              </button>
              <button 
                onClick={() => startTransition(() => { setShowRawText(true); setShowImagesOnly(false); })}
                className={cn(
                  "px-3 py-1.5 text-[10px] font-bold rounded-lg transition-all",
                  showRawText ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Текст
              </button>
              {(isPdf || isPpt) && videoMeta?.frames?.length > 0 && (
                <button 
                onClick={() => startTransition(() => { setShowRawText(false); setShowImagesOnly(true); })}
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
              <div data-export-menu className="absolute right-0 top-full mt-1 bg-card border border-border/60 rounded-xl shadow-xl z-50 py-1 min-w-[160px] animate-scaleIn">
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

        {/* ── PDF: страницы как изображения (быстрее) или PDF.js fallback ── */}
        {!loading && isPdf && (
          videoMeta?.frames?.length > 0 ? (
            <div ref={pdfImageScrollRef} className="absolute inset-0 z-10 overflow-y-auto custom-scrollbar" style={{ display: showRawText || showImagesOnly ? 'none' : 'block' }}>
              <div style={{ height: `${pdfImageVirtualizer.getTotalSize()}px`, position: 'relative' }}>
                {pdfImageVirtualizer.getVirtualItems().map(virtualRow => {
                  const f = videoMeta.frames[virtualRow.index];
                  return (
                    <div key={virtualRow.key} data-index={virtualRow.index} ref={measurePdfImageRow}
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', transform: `translateY(${virtualRow.start}px)` }}>
                      <PdfPageCard frame={f} notebookId={notebook.id} />
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="absolute inset-0 z-10" style={{ display: showRawText || showImagesOnly ? 'none' : 'block' }}>
              <iframe 
                key={`${filename}__p${page ?? 'all'}`}
                src={viewerUrl}
                className="w-full h-full border-none"
              />
            </div>
          )
        )}
        {loading ? (
          <div className="h-full flex flex-col items-center justify-center gap-4 text-muted-foreground">
             <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin shadow-[0_0_15px_rgba(var(--primary),0.3)]" />
             <span className="text-[10px] font-bold uppercase tracking-tighter">Загрузка контента...</span>
          </div>
        ) : showImagesOnly ? (
          <div ref={imageScrollRef} className="h-full overflow-y-auto p-6 custom-scrollbar bg-muted/5" style={{ position: 'relative' }}>
            {(videoMeta?.frames?.length || 0) > 0 ? (
              <div style={{ height: `${imageVirtualizer.getTotalSize()}px`, position: 'relative' }}>
                {imageVirtualizer.getVirtualItems().map(virtualRow => {
                  const f = videoMeta.frames[virtualRow.index];
                  return (
                    <div
                      key={virtualRow.key}
                      data-index={virtualRow.index}
                      ref={measureImageRow}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        transform: `translateY(${virtualRow.start}px)`,
                        paddingBottom: '1.5rem',
                      }}
                    >
                      <FrameCard frame={f} notebookId={notebook.id} />
                    </div>
                  );
                })}
              </div>
            ) : null}
          </div>
        ) : showRawText ? (
          <div ref={textScrollRef} className="flex-1 p-8 overflow-y-auto custom-scrollbar min-h-0" style={{ position: 'relative' }}>
            {contentLoading ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Извлечение текста...</span>
              </div>
            ) : sections.length > 0 ? (
              <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: 'relative' }}>
                {rowVirtualizer.getVirtualItems().map(virtualRow => (
                  <div
                    key={virtualRow.key}
                    data-index={virtualRow.index}
                    ref={measureTextRow}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      transform: `translateY(${virtualRow.start}px)`,
                    }}
                  >
                    <TextSection text={sections[virtualRow.index]} />
                  </div>
                ))}
              </div>
            ) : <p className="text-muted-foreground text-sm">Текст пуст.</p>}
          </div>
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
                <div className="md-content max-w-none" dangerouslySetInnerHTML={{ __html: marked.parse(videoMeta.frames[0].description, { gfm: true, breaks: true }) }} />
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
                      <div 
                        key={i}
                        ref={el => itemRefs.current[i] = el}
                        className={cn(
                          "group p-4 rounded-2xl transition-all duration-300 cursor-pointer border animate-fadeIn",
                          isActive 
                            ? "bg-primary/10 border-primary/40 shadow-[0_8px_20px_rgba(var(--primary),0.1)] ring-1 ring-primary/20 scale-[1.02]" 
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
                        )} dangerouslySetInnerHTML={{ __html: marked.parse(ev.text || ev.description || '', { gfm: true, breaks: true }) }} />
                      </div>
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
            src={`/static/pdfjs/web/viewer.html?file=${encodeURIComponent(`/files/${notebook.id}/data/${pptxData.pdf_name}`)}&textlayer=off&disablefontface=true${page ? `#page=${page}` : ''}`}
            className="w-full h-full border-none"
          />
        ) : !isPdf && (
          <div className="h-full p-8 overflow-y-auto custom-scrollbar min-h-0">
            <div className="md-content max-w-none" dangerouslySetInnerHTML={{ __html: marked.parse(content || 'Контент пуст или не может быть отображен.', { gfm: true, breaks: true }) }} />
          </div>
        )}
      </div>
    </div>
  );
}
