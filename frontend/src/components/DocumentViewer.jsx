import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, FileText, Play, Image as ImageIcon, ExternalLink, Clock, AlertCircle } from 'lucide-react';
import { cn } from '../lib/utils';

export default function DocumentViewer({ file, notebook, onClose }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [videoMeta, setVideoMeta] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const vidRef = useRef(null);
  const feedRef = useRef(null);
  const itemRefs = useRef({});
  const lastSoughtTime = useRef(null);

  // file может быть строкой (имя файла) или объектом (из чата)
  const filename = typeof file === 'string' ? file : file?.file_name;
  const page = typeof file === 'object' ? file?.page : null;
  const startTime = typeof file === 'object' ? file?.time : null;
  
  const isPdf = filename?.toLowerCase().endsWith('.pdf');
  const isVideo = filename?.toLowerCase().match(/\.(mp4|avi|mov|mkv)$/i);
  
  let fileUrl = filename ? `/files/${notebook.id}/data/${encodeURIComponent(filename)}` : '';
  
  // Для PDF добавляем навигацию по страницам
  const viewerUrl = isPdf 
    ? `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(fileUrl)}${page ? `#page=${page}` : ''}`
    : fileUrl;

  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    
    if (isVideo) {
      fetch(`/api/video_metadata?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}`)
        .then(r => r.json())
        .then(data => {
          if (!data.error) setVideoMeta(data);
          setLoading(false);
        });
    } else if (!isPdf) {
      fetch(`/api/source_content?filename=${encodeURIComponent(filename)}&notebook_id=${notebook.id}`)
        .then(r => r.json())
        .then(data => {
          setContent(data.text);
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, [filename, notebook.id]);

  // Обработка перехода по времени при клике из чата
  useEffect(() => {
    // Ищем только если startTime изменился и мы еще не перешли к нему для ЭТОГО объекта file
    if (isVideo && startTime !== null && vidRef.current && lastSoughtTime.current !== file) {
      vidRef.current.currentTime = startTime;
      vidRef.current.play().catch(() => {});
      lastSoughtTime.current = file; // Запоминаем именно объект, чтобы при его смене сработал переход
    }
  }, [startTime, isVideo, file]);

  const seekVideo = (time) => {
    if (vidRef.current) {
      vidRef.current.currentTime = time;
      vidRef.current.play();
    }
  };

  const fmtTime = (s) => {
    let m = Math.floor(s / 60);
    let sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  // Объединяем транскрипт и кадры в единую ленту
  const feedItems = useMemo(() => {
    if (!videoMeta) return [];
    const items = [
      ...(videoMeta.transcript || []).map(t => ({ ...t, type: 'text', time: t.start })),
      ...(videoMeta.frames || []).map(f => ({ ...f, type: 'image', time: f.time }))
    ];
    return items.sort((a, b) => a.time - b.time);
  }, [videoMeta]);

  // Определяем активный элемент
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

  // Автопрокрутка к активному элементу
  useEffect(() => {
    if (activeIndex !== -1 && itemRefs.current[activeIndex]) {
      itemRefs.current[activeIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [activeIndex]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-background">
      {/* Header */}
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
        <button onClick={onClose} className="p-2 hover:bg-muted rounded-xl transition-all hover:rotate-90">
          <X size={18} />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden relative">
        {loading ? (
          <div className="h-full flex flex-col items-center justify-center gap-4 text-muted-foreground">
             <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin shadow-[0_0_15px_rgba(var(--primary),0.3)]" />
             <span className="text-[10px] font-bold uppercase tracking-tighter">Загрузка контента...</span>
          </div>
        ) : isPdf ? (
          <iframe 
            src={viewerUrl}
            className="w-full h-full border-none"
          />
        ) : isVideo ? (
          <div className="h-full flex flex-col overflow-hidden">
            {/* Video Underlay & Player */}
            <div className="relative p-6 pb-0 group">
              {/* Эффект подложки (Glow) */}
              <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none opacity-50" />
              <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-full h-40 bg-primary/10 blur-[100px] rounded-full pointer-events-none" />
              
              <div className="relative z-10 rounded-2xl overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-white/10 bg-black aspect-video">
                <video 
                  ref={vidRef}
                  src={fileUrl} 
                  controls 
                  onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
            
            {/* Intellectual Feed */}
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
                        onClick={() => seekVideo(ev.time)}
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
                        
                        <p className={cn(
                          "text-xs leading-relaxed transition-colors",
                          isActive ? "text-foreground font-medium" : "text-foreground/70"
                        )}>
                          {ev.text || ev.description}
                        </p>
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
