import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './Sidebar';
import ChatArea from './ChatArea';
import DocumentViewer from './DocumentViewer';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import { cn } from '../lib/utils';

export default function MainApp({ notebook, onExit }) {
  const [sources, setSources] = useState([]);
  const [selectedSources, setSelectedSources] = useState([]);
  const [viewerFile, setViewerFile] = useState(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [viewerWidth, setViewerWidth] = useState(600);
  const [isResizing, setIsResizing] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [chatMaxWidth, setChatMaxWidth] = useState(() => {
    const saved = localStorage.getItem('chat_max_width');
    return saved ? parseInt(saved) : 1400;
  });
  // F-fix #29: для viewer-resizer нужны snapshot-значения (startX/startWidth),
  // поэтому хук useDragWidth не подходит. Делаем cleanup вручную через ref.
  const viewerDragCleanupRef = useRef(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [uploadState, setUploadState] = useState({
    isUploading: false,
    progress: 0,
    batchProgress: 0,
    currentFile: 0,
    totalFiles: 0,
    status: ''
  });
  const [llmSettings, setLlmSettings] = useState(() => {
		const saved = localStorage.getItem('llm_settings');
		// F-fix #27: API-ключ читаем из sessionStorage (а не localStorage).
		// Если ключа нет (новая вкладка / после закрытия) — fallback 'lm-studio'.
		let apiKey = 'lm-studio';
		try { apiKey = sessionStorage.getItem('llm_api_key') || apiKey; } catch {}
		return saved ? { ...JSON.parse(saved), llm_api_key: apiKey } : {
			llm_url: 'http://localhost:1234/v1',
			llm_api_key: apiKey,
			llm_model: 'gpt-4o',
			use_gguf: '',
			gguf_model_path: '',
			gguf_mmproj_path: '',
			gguf_ctx_size: 32768,
			mtp_enabled: false,
			gguf_batch_size: 512,
			gguf_ubatch_size: 256,
			vision_mtp_enabled: false,
			vision_batch_size: 512,
			vision_ubatch_size: 256,
		};
	});
  const cancelRef = useRef(false);

  const cancelUpload = async () => {
    cancelRef.current = true;
    try {
      await fetch(`/api/upload/cancel?notebook_id=${encodeURIComponent(notebook.id)}`, { method: 'POST' });
    } catch (e) {
      console.error('[UPLOAD] Cancel request failed', e);
    }
  };

  const handleUpload = async (filesToUpload) => {
    const files = Array.from(filesToUpload);
    if (!files.length) return;

    cancelRef.current = false;
    setUploadState(prev => ({ ...prev, isUploading: true, totalFiles: files.length, currentFile: 1, batchProgress: 0 }));

    let fileIdx = 1;
    const totalFiles = files.length;

    for (const file of files) {
      if (cancelRef.current) {
        console.log(`[UPLOAD] Cancelled before file ${fileIdx}/${totalFiles} (${file.name})`);
        break;
      }
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadUrl = new URL(`/api/upload`, window.location.origin);
      uploadUrl.searchParams.append('notebook_id', notebook.id);
      uploadUrl.searchParams.append('current_idx', fileIdx.toString());
      uploadUrl.searchParams.append('total_count', totalFiles.toString());

      if (llmSettings) {
        Object.entries(llmSettings).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            uploadUrl.searchParams.append(key, value.toString());
          }
        });
      }

      try {
        console.log(`[UPLOAD] Starting file ${fileIdx}/${totalFiles}: ${file.name}`);
        const response = await fetch(uploadUrl.toString(), {
          method: 'POST',
          body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            console.log(`[UPLOAD] Stream closed naturally for ${file.name}`);
            break;
          }
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          let shouldBreak = false;
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'progress') {
                  setUploadState(prev => ({ ...prev, progress: data.pct, status: data.msg }));
                } else if (data.type === 'done') {
                  console.log(`[UPLOAD] Received 'done' event for ${file.name}. Breaking stream.`);
                  setUploadState(prev => ({ ...prev, status: `Готово: ${data.filename}` }));
                  shouldBreak = true;
                } else if (data.type === 'cancelled') {
                  console.log(`[UPLOAD] Загрузка ${data.filename} отменена пользователем.`);
                  setUploadState({ isUploading: false, progress: 0, batchProgress: 0, currentFile: 0, totalFiles: 0, status: 'Отменено' });
                  shouldBreak = true;
                } else if (data.type === 'error') {
                  console.error(`[UPLOAD] Server error: ${data.msg}`);
                  setUploadState(prev => ({ ...prev, status: `Ошибка: ${data.msg}` }));
                }
              } catch (e) {
                console.warn("[UPLOAD] JSON parse error", e);
              }
            }
          }
          if (shouldBreak) break;
        }
        
        const newBatchProgress = (fileIdx / totalFiles) * 100;
        console.log(`[UPLOAD] Finished file ${fileIdx}/${totalFiles}. Progress: ${newBatchProgress}%`);
        fileIdx++;
        
        setUploadState(prev => ({ 
          ...prev, 
          batchProgress: newBatchProgress, 
          currentFile: Math.min(fileIdx, totalFiles) 
        }));
        
        try {
          await fetchSources();
        } catch (e) {
          console.error("[UPLOAD] Error refreshing sources:", e);
        }

        // Небольшая пауза перед следующим файлом для стабильности
        await new Promise(r => setTimeout(r, 500));
      } catch (err) {
        console.error(`[UPLOAD] Error uploading ${file.name}:`, err);
      }
    }
    setUploadState({ isUploading: false, progress: 0, batchProgress: 0, currentFile: 0, totalFiles: 0, status: '' });
    fetchSources();
  };

  useEffect(() => {
    // F-fix #29: при unmount снять viewer-resizer listeners если drag в процессе
    return () => {
      if (viewerDragCleanupRef.current) {
        try { viewerDragCleanupRef.current(); } catch { /* ignore */ }
        viewerDragCleanupRef.current = null;
      }
      if (sidebarDragCleanupRef.current) {
        try { sidebarDragCleanupRef.current(); } catch { /* ignore */ }
        sidebarDragCleanupRef.current = null;
      }
      if (chatResizeCleanupRef.current) {
        try { chatResizeCleanupRef.current(); } catch { /* ignore */ }
        chatResizeCleanupRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    fetchSources();
    let timerId;
    const checkStatus = async () => {
      try {
        const res = await fetch(`/api/ingestion_status?notebook_id=${notebook.id}`);
        if (res.ok) {
          const data = await res.json();
          setUploadState(current => {
            if (data.is_uploading) {
              return {
                isUploading: true,
                progress: data.progress,
                batchProgress: data.batch_progress,
                currentFile: data.current_file,
                totalFiles: data.total_files,
                status: data.status
              };
            } else if (current.isUploading) {
              fetchSources();
              return { isUploading: false, progress: 0, batchProgress: 0, currentFile: 0, totalFiles: 0, status: '' };
            }
            return current;
          });
        }
      } catch (e) {
        console.warn("[STATUS] Polling error:", e);
      } finally {
        timerId = setTimeout(checkStatus, 3000);
      }
    };

    checkStatus();
    return () => clearTimeout(timerId);
  }, [notebook.id]);

  const fetchSources = async () => {
    try {
      const res = await fetch(`/api/files?notebook_id=${notebook.id}`);
      const data = await res.json();
      const files = data.files || [];
      setSources(files);
      // Сохраняем выбор пользователя: автоматически выбираем только НОВЫЕ файлы, не сбрасывая текущий выбор
      setSelectedSources(prev => {
        if (prev.length === 0) return files; // Первая загрузка — выбираем все
        const newFiles = files.filter(f => !prev.includes(f) && !sources.includes(f));
        const kept = prev.filter(f => files.includes(f)); // Убираем удалённые файлы из выбора
        return [...kept, ...newFiles];
      });
    } catch (err) {
      console.error(err);
    }
  };

  const openViewer = (file) => {
    setViewerFile(file);
    setIsViewerOpen(true);
  };

  const mainRef = useRef(null);
  const sidebarDragCleanupRef = useRef(null);
  const chatResizeCleanupRef = useRef(null);

  const onSidebarMouseDown = (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = sidebarWidth;
    const onMouseMove = (ev) => {
      const w = ev.clientX;
      if (w > 240 && w < sidebarMax) {
        setSidebarWidth(w);
      }
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      sidebarDragCleanupRef.current = null;
    };
    sidebarDragCleanupRef.current = onMouseUp;
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };

  useEffect(() => {
    localStorage.setItem('chat_max_width', chatMaxWidth);
  }, [chatMaxWidth]);

  const chatAvailWidth = isSidebarOpen
    ? window.innerWidth - sidebarWidth * 2
    : window.innerWidth;
  const chatRenderWidth = Math.max(765, Math.min(chatMaxWidth, chatAvailWidth));

  const sidebarMax = Math.floor((window.innerWidth - chatRenderWidth) / 2);

  const handleExit = () => {
    setIsViewerOpen(false);
    setViewerFile(null);
    onExit();
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background relative">
      {/* Боковая панель с фиксированным позиционированием */}
      <AnimatePresence initial={false}>
        {isSidebarOpen && (
          <motion.div
            initial={{ x: '-100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '-100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed left-0 top-0 bottom-0 border-r z-[100] bg-background shadow-2xl flex-shrink-0"
            style={{ width: sidebarWidth }}
          >
            <Sidebar 
              notebook={notebook}
              sources={sources}
              selectedSources={selectedSources}
              onSelectSources={setSelectedSources}
              onRefresh={fetchSources}
              onExit={handleExit}
              onOpenFile={openViewer}
              llmSettings={llmSettings}
              width={sidebarWidth}
              onToggle={() => setIsSidebarOpen(false)}
              uploadState={uploadState}
              onUpload={handleUpload}
            />

            {/* Ресайзер боковой панели (F-fix #29: см. useDragWidth cleanup) */}
            <div
              className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize z-50 group/s-resizer hover:bg-primary/30 transition-colors"
              onMouseDown={onSidebarMouseDown}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Кнопка переключения скрытой панели */}
      {!isSidebarOpen && (
        <motion.button 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => setIsSidebarOpen(true)}
          className="fixed top-4 left-4 z-[100] w-10 h-10 bg-card border border-border rounded-xl flex items-center justify-center shadow-lg hover:bg-muted transition-all text-muted-foreground"
        >
          <ChevronRight size={20} />
        </motion.button>
      )}

      {/* Основная зона чата */}
      <main
        ref={mainRef}
        className="flex-1 flex flex-col min-w-0 bg-background relative z-0"
      >
        <div
          className="flex-1 flex flex-col mx-auto w-full relative"
          style={{ maxWidth: chatRenderWidth }}
        >
          {/* Ресайзер на правом краю чата */}
          <div
            className="absolute right-0 top-0 bottom-0 w-4 cursor-col-resize z-[99] group/c-resizer flex items-center justify-center"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startRendered = chatRenderWidth;
              const onMouseMove = (ev) => {
                const newVal = Math.max(765, startRendered + (ev.clientX - startX));
                setChatMaxWidth(newVal);
              };
              const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                chatResizeCleanupRef.current = null;
              };
              chatResizeCleanupRef.current = onMouseUp;
              document.addEventListener('mousemove', onMouseMove);
              document.addEventListener('mouseup', onMouseUp);
            }}
          >
            <div className="w-[3px] h-12 rounded-full bg-muted-foreground/20 group-hover/c-resizer:bg-primary/40 transition-colors" />
          </div>

          <ChatArea 
            notebook={notebook}
            selectedSources={selectedSources}
            llmSettings={llmSettings}
            setLlmSettings={setLlmSettings}
            onOpenSource={(src) => {
               setViewerFile(src);
               setIsViewerOpen(true);
            }}
          />
        </div>
      </main>

      {/* Оверлей просмотра документа */}
      <AnimatePresence>
        {isViewerOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsViewerOpen(false)}
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
            />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              style={{ width: viewerWidth }}
              className="fixed right-0 top-0 bottom-0 glass z-50 border-l flex flex-col shadow-2xl"
            >
              {/* Глобальный невидимый оверлей при ресайзе */}
              {isResizing && <div className="fixed inset-0 z-[100] cursor-col-resize" />}

              {/* Ресайзер просмотрщика */}
              <div
                className="absolute left-0 top-0 bottom-0 w-4 -left-2 cursor-col-resize z-[60] group/resizer"
                onMouseDown={(e) => {
                  e.preventDefault();
                  setIsResizing(true);
                  const startX = e.clientX;
                  const startWidth = viewerWidth;
                  const onMouseMove = (e) => {
                    const newWidth = startWidth + (startX - e.clientX);
                    if (newWidth > 350 && newWidth < window.innerWidth * 0.9) {
                      setViewerWidth(newWidth);
                    }
                  };
                  const onMouseUp = () => {
                    setIsResizing(false);
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    viewerDragCleanupRef.current = null;
                  };
                  // F-fix #29: сохраняем cleanup для unmount-safety
                  viewerDragCleanupRef.current = onMouseUp;
                  document.addEventListener('mousemove', onMouseMove);
                  document.addEventListener('mouseup', onMouseUp);
                }}
              >
                <div className="w-[1.5px] h-full bg-border group-hover/resizer:bg-primary/50 mx-auto transition-colors shadow-[0_0_5px_rgba(var(--primary),0.2)]" />
              </div>

              <div className={cn("flex-1 flex flex-col min-h-0 overflow-hidden", isResizing && "pointer-events-none select-none")}>
                <DocumentViewer 
                  file={viewerFile} 
                  notebook={notebook}
                  onClose={() => setIsViewerOpen(false)} 
                />
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Глобальная плавающая карточка загрузки */}
      <AnimatePresence>
        {uploadState.isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            className="fixed bottom-6 right-6 z-[1000] w-72 glass border border-primary/20 rounded-3xl p-5 shadow-[0_20px_50px_rgba(0,0,0,0.3)] overflow-hidden"
          >
            {/* Фоновое свечение */}
            <div className="absolute -top-10 -right-10 w-24 h-24 bg-primary/20 blur-[40px] rounded-full pointer-events-none" />
            
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                   <div className="w-2 h-2 bg-primary rounded-full animate-pulse shadow-[0_0_8px_rgba(var(--primary),0.5)]" />
                   <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Загрузка</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                    {uploadState.currentFile} из {uploadState.totalFiles}
                  </span>
                  <button
                    onClick={cancelUpload}
                    title="Остановить загрузку"
                    className="w-5 h-5 flex items-center justify-center text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded-full transition-colors"
                  >
                    <span className="text-base leading-none">×</span>
                  </button>
                </div>
              </div>

              {/* Полосы прогресса */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-[9px] font-bold text-foreground/60 uppercase">Файл</span>
                    <span className="text-[9px] font-bold text-primary">{Math.round(uploadState.progress)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-primary/10 rounded-full overflow-hidden shadow-inner">
                    <motion.div 
                      className="bg-primary h-full shadow-[0_0_10px_rgba(var(--primary),0.3)]"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadState.progress}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-[9px] font-bold text-foreground/60 uppercase">Всего</span>
                    <span className="text-[9px] font-bold text-muted-foreground">{Math.round(uploadState.batchProgress)}%</span>
                  </div>
                  <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                    <motion.div 
                      className="bg-muted-foreground/40 h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadState.batchProgress}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-primary/10">
                <p className="text-[9px] font-bold text-muted-foreground text-center truncate italic">
                  {uploadState.status}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
