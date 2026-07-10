// Основной layout: боковая панель + чат + просмотрщик документов.
// ── Импорты ──
import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import Sidebar from './Sidebar';
import ChatArea from './ChatArea';
const DocumentViewer = lazy(() => import('./DocumentViewer'));
import { ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';
// ── Состояние компонента ──
export default function MainApp({ notebook, onExit }) {
  const [sources, setSources] = useState([]);
  const [selectedSources, setSelectedSources] = useState([]);
  const [viewerFile, setViewerFile] = useState(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [viewerClosing, setViewerClosing] = useState(false);
  const [viewerWidth, setViewerWidth] = useState(600);
  const [isResizing, setIsResizing] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [chatMaxWidth, setChatMaxWidth] = useState(() => {
    const saved = localStorage.getItem('chat_max_width');
    return saved ? parseInt(saved) : 1400;
  });
  const viewerDragCleanupRef = useRef(null);
  const viewerPanelRef = useRef(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sidebarClosing, setSidebarClosing] = useState(false);
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
		let apiKey = 'lm-studio';
		try { apiKey = sessionStorage.getItem('llm_api_key') || apiKey; } catch {}
		return saved ? { ...JSON.parse(saved), llm_api_key: apiKey } : {
			llm_url: 'http://localhost:1234/v1',
			llm_api_key: apiKey,
			llm_model: 'gpt-4o',
			llm_ctx_size: 8192,
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

  // ── Обработчики загрузки файлов ──
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

        // Пауза 500мс перед следующим файлом для стабильности ingestion-пайплайна
        await new Promise(r => setTimeout(r, 500));
      } catch (err) {
        console.error(`[UPLOAD] Error uploading ${file.name}:`, err);
      }
    }
    setUploadState({ isUploading: false, progress: 0, batchProgress: 0, currentFile: 0, totalFiles: 0, status: '' });
    fetchSources();
  };

  // ── Эффекты жизненного цикла ──
  useEffect(() => {
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
    const controller = new AbortController();
    let timerId;
    const checkStatus = async () => {
      try {
        const res = await fetch(`/api/ingestion_status?notebook_id=${notebook.id}`, {
          signal: controller.signal
        });
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
        if (e.name !== 'AbortError') console.warn("[STATUS] Polling error:", e);
      } finally {
        if (!controller.signal.aborted) {
          timerId = setTimeout(checkStatus, 3000);
        }
      }
    };

    checkStatus();
    return () => { controller.abort(); clearTimeout(timerId); };
  }, [notebook.id]);

  // ── Управление источниками и панелями ──
  const fetchSources = async () => {
    try {
      const res = await fetch(`/api/files?notebook_id=${notebook.id}`);
      const data = await res.json();
      const files = data.files || [];
      setSources(files);
      // Сохраняем выбор: новые файлы добавляются автоматически, удалённые убираются из выделения
      setSelectedSources(prev => {
        if (prev.length === 0) return files;
        const newFiles = files.filter(f => !prev.includes(f) && !sources.includes(f));
        const kept = prev.filter(f => files.includes(f));
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
  const chatRenderWidth = Math.max(771, Math.min(chatMaxWidth, chatAvailWidth));

  const sidebarMax = Math.floor((window.innerWidth - chatRenderWidth) / 2);

  const handleExit = () => {
    setIsViewerOpen(false);
    setViewerClosing(false);
    setViewerFile(null);
    onExit();
  };

  const closeViewer = () => {
    if (!isViewerOpen || viewerClosing) return;
    setViewerClosing(true);
  };

  const onViewerCloseAnimationEnd = () => {
    setIsViewerOpen(false);
    setViewerClosing(false);
    setViewerFile(null);
  };

  const closeSidebar = () => {
    if (!isSidebarOpen || sidebarClosing) return;
    setSidebarClosing(true);
  };

  const onSidebarCloseAnimationEnd = () => {
    setIsSidebarOpen(false);
    setSidebarClosing(false);
  };

  // ── Рендер компонента ──
  return (
    <div className="flex h-screen w-full overflow-hidden bg-background relative">

      {(isSidebarOpen || sidebarClosing) && (
          <div
            onAnimationEnd={sidebarClosing ? onSidebarCloseAnimationEnd : undefined}
            className={`fixed left-0 top-0 bottom-0 border-r z-[100] bg-background shadow-2xl flex-shrink-0 ${sidebarClosing ? 'animate-slideOutLeft' : 'animate-slideInLeft'}`}
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
              width={sidebarWidth}
              onToggle={closeSidebar}
              uploadState={uploadState}
              onUpload={handleUpload}
            />


            <div
              className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize z-50 group/s-resizer hover:bg-primary/30 transition-colors"
              onMouseDown={onSidebarMouseDown}
            />
          </div>
        )}


      {!isSidebarOpen && (
        <button 
          onClick={() => setIsSidebarOpen(true)}
          className="fixed top-4 left-4 z-[100] w-10 h-10 bg-card border border-border rounded-xl flex items-center justify-center shadow-lg hover:bg-muted transition-all text-muted-foreground animate-fadeIn"
        >
          <ChevronRight size={20} />
        </button>
      )}


      <main
        ref={mainRef}
        className="flex-1 flex flex-col min-w-0 bg-background relative z-0"
      >
        <div
          className="flex-1 flex flex-col mx-auto w-full relative"
          style={{ maxWidth: chatRenderWidth }}
        >

          <div
            className="absolute right-0 top-0 bottom-0 w-4 cursor-col-resize z-[99] group/c-resizer flex items-center justify-center"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startRendered = chatRenderWidth;
              const onMouseMove = (ev) => {
                const newVal = Math.max(771, startRendered + (ev.clientX - startX));
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


      {(isViewerOpen || viewerClosing) && (
          <>
            <div 
              onClick={closeViewer}
              onAnimationEnd={viewerClosing ? onViewerCloseAnimationEnd : undefined}
              className={`fixed inset-0 bg-black/40 backdrop-blur-sm z-40 ${viewerClosing ? 'animate-fadeOut' : 'animate-fadeIn'}`}
            />
            <div
              ref={viewerPanelRef}
              style={{ width: viewerWidth }}
              className={`fixed right-0 top-0 bottom-0 glass z-50 border-l flex flex-col shadow-2xl ${viewerClosing ? 'animate-slideOutRight' : 'animate-slideInRight'}`}
            >

              {isResizing && <div className="fixed inset-0 z-[100] cursor-col-resize" />}

              <div
                className="absolute left-0 top-0 bottom-0 w-4 -left-2 cursor-col-resize z-[60] group/resizer"
                onMouseDown={(e) => {
                  e.preventDefault();
                  setIsResizing(true);
                  const startX = e.clientX;
                  const startWidth = viewerWidth;
                  const panel = viewerPanelRef.current;
                  // Прямое DOM-обновление ширины: без React re-render, без reflow iframe
                  const onMouseMove = (e) => {
                    const newWidth = startWidth + (startX - e.clientX);
                    if (newWidth > 350 && newWidth < window.innerWidth * 0.9) {
                      if (panel) panel.style.width = newWidth + 'px';
                    }
                  };
                  const onMouseUp = (e) => {
                    setIsResizing(false);
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    viewerDragCleanupRef.current = null;
                    // Коммит финальной ширины в React state только один раз
                    const finalWidth = startWidth + (startX - e.clientX);
                    if (finalWidth > 350 && finalWidth < window.innerWidth * 0.9) {
                      setViewerWidth(finalWidth);
                    }
                  };
                  viewerDragCleanupRef.current = onMouseUp;
                  document.addEventListener('mousemove', onMouseMove);
                  document.addEventListener('mouseup', onMouseUp);
                }}
              >
                <div className="w-[1.5px] h-full bg-border group-hover/resizer:bg-primary/50 mx-auto transition-colors shadow-[0_0_5px_rgba(var(--primary),0.2)]" />
              </div>

              <div className={cn("flex-1 flex flex-col min-h-0 overflow-hidden", isResizing && "pointer-events-none select-none")}
                   style={{ contain: 'layout' }}>
                <Suspense fallback={
                  <div className="h-full flex items-center justify-center text-muted-foreground">
                    <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  </div>
                }>
                  <DocumentViewer 
                    file={viewerFile} 
                    notebook={notebook}
                    onClose={closeViewer} 
                  />
                </Suspense>
              </div>
            </div>
          </>
        )}


      {uploadState.isUploading && (
          <div
            className="fixed bottom-6 right-6 z-[1000] w-72 glass border border-primary/20 rounded-3xl p-5 shadow-[0_20px_50px_rgba(0,0,0,0.3)] overflow-hidden animate-popIn"
          >

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


              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-[9px] font-bold text-foreground/60 uppercase">Файл</span>
                    <span className="text-[9px] font-bold text-primary">{Math.round(uploadState.progress)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-primary/10 rounded-full overflow-hidden shadow-inner">
                    <div 
                      className="bg-primary h-full shadow-[0_0_10px_rgba(var(--primary),0.3)] transition-[width] duration-300 ease-out"
                      style={{ width: `${uploadState.progress}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-[9px] font-bold text-foreground/60 uppercase">Всего</span>
                    <span className="text-[9px] font-bold text-muted-foreground">{Math.round(uploadState.batchProgress)}%</span>
                  </div>
                  <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="bg-muted-foreground/40 h-full transition-[width] duration-300 ease-out"
                      style={{ width: `${uploadState.batchProgress}%` }}
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
          </div>
        )}
    </div>
  );
}
