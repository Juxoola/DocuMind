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
  const [viewerWidth, setViewerWidth] = useState(500);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [llmSettings, setLlmSettings] = useState(() => {
    const saved = localStorage.getItem('llm_settings');
    return saved ? JSON.parse(saved) : {
      llm_url: 'http://localhost:1234/v1',
      llm_api_key: 'lm-studio',
      llm_model: 'gpt-4o'
    };
  });

  useEffect(() => {
    fetchSources();
  }, [notebook.id]);

  const fetchSources = async () => {
    try {
      const res = await fetch(`/api/files?notebook_id=${notebook.id}`);
      const data = await res.json();
      const files = data.files || [];
      setSources(files);
      setSelectedSources(files);
    } catch (err) {
      console.error(err);
    }
  };

  const openViewer = (filename) => {
    setViewerFile(filename);
    setIsViewerOpen(true);
  };

  const handleExit = () => {
    setIsViewerOpen(false);
    setViewerFile(null);
    onExit();
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background relative">
      {/* Sidebar with fixed positioning to avoid squishing chat */}
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
            />

            {/* Sidebar Resizer */}
            <div 
              className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize z-50 group/s-resizer hover:bg-primary/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const onMouseMove = (e) => {
                  const newWidth = e.clientX;
                  // Ограничиваем, чтобы не налезать на чат (max-w-4xl = 896px)
                  const maxAllowed = (window.innerWidth - 896) / 2 - 20;
                  if (newWidth > 240 && newWidth < Math.max(240, Math.min(600, maxAllowed))) {
                    setSidebarWidth(newWidth);
                  }
                };
                const onMouseUp = () => {
                  document.removeEventListener('mousemove', onMouseMove);
                  document.removeEventListener('mouseup', onMouseUp);
                };
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button for Hidden Sidebar */}
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

      {/* Main Chat Area - Always centered relative to the window, Sidebar is now an overlay */}
      <main className="flex-1 flex flex-col min-w-0 bg-background relative z-0">
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
      </main>

      {/* Document Viewer Overlay */}
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
              {/* Viewer Resizer */}
              <div 
                className="absolute left-0 top-0 bottom-0 w-2 -left-1 cursor-col-resize z-[60] group/resizer"
                onMouseDown={(e) => {
                  const startX = e.clientX;
                  const startWidth = viewerWidth;
                  const onMouseMove = (e) => {
                    const newWidth = startWidth + (startX - e.clientX);
                    if (newWidth > 350 && newWidth < window.innerWidth * 0.8) {
                      setViewerWidth(newWidth);
                    }
                  };
                  const onMouseUp = () => {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                  };
                  document.addEventListener('mousemove', onMouseMove);
                  document.addEventListener('mouseup', onMouseUp);
                }}
              >
                <div className="w-[1px] h-full bg-border group-hover/resizer:bg-primary/50 mx-auto transition-colors" />
              </div>

              <DocumentViewer 
                file={viewerFile} 
                notebook={notebook}
                onClose={() => setIsViewerOpen(false)} 
              />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
