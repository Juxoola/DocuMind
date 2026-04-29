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

  useEffect(() => {
    fetchSources();
  }, [notebook.id]);

  const fetchSources = async () => {
    try {
      const res = await fetch(`/api/files?notebook_id=${notebook.id}`);
      const data = await res.json();
      const files = data.files || [];
      setSources(files);
      // По умолчанию выбираем все новые файлы
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
      {/* Sidebar */}
      <Sidebar 
        notebook={notebook}
        sources={sources}
        selectedSources={selectedSources}
        onSelectSources={setSelectedSources}
        onRefresh={fetchSources}
        onExit={handleExit}
        onOpenFile={openViewer}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-background relative z-0">
        <ChatArea 
          notebook={notebook}
          selectedSources={selectedSources}
          onOpenSource={(src) => {
             // src может быть объектом {file_name, text, index} из RAG
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
              {/* Resizer */}
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
