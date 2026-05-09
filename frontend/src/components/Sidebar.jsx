import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, 
  Upload, 
  X, 
  LogOut, 
  ChevronLeft, 
  CheckCircle2, 
  AlertCircle,
  Database,
  Trash2
} from 'lucide-react';
import { cn } from '../lib/utils';
import axios from 'axios';

export default function Sidebar({ 
  notebook, 
  sources, 
  selectedSources, 
  onSelectSources, 
  onRefresh, 
  onExit,
  onOpenFile,
  llmSettings,
  width,
  onToggle,
  uploadState,
  onUpload
}) {
  const [isDragging, setIsDragging] = useState(false);

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

  return (
    <div style={{ width }} className="h-full border-r glass flex flex-col z-10 overflow-hidden">
      {/* Шапка */}
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

      {/* Зона загрузки */}
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
              {/* Текущий файл */}
              <div className="w-full mb-4">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-[9px] font-black uppercase tracking-widest text-primary">Файл</span>
                  <span className="text-[9px] font-bold text-primary">{Math.round(uploadState.progress)}%</span>
                </div>
                <div className="w-full bg-primary/10 h-1.5 rounded-full overflow-hidden shadow-inner">
                  <motion.div 
                    className="bg-primary h-full shadow-[0_0_10px_rgba(var(--primary),0.5)]" 
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadState.progress}%` }}
                    transition={{ type: "spring", stiffness: 100, damping: 20 }}
                  />
                </div>
              </div>

              {/* Общий прогресс */}
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
                  <motion.div 
                    className="bg-muted-foreground/40 h-full" 
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadState.batchProgress}%` }}
                  />
                </div>
              </div>

              {/* Статус в нижней части */}
              <div className="absolute bottom-0 left-0 right-0 p-2 bg-primary/10 border-t border-primary/20 backdrop-blur-sm">
                <p className="text-[9px] font-bold text-primary text-center break-words leading-tight">
                  {uploadState.status || 'Подготовка...'}
                </p>
              </div>
            </div>
          )}
        </label>
      </div>

      {/* Список источников */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 custom-scrollbar">
        {(() => {
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
                    <div className="p-1.5 bg-muted rounded-lg group-hover:bg-primary/10 transition-colors">
                      <FileText size={14} className="text-muted-foreground group-hover:text-primary" />
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
      </div>

      <div className="mt-auto flex flex-col border-t bg-muted/5">
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
    </div>
  );
}
