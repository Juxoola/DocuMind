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
  onToggle
}) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  const handleUpload = async (filesToUpload) => {
    const files = Array.from(filesToUpload);
    if (!files.length) return;

    setUploading(true);
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadUrl = new URL(`/api/upload`, window.location.origin);
      uploadUrl.searchParams.append('notebook_id', notebook.id);
      if (llmSettings) {
        if (llmSettings.llm_url) uploadUrl.searchParams.append('llm_url', llmSettings.llm_url);
        if (llmSettings.llm_api_key) uploadUrl.searchParams.append('llm_api_key', llmSettings.llm_api_key);
        if (llmSettings.llm_model) uploadUrl.searchParams.append('llm_model', llmSettings.llm_model);
      }

      try {
        const response = await fetch(uploadUrl.toString(), {
          method: 'POST',
          body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'progress') {
                  setProgress(data.pct);
                  setStatus(data.msg);
                }
              } catch (e) {}
            }
          }
        }
        // Вызываем обновление после каждого файла для постепенного появления
        onRefresh();
      } catch (err) {
        console.error(err);
      }
    }
    setUploading(false);
    setProgress(0);
    setStatus('');
    onRefresh();
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
      {/* Header */}
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

      {/* Upload Zone */}
      <div className="p-4 flex-shrink-0">
        <label 
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={cn(
            "relative flex flex-col items-center justify-center p-6 border-2 border-dashed border-border rounded-2xl cursor-pointer hover:border-primary/50 hover:bg-primary/5 transition-all group",
            isDragging && "border-primary bg-primary/10 scale-[1.02]",
            uploading && "pointer-events-none opacity-50"
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

          {uploading && (
            <div className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center p-4">
              <div className="w-full bg-muted h-1 rounded-full overflow-hidden mb-2">
                <motion.div 
                  className="bg-primary h-full" 
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                />
              </div>
              <span className="text-[10px] font-medium text-center truncate w-full">{status}</span>
            </div>
          )}
        </label>
      </div>

      {/* Sources List */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 custom-scrollbar">
        <div className="flex items-center justify-between px-2 mb-2">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Источники ({sources.length})</span>
          <button onClick={clearDatabase} className="text-destructive hover:bg-destructive/10 p-1 rounded-md transition-colors" title="Очистить базу">
             <Database size={12} />
          </button>
        </div>
        
        {sources.map((file) => (
          <div 
            key={file}
            className={cn(
              "group flex items-center justify-between gap-3 p-3 rounded-xl border transition-all cursor-pointer",
              selectedSources.includes(file) 
                ? "bg-primary/5 border-primary/20" 
                : "bg-transparent border-transparent hover:bg-muted"
            )}
            onClick={() => {
              if (selectedSources.includes(file)) {
                onSelectSources(selectedSources.filter(s => s !== file));
              } else {
                onSelectSources([...selectedSources, file]);
              }
            }}
          >
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <div className="p-1.5 bg-muted rounded-lg group-hover:bg-primary/10 transition-colors">
                <FileText size={14} className="text-muted-foreground group-hover:text-primary" />
              </div>
              <div className="min-w-0 flex-1" onClick={(e) => { e.stopPropagation(); onOpenFile(file); }}>
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
              
              <div className={cn(
                "transition-all duration-300 transform",
                selectedSources.includes(file) 
                  ? "text-primary scale-125 opacity-100" 
                  : "text-muted-foreground/20 scale-100 opacity-0 group-hover:opacity-30"
              )}>
                <CheckCircle2 size={24} strokeWidth={2.5} />
              </div>
            </div>
          </div>
        ))}

        {sources.length === 0 && !uploading && (
          <div className="text-center py-8 opacity-40">
            <FileText size={32} className="mx-auto mb-2" />
            <p className="text-[10px]">Нет источников</p>
          </div>
        )}
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
