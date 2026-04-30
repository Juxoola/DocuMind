import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, Globe, Key, Cpu } from 'lucide-react';
import { cn } from '../lib/utils';

export default function SettingsModal({ isOpen, onClose, settings, onSave }) {
  const [localSettings, setLocalSettings] = useState(settings);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-black/60 backdrop-blur-md"
        />
        
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          className="relative w-full max-w-md bg-card border border-border shadow-2xl rounded-3xl overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border/50">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <span className="p-2 bg-primary/10 rounded-xl text-primary">
                <Cpu size={20} />
              </span>
              Настройки LLM
            </h2>
            <button onClick={onClose} className="p-2 hover:bg-muted rounded-full transition-colors">
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                <Globe size={12} /> API Base URL
              </label>
              <input 
                type="text"
                value={localSettings.llm_url}
                onChange={(e) => setLocalSettings({...localSettings, llm_url: e.target.value})}
                placeholder="http://localhost:8889/v1"
                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
              />
              <p className="text-[10px] text-muted-foreground/60 italic">Для LM Studio обычно: http://localhost:8889/v1</p>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                <Key size={12} /> API Key
              </label>
              <input 
                type="password"
                value={localSettings.llm_api_key}
                onChange={(e) => setLocalSettings({...localSettings, llm_api_key: e.target.value})}
                placeholder="lm-studio"
                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
              />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                <Cpu size={12} /> Model Name
              </label>
              <input 
                type="text"
                value={localSettings.llm_model}
                onChange={(e) => setLocalSettings({...localSettings, llm_model: e.target.value})}
                placeholder="gpt-4o"
                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="p-6 bg-muted/30 flex gap-3">
            <button 
              onClick={onClose}
              className="flex-1 px-4 py-3 rounded-xl border border-border font-bold text-sm hover:bg-muted transition-all"
            >
              Отмена
            </button>
            <button 
              onClick={() => {
                onSave(localSettings);
                onClose();
              }}
              className="flex-1 px-4 py-3 rounded-xl bg-primary text-white font-bold text-sm shadow-lg shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all flex items-center justify-center gap-2"
            >
              <Save size={16} /> Сохранить
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
