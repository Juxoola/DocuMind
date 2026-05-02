import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, Globe, Key, Cpu, HardDrive, Server, RefreshCw, FolderOpen, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

export default function SettingsModal({ isOpen, onClose, settings, onSave }) {
    const [localSettings, setLocalSettings] = useState(settings);
    const [activeTab, setActiveTab] = useState('api'); // 'api' | 'gguf'
    
    // GGUF state
    const [ggufModels, setGgufModels] = useState([]);
    const [ggufLoading, setGgufLoading] = useState(false);
    const [ggufLoadedModels, setGgufLoadedModels] = useState([]);
    const [expandedDirs, setExpandedDirs] = useState({});
    const [ggufConfig, setGgufConfig] = useState({});

    useEffect(() => {
        if (isOpen) {
            setLocalSettings(settings);
            fetchGgufModels();
            fetchGgufLoadedModels();
            fetchGgufConfig();
        }
    }, [isOpen, settings]);

    const fetchGgufModels = async () => {
        setGgufLoading(true);
        try {
            const res = await fetch('/api/gguf-models');
            const data = await res.json();
            setGgufModels(data.models || []);
        } catch (err) {
            console.error('Ошибка загрузки GGUF моделей:', err);
        } finally {
            setGgufLoading(false);
        }
    };

    const fetchGgufLoadedModels = async () => {
        try {
            const res = await fetch('/api/gguf-loaded');
            const data = await res.json();
            setGgufLoadedModels(data.loaded_models || []);
        } catch (err) {
            console.error('Ошибка загрузки списка моделей:', err);
        }
    };

    const fetchGgufConfig = async () => {
        try {
            const res = await fetch('/api/gguf-config');
            const data = await res.json();
            setGgufConfig(data);
        } catch (err) {
            console.error('Ошибка загрузки конфига GGUF:', err);
        }
    };

    const selectModel = (modelPath, mmprojPath) => {
        const newSettings = {
            ...localSettings,
            use_gguf: 'true',
            gguf_model_path: modelPath,
            gguf_mmproj_path: mmprojPath || '',
            llm_url: '', // Очищаем API URL
        };
        setLocalSettings(newSettings);
    };

    const unloadAllModels = async () => {
        try {
            await fetch('/api/gguf-unload', { method: 'POST' });
            await fetchGgufLoadedModels();
        } catch (err) {
            console.error('Ошибка выгрузки моделей:', err);
        }
    };

    const toggleDir = (dirPath) => {
        setExpandedDirs(prev => ({ ...prev, [dirPath]: !prev[dirPath] }));
    };

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
                    className="relative w-full max-w-lg bg-card border border-border shadow-2xl rounded-3xl overflow-hidden max-h-[90vh] flex flex-col"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between p-6 border-b border-border/50 flex-shrink-0">
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

                    {/* Tabs */}
                    <div className="flex border-b border-border/50 flex-shrink-0">
                        <button
                            onClick={() => setActiveTab('api')}
                            className={cn(
                                "flex-1 px-4 py-3 text-sm font-bold transition-all flex items-center justify-center gap-2",
                                activeTab === 'api'
                                    ? "text-primary border-b-2 border-primary bg-primary/5"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <Globe size={14} /> API / LM Studio
                        </button>
                        <button
                            onClick={() => setActiveTab('gguf')}
                            className={cn(
                                "flex-1 px-4 py-3 text-sm font-bold transition-all flex items-center justify-center gap-2",
                                activeTab === 'gguf'
                                    ? "text-primary border-b-2 border-primary bg-primary/5"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <HardDrive size={14} /> Локальные GGUF
                        </button>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-y-auto">
                        {/* API Tab */}
                        {activeTab === 'api' && (
                            <div className="p-6 space-y-6">
                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                        <Globe size={12} /> API Base URL
                                    </label>
                                    <input 
                                        type="text"
                                        value={localSettings.llm_url}
                                        onChange={(e) => setLocalSettings({...localSettings, llm_url: e.target.value, use_gguf: ''})}
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
                        )}

                        {/* GGUF Tab */}
                        {activeTab === 'gguf' && (
                            <div className="p-6 space-y-4">
                                {/* Loaded Models Status */}
                                {ggufLoadedModels.length > 0 && (
                                    <div className="rounded-2xl border border-primary/20 bg-primary/5 p-4">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                                <Server size={16} className="text-primary" />
                                                <p className="text-sm font-bold">Загружено в память</p>
                                            </div>
                                            <button
                                                onClick={unloadAllModels}
                                                className="px-2.5 py-1 bg-red-500/10 hover:bg-red-500/20 text-red-500 rounded-lg text-[10px] font-bold transition-all"
                                            >
                                                Выгрузить все
                                            </button>
                                        </div>
                                        <div className="space-y-1">
                                            {ggufLoadedModels.map((model, idx) => (
                                                <p key={idx} className="text-[10px] text-muted-foreground font-mono">
                                                    • {model}
                                                </p>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Current Selection */}
                                {localSettings.use_gguf === 'true' && localSettings.gguf_model_path && (
                                    <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-4">
                                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">
                                            Выбранная модель
                                        </p>
                                        <p className="text-xs font-medium text-green-600 dark:text-green-400 break-all">
                                            {localSettings.gguf_model_path.split('/').pop()}
                                        </p>
                                        {localSettings.gguf_mmproj_path && (
                                            <p className="text-[9px] text-muted-foreground mt-1">
                                                + mmproj: {localSettings.gguf_mmproj_path.split('/').pop()}
                                            </p>
                                        )}
                                    </div>
                                )}

                                {/* Model List */}
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between">
                                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-1.5">
                                            <FolderOpen size={12} /> Доступные модели
                                        </p>
                                        <button
                                            onClick={fetchGgufModels}
                                            disabled={ggufLoading}
                                            className="p-1.5 hover:bg-muted rounded-lg text-muted-foreground transition-all"
                                        >
                                            <RefreshCw size={12} className={ggufLoading ? "animate-spin" : ""} />
                                        </button>
                                    </div>
                                    
                                    {ggufConfig.search_dirs && (
                                        <p className="text-[9px] text-muted-foreground/50 font-mono">
                                            Директории поиска: {ggufConfig.search_dirs}
                                        </p>
                                    )}

                                    {ggufModels.length === 0 && !ggufLoading && (
                                        <div className="text-center py-6 opacity-40">
                                            <HardDrive size={32} className="mx-auto mb-2" />
                                            <p className="text-[10px]">GGUF модели не найдены</p>
                                            <p className="text-[9px] mt-1">Укажите директории в переменной GGUF_SEARCH_DIRS</p>
                                        </div>
                                    )}

                                    <div className="space-y-1.5 max-h-[40vh] overflow-y-auto custom-scrollbar">
                                        {ggufModels.map((modelGroup) => (
                                            <div key={modelGroup.dir} className="border border-border/30 rounded-xl overflow-hidden">
                                                {/* Directory Header */}
                                                <button
                                                    onClick={() => toggleDir(modelGroup.dir)}
                                                    className="w-full flex items-center justify-between p-3 hover:bg-muted/30 transition-all text-left"
                                                >
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        {expandedDirs[modelGroup.dir] ? (
                                                            <ChevronDown size={14} className="text-muted-foreground flex-shrink-0" />
                                                        ) : (
                                                            <ChevronRight size={14} className="text-muted-foreground flex-shrink-0" />
                                                        )}
                                                        <FolderOpen size={14} className="text-primary flex-shrink-0" />
                                                        <span className="text-xs font-bold truncate">{modelGroup.dir_name}</span>
                                                        <span className="text-[9px] text-muted-foreground/50">
                                                            ({modelGroup.gguf_files.length} моделей
                                                            {modelGroup.mmproj_files.length > 0 && `, ${modelGroup.mmproj_files.length} mmproj`})
                                                        </span>
                                                    </div>
                                                </button>

                                                {/* Expanded Content */}
                                                <AnimatePresence>
                                                    {expandedDirs[modelGroup.dir] && (
                                                        <motion.div
                                                            initial={{ height: 0, opacity: 0 }}
                                                            animate={{ height: "auto", opacity: 1 }}
                                                            exit={{ height: 0, opacity: 0 }}
                                                            className="overflow-hidden"
                                                        >
                                                            <div className="px-3 pb-3 space-y-2">
                                                                {/* LLM Models */}
                                                                {modelGroup.gguf_files.map((ggufFile) => {
                                                                    const fullPath = modelGroup.dir + '/' + ggufFile;
                                                                    const isSelected = localSettings.use_gguf === 'true' && 
                                                                        localSettings.gguf_model_path === fullPath;
                                                                    return (
                                                                        <div
                                                                            key={ggufFile}
                                                                            className={cn(
                                                                                "flex items-center justify-between p-2.5 rounded-lg border transition-all",
                                                                                isSelected
                                                                                    ? "bg-green-500/5 border-green-500/20"
                                                                                    : "bg-muted/10 border-border/20 hover:border-primary/30"
                                                                            )}
                                                                        >
                                                                            <div className="min-w-0 flex-1">
                                                                                <p className="text-[11px] font-medium truncate">{ggufFile}</p>
                                                                                {isSelected && (
                                                                                    <p className="text-[9px] text-green-500 font-bold mt-0.5">ВЫБРАНА</p>
                                                                                )}
                                                                            </div>
                                                                            <button
                                                                                onClick={() => {
                                                                                    // Находим mmproj в этой же директории
                                                                                    const mmproj = modelGroup.mmproj_files.length > 0
                                                                                        ? modelGroup.dir + '/' + modelGroup.mmproj_files[0]
                                                                                        : null;
                                                                                    selectModel(fullPath, mmproj);
                                                                                }}
                                                                                className={cn(
                                                                                    "px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all",
                                                                                    isSelected
                                                                                        ? "bg-green-500/10 text-green-500"
                                                                                        : "bg-primary/10 text-primary hover:bg-primary hover:text-white"
                                                                                )}
                                                                            >
                                                                                {isSelected ? "Выбрана" : "Выбрать"}
                                                                            </button>
                                                                        </div>
                                                                    );
                                                                })}

                                                                {/* mmproj files info */}
                                                                {modelGroup.mmproj_files.length > 0 && (
                                                                    <div className="pt-1">
                                                                        <p className="text-[9px] text-muted-foreground/50 uppercase tracking-wider font-bold">mmproj файлы (авто-подключение):</p>
                                                                        {modelGroup.mmproj_files.map((mp) => (
                                                                            <p key={mp} className="text-[9px] text-muted-foreground/40 font-mono mt-0.5">
                                                                                {mp}
                                                                            </p>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </motion.div>
                                                    )}
                                                </AnimatePresence>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="p-6 bg-muted/30 flex gap-3 flex-shrink-0">
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
