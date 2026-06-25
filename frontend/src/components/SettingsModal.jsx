// Модалка настроек: LLM, RAG, GGUF-модели, конфигурация.
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, Globe, Key, Cpu, HardDrive, Server, RefreshCw, FolderOpen, ChevronDown, ChevronRight, Zap, Filter, MessageSquare, Database, Search } from 'lucide-react';
import { cn } from '../lib/utils';

export default function SettingsModal({ isOpen, onClose, settings, onSave }) {
    const [localSettings, setLocalSettings] = useState(settings);
    const [activeTab, setActiveTab] = useState('llm');
    // Режим работы: true = внешний API, false = локальный GGUF
    const isApiMode = localSettings.use_gguf !== 'true';
    
    const [ggufModels, setGgufModels] = useState([]);
    const [ggufLoading, setGgufLoading] = useState(false);
    const [ggufLoadedModels, setGgufLoadedModels] = useState([]);
    const [expandedDirs, setExpandedDirs] = useState({});
    // SSE-подписка на статус загрузки модели в реальном времени
    const [llmLoadState, setLlmLoadState] = useState({ state: 'idle', phase: null, model: null, elapsed: 0, eta: null, error: null });
    const [ggufConfig, setGgufConfig] = useState({
        search_dirs: '',
        // Q8_0 — единственное квантование KV-cache, совместимое с llama-server
        gguf_kv_quant: 8,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_penalty: 1.1,
        top_p: 0.9,
        min_p: 0.05
    });
    const [ragConfig, setRagConfig] = useState({
        embedding_model: '',
        reranker_model: '',
        embedding_n_parallel: 2,
        top_k_per_file: 5,
        rerank_pool: 30,
        final_top_n: 10,
        use_reranker: true,
        query_expansion: true,
        rerank_score_threshold: 0.1
    });

    useEffect(() => {
        if (isOpen) {
            setLocalSettings(settings);
            fetchGgufModels();
            fetchGgufLoadedModels();
            fetchGgufConfig();
            fetchRagConfig();
        }
    }, [isOpen, settings]);

    useEffect(() => {
        if (!isOpen) return;
        let es;
        let reconnectTimer;
        const connect = () => {
            es = new EventSource('/api/llm-status/stream');
            es.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    setLlmLoadState(prev => ({ ...prev, ...data }));
                } catch (err) { }
            };
            es.onerror = () => {
                if (es) { es.close(); es = null; }
                reconnectTimer = setTimeout(connect, 2000);
            };
        };
        connect();
        return () => {
            if (es) es.close();
            if (reconnectTimer) clearTimeout(reconnectTimer);
        };
    }, [isOpen]);

    const fetchRagConfig = async () => {
        try {
            const res = await fetch('/api/rag-config');
            const data = await res.json();
            setRagConfig(data);
        } catch (err) {
            console.error('Ошибка загрузки RAG конфига:', err);
        }
    };

    const updateRagConfig = async () => {
        try {
            const res = await fetch('/api/update-rag-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(ragConfig)
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                console.error('Ошибка RAG конфига:', res.status, err);
            }
        } catch (err) {
            console.error('Ошибка обновления RAG конфига:', err);
        }
    };

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

    const selectModel = async (modelPath, mmprojPath) => {
        const newSettings = {
            ...localSettings,
            use_gguf: 'true',
            gguf_model_path: modelPath,
            gguf_mmproj_path: mmprojPath || '',
            llm_url: '',
        };
        setLocalSettings(newSettings);
        onSave(newSettings);
        localStorage.setItem('llm_settings', JSON.stringify(newSettings));
        // Hot-swap: предзагрузка модели в фоне убирает задержку 10-30с при первом сообщении
        try {
            setLlmLoadState({ state: 'loading', model: modelPath, phase: 'starting', elapsed: 0, eta: null });
            const res = await fetch('/api/preload-llm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gguf_model_path: modelPath,
                    gguf_mmproj_path: mmprojPath || null,
                    gguf_ctx_size: newSettings.gguf_ctx_size,
                    gguf_gpu_layers: newSettings.gguf_gpu_layers,
                    gguf_threads: newSettings.gguf_threads,
                    gguf_batch_size: newSettings.gguf_batch_size,
                    gguf_ubatch_size: newSettings.gguf_ubatch_size,
                    gguf_flash_attn: newSettings.gguf_flash_attn,
                    max_tokens: newSettings.max_tokens,
                    gguf_kv_quant: newSettings.gguf_kv_quant,
                    thinking_mode: newSettings.thinking_mode,
                    thinking_budget: newSettings.thinking_budget,
                    mtp_enabled: newSettings.mtp_enabled,
                }),
            });
            const data = await res.json();
            if (data.status === 'error') {
                setLlmLoadState(prev => ({ ...prev, state: 'error', error: data.error || 'unknown' }));
            }
        } catch (err) {
            console.error('preload-llm failed:', err);
            setLlmLoadState(prev => ({ ...prev, state: 'error', error: err.message }));
        }
    };

    const unloadAllModels = async () => {
        try {
            await fetch('/api/gguf-unload', { method: 'POST' });
            await fetchGgufLoadedModels();
        } catch (err) {
            console.error('Ошибка выгрузки моделей:', err);
        }
    };

    
    const updateSearchDirs = async (dirs) => {
        try {
            await fetch('/api/update-model-dirs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dirs })
            });
            fetchGgufConfig();
            fetchGgufModels();
        } catch (err) {
            console.error('Ошибка обновления директорий:', err);
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

                    <div className="flex border-b border-border/50 flex-shrink-0">
                        <button
                            onClick={() => setActiveTab('llm')}
                            className={cn(
                                "flex-1 px-4 py-3 text-sm font-bold transition-all flex items-center justify-center gap-2",
                                activeTab === 'llm'
                                    ? "text-primary border-b-2 border-primary bg-primary/5"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <Cpu size={14} /> LLM
                        </button>
                        <button
                            onClick={() => setActiveTab('rag')}
                            className={cn(
                                "flex-1 px-4 py-3 text-sm font-bold transition-all flex items-center justify-center gap-2",
                                activeTab === 'rag'
                                    ? "text-primary border-b-2 border-primary bg-primary/5"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <RefreshCw size={14} /> Поиск и RAG
                        </button>
                    </div>

                    <div className="flex-1 overflow-y-auto">
                        {activeTab === 'llm' && (
                            <div className="p-6 space-y-6">
                                <div className="flex items-center justify-between p-4 rounded-2xl bg-muted/30 border border-border/50">
                                    <div className="space-y-1">
                                        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                            <Globe size={12} /> Использовать API
                                        </label>
                                        <p className="text-[10px] text-muted-foreground/60">
                                            {isApiMode ? 'Подключение к внешнему API' : 'Локальный запуск GGUF'}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => {
                                            if (isApiMode) {
                                                setLocalSettings({...localSettings, use_gguf: 'true', llm_url: ''});
                                            } else {
                                                setLocalSettings({...localSettings, use_gguf: '', gguf_model_path: ''});
                                            }
                                        }}
                                        className={cn(
                                            "w-10 h-6 rounded-full transition-all relative",
                                            isApiMode ? "bg-primary" : "bg-muted-foreground/30"
                                        )}
                                    >
                                        <div className={cn(
                                            "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                                            isApiMode ? "left-5" : "left-1"
                                        )} />
                                    </button>
                                </div>

                                {isApiMode && (
                                    <div className="space-y-4">
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
                                            <p className="text-[10px] text-muted-foreground/60 italic">Для LM Studio: http://localhost:8889/v1</p>
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
                                                autoComplete="off"
                                                spellCheck="false"
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            />
                                            <p className="text-[10px] text-amber-500/80 italic flex items-start gap-1">
                                                <span>⚠</span>
                                                <span>API-ключ хранится в localStorage. Используйте только локальные ключи (lm-studio).</span>
                                            </p>
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

                                        <div className="space-y-2">
                                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                                <Cpu size={12} /> Context Window
                                            </label>
                                            <input
                                                type="number"
                                                step="1024"
                                                value={localSettings.llm_ctx_size || 8192}
                                                onChange={(e) => setLocalSettings({...localSettings, llm_ctx_size: parseInt(e.target.value) || 8192})}
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            />
                                            <p className="text-[10px] text-muted-foreground/60 italic">Максимальный размер контекстного окна модели в токенах</p>
                                        </div>
                                    </div>
                                )}

                                {!isApiMode && (
                                    <div className="space-y-4">
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

                                        {localSettings.gguf_model_path && (
                                        <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-4">
                                            <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2" title="Основная модель используется для ответов на вопросы и работы с текстом">
                                                Выбранная основная модель ℹ️
                                            </p>
                                            <p className="text-xs font-medium text-green-600 dark:text-green-400 break-all">
                                                {localSettings.gguf_model_path.split('/').pop()}
                                            </p>
                                            {localSettings.gguf_mmproj_path && (
                                                <p className="text-[9px] text-muted-foreground mt-1">
                                                    + mmproj: {localSettings.gguf_mmproj_path.split('/').pop()}
                                                </p>
                                            )}
                                            {llmLoadState.state === 'loading' && (
                                                <div className="mt-3 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <RefreshCw className="w-3 h-3 text-blue-500 animate-spin" />
                                                        <span className="text-[11px] font-bold text-blue-600 dark:text-blue-400">
                                                            Загрузка модели...
                                                        </span>
                                                    </div>
                                                    <p className="text-[10px] text-muted-foreground">
                                                        Фаза: {llmLoadState.phase === 'freeing' ? 'выгрузка старой' :
                                                               llmLoadState.phase === 'starting' ? 'подготовка' :
                                                               llmLoadState.phase === 'loading_model' ? 'загрузка в память' :
                                                               llmLoadState.phase || '...'}
                                                        {llmLoadState.elapsed != null && ` · ${llmLoadState.elapsed.toFixed(0)}с`}
                                                        {llmLoadState.eta != null && llmLoadState.eta > 0 && ` · ~${llmLoadState.eta.toFixed(0)}с осталось`}
                                                    </p>
                                                </div>
                                            )}
                                            {llmLoadState.state === 'ready' && (
                                                <div className="mt-3 p-2 rounded-xl bg-green-500/10 border border-green-500/20">
                                                    <span className="text-[10px] font-bold text-green-600 dark:text-green-400">
                                                        ✅ Модель готова{llmLoadState.elapsed != null && ` (загружена за ${llmLoadState.elapsed.toFixed(0)}с)`}
                                                    </span>
                                                </div>
                                            )}
                                            {llmLoadState.state === 'error' && (
                                                <div className="mt-3 p-2 rounded-xl bg-red-500/10 border border-red-500/20">
                                                    <p className="text-[10px] font-bold text-red-600 dark:text-red-400">
                                                        ❌ Ошибка загрузки
                                                    </p>
                                                    <p className="text-[9px] text-muted-foreground mt-1 break-all">
                                                        {llmLoadState.error}
                                                    </p>
                                                </div>
                                            )}
                                            <div className="grid grid-cols-2 gap-3 mt-3">
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Температура (0.0 - 2.0). Чем выше, тем более креативный ответ.">
                                                    Температура ℹ️
                                                    <input type="number" step="0.1" value={localSettings.gguf_temperature || 0.1} onChange={e => setLocalSettings({...localSettings, gguf_temperature: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Размер контекста (токены). Сколько текста модель может 'помнить'.">
                                                    Контекст ℹ️
                                                    <input type="number" step="1024" value={localSettings.gguf_ctx_size || 8192} onChange={e => setLocalSettings({...localSettings, gguf_ctx_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Количество GPU слоев (-1 означает загрузить всё в видеокарту).">
                                                    GPU Слои ℹ️
                                                    <input type="number" step="1" value={localSettings.gguf_gpu_layers !== undefined ? localSettings.gguf_gpu_layers : -1} onChange={e => setLocalSettings({...localSettings, gguf_gpu_layers: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Потоки процессора. Обычно 8 достаточно.">
                                                    CPU Потоки ℹ️
                                                    <input type="number" step="1" value={localSettings.gguf_threads || 8} onChange={e => setLocalSettings({...localSettings, gguf_threads: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Логический размер батча (влияет на скорость обработки длинного текста).">
                                                    Batch Size ℹ️
                                                    <input type="number" step="256" value={localSettings.gguf_batch_size || 512} onChange={e => setLocalSettings({...localSettings, gguf_batch_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Физический (micro-batch) размер. Должен быть <= Batch Size. Уменьшите при нехватке VRAM.">
                                                    UBatch Size ℹ️
                                                    <input type="number" step="64" value={localSettings.gguf_ubatch_size || 256} onChange={e => setLocalSettings({...localSettings, gguf_ubatch_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Использовать Flash Attention для ускорения.">
                                                    Flash Attention ℹ️
                                                    <select value={localSettings.gguf_flash_attn || 'false'} onChange={e => setLocalSettings({...localSettings, gguf_flash_attn: e.target.value})} className="bg-background border border-border rounded px-2 py-1 text-foreground">
                                                        <option value="true">Вкл</option>
                                                        <option value="false">Выкл</option>
                                                    </select>
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Multi-Token Prediction (--spec-type draft-mtp). Включайте только если модель обучена для MTP (например, Qwen3). Ускоряет генерацию, но увеличивает VRAM.">
                                                    MTP (draft-mtp) ℹ️
                                                    <select value={(localSettings.mtp_enabled === true || localSettings.mtp_enabled === 'true') ? 'true' : 'false'} onChange={e => setLocalSettings({...localSettings, mtp_enabled: e.target.value === 'true'})} className="bg-background border border-border rounded px-2 py-1 text-foreground">
                                                        <option value="false">Выкл</option>
                                                        <option value="true">Вкл</option>
                                                    </select>
                                                </label>
                                            </div>

                                            <div className="mt-4 pt-4 border-t border-border/20 space-y-4">
                                                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-1.5">
                                                    <Cpu size={10} /> Тонкие настройки генерации
                                                </p>
                                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Сжатие памяти кэша. Экономит VRAM (видеопамять), позволяя использовать более длинный контекст.">
                                                        KV Cache Quant ℹ️
                                                        <select 
                                                            value={localSettings.gguf_kv_quant || 2}
                                                            onChange={(e) => setLocalSettings({...localSettings, gguf_kv_quant: parseInt(e.target.value)})}
                                                            className="bg-background border border-border rounded px-2 py-1 text-foreground"
                                                        >
                                                            <option value={0}>F16</option>
                                                            <option value={8}>Q8_0</option>
                                                            <option value={6}>Q5_K</option>
                                                            <option value={4}>Q4_K</option>
                                                        </select>
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за повторения. Помогает модели не зацикливаться на одних и тех же словах.">
                                                        Repeat Pen. ℹ️
                                                        <input type="number" step="0.1" value={localSettings.repeat_penalty || 1.1} onChange={e => setLocalSettings({...localSettings, repeat_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за упоминание уже использованных тем. Делает ответы более разнообразными.">
                                                        Presence Pen. ℹ️
                                                        <input type="number" step="0.1" value={localSettings.presence_penalty || 0.0} onChange={e => setLocalSettings({...localSettings, presence_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Вероятностная фильтрация (Nucleus sampling). 0.9 - стандарт.">
                                                        Top-P ℹ️
                                                        <input type="number" step="0.05" value={localSettings.top_p || 0.9} onChange={e => setLocalSettings({...localSettings, top_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Минимальный порог вероятности. Отсеивает маловероятные 'мусорные' токены.">
                                                        Min-P ℹ️
                                                        <input type="number" step="0.01" value={localSettings.min_p || 0.05} onChange={e => setLocalSettings({...localSettings, min_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                        )}

                                        {localSettings.vision_model_path && (
                                        <div className="rounded-2xl border border-purple-500/20 bg-purple-500/5 p-4">
                                            <div className="flex justify-between items-start">
                                                <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2" title="Vision модель загружается только при анализе видео/фото для создания описаний.">
                                                    Vision модель (для описаний) ℹ️
                                                </p>
                                                <button onClick={() => setLocalSettings({...localSettings, vision_model_path: '', vision_mmproj_path: ''})} className="text-red-500 hover:text-red-400 text-[10px]">Удалить</button>
                                            </div>
                                            <p className="text-xs font-medium text-purple-600 dark:text-purple-400 break-all">
                                                {localSettings.vision_model_path.split('/').pop()}
                                            </p>
                                            {localSettings.vision_mmproj_path && (
                                                <p className="text-[9px] text-muted-foreground mt-1">
                                                    + mmproj: {localSettings.vision_mmproj_path.split('/').pop()}
                                                </p>
                                            )}
                                            <div className="grid grid-cols-2 gap-3 mt-3">
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Температура для генерации описаний картинок.">
                                                    Температура ℹ️
                                                    <input type="number" step="0.1" value={localSettings.vision_temperature || 0.2} onChange={e => setLocalSettings({...localSettings, vision_temperature: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Контекст для обработки картинок (обычно 4096 или 8192)">
                                                    Контекст ℹ️
                                                    <input type="number" step="1024" value={localSettings.vision_ctx_size || 8192} onChange={e => setLocalSettings({...localSettings, vision_ctx_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Количество GPU слоев (-1 означает загрузить всё в видеокарту).">
                                                    GPU Слои ℹ️
                                                    <input type="number" step="1" value={localSettings.vision_gpu_layers !== undefined ? localSettings.vision_gpu_layers : -1} onChange={e => setLocalSettings({...localSettings, vision_gpu_layers: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Потоки процессора. Обычно 8 достаточно.">
                                                    CPU Потоки ℹ️
                                                    <input type="number" step="1" value={localSettings.vision_threads || 8} onChange={e => setLocalSettings({...localSettings, vision_threads: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Логический размер батча для Vision модели.">
                                                    Batch Size ℹ️
                                                    <input type="number" step="256" value={localSettings.vision_batch_size || 512} onChange={e => setLocalSettings({...localSettings, vision_batch_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Физический (micro-batch) размер для Vision модели. Должен быть <= Batch Size.">
                                                    UBatch Size ℹ️
                                                    <input type="number" step="64" value={localSettings.vision_ubatch_size || 256} onChange={e => setLocalSettings({...localSettings, vision_ubatch_size: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Использовать Flash Attention для ускорения.">
                                                    Flash Attention ℹ️
                                                    <select value={localSettings.vision_flash_attn || 'true'} onChange={e => setLocalSettings({...localSettings, vision_flash_attn: e.target.value})} className="bg-background border border-border rounded px-2 py-1 text-foreground">
                                                        <option value="true">Вкл</option>
                                                        <option value="false">Выкл</option>
                                                    </select>
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Multi-Token Prediction (--spec-type draft-mtp) для Vision модели. Включайте только если модель обучена для MTP.">
                                                    MTP (draft-mtp) ℹ️
                                                    <select value={(localSettings.vision_mtp_enabled === true || localSettings.vision_mtp_enabled === 'true') ? 'true' : 'false'} onChange={e => setLocalSettings({...localSettings, vision_mtp_enabled: e.target.value === 'true'})} className="bg-background border border-border rounded px-2 py-1 text-foreground">
                                                        <option value="false">Выкл</option>
                                                        <option value="true">Вкл</option>
                                                    </select>
                                                </label>
                                                <label className="flex flex-col gap-1 text-[10px] text-muted-foreground col-span-2" title="Максимальное количество токенов в ответе.">
                                                    Max Tokens ℹ️
                                                    <input type="number" step="64" value={localSettings.vision_max_tokens || 512} onChange={e => setLocalSettings({...localSettings, vision_max_tokens: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                </label>
                                            </div>
                                            <div className="mt-4 pt-4 border-t border-border/20 space-y-4">
                                                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-1.5">
                                                    <Cpu size={10} /> Тонкие настройки генерации
                                                </p>
                                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Сжатие памяти кэша для Vision модели. Помогает при анализе длинных видео.">
                                                        KV Cache Quant ℹ️
                                                        <select 
                                                            value={localSettings.vision_kv_quant || 4}
                                                            onChange={(e) => setLocalSettings({...localSettings, vision_kv_quant: parseInt(e.target.value)})}
                                                            className="bg-background border border-border rounded px-2 py-1 text-foreground"
                                                        >
                                                            <option value={0}>F16</option>
                                                            <option value={8}>Q8_0</option>
                                                            <option value={6}>Q5_K</option>
                                                            <option value={4}>Q4_K</option>
                                                        </select>
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за повторения для Vision модели.">
                                                        Repeat Pen. ℹ️
                                                        <input type="number" step="0.1" value={localSettings.vision_repeat_penalty || 1.2} onChange={e => setLocalSettings({...localSettings, vision_repeat_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за упоминание тем для Vision модели.">
                                                        Presence Pen. ℹ️
                                                        <input type="number" step="0.1" value={localSettings.vision_presence_penalty || 0.0} onChange={e => setLocalSettings({...localSettings, vision_presence_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Вероятностная фильтрация для Vision модели.">
                                                        Top-P ℹ️
                                                        <input type="number" step="0.05" value={localSettings.vision_top_p || 0.9} onChange={e => setLocalSettings({...localSettings, vision_top_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Минимальный порог вероятности для Vision модели.">
                                                        Min-P ℹ️
                                                        <input type="number" step="0.01" value={localSettings.vision_min_p || 0.05} onChange={e => setLocalSettings({...localSettings, vision_min_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                    <label className="flex flex-col gap-1 text-[10px] text-muted-foreground col-span-2" title="Количество параллельных потоков анализа изображений. Ускоряет обработку видео.">
                                                        Concurrency ℹ️
                                                        <input type="number" min="1" max="8" value={localSettings.vision_concurrency || 1} onChange={e => setLocalSettings({...localSettings, vision_concurrency: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                        )}

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
                                            {ggufConfig.search_dirs !== undefined && (
                                                <div className="flex items-center gap-2 mt-2 mb-2">
                                                    <input type="text" value={ggufConfig.search_dirs} onChange={e => setGgufConfig({...ggufConfig, search_dirs: e.target.value})} className="flex-1 bg-background border border-border rounded px-2 py-1 text-[10px] text-foreground font-mono" placeholder="C:/models, D:/llms" />
                                                    <button onClick={() => updateSearchDirs(ggufConfig.search_dirs)} className="px-2 py-1 bg-primary/20 hover:bg-primary/40 text-primary rounded text-[10px]">Сохранить</button>
                                                </div>
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
                                                        <AnimatePresence>
                                                            {expandedDirs[modelGroup.dir] && (
                                                                <motion.div
                                                                    initial={{ height: 0, opacity: 0 }}
                                                                    animate={{ height: "auto", opacity: 1 }}
                                                                    exit={{ height: 0, opacity: 0 }}
                                                                    className="overflow-hidden"
                                                                >
                                                                    <div className="px-3 pb-3 space-y-2">
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
                                                                                    <div className="flex gap-2">
                                                                                        <button
                                                                                            onClick={() => {
                                                                                                const mmproj = modelGroup.mmproj_files.length > 0 ? modelGroup.dir + '/' + modelGroup.mmproj_files[0] : null;
                                                                                                selectModel(fullPath, mmproj);
                                                                                            }}
                                                                                            className={cn("px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all", isSelected ? "bg-green-500/10 text-green-500" : "bg-primary/10 text-primary hover:bg-primary hover:text-white")}
                                                                                        >
                                                                                            {isSelected ? "Основа" : "Выбрать"}
                                                                                        </button>
                                                                                        <button
                                                                                            onClick={() => {
                                                                                                const mmproj = modelGroup.mmproj_files.length > 0 ? modelGroup.dir + '/' + modelGroup.mmproj_files[0] : null;
                                                                                                setLocalSettings({...localSettings, vision_model_path: fullPath, vision_mmproj_path: mmproj || '', use_gguf: 'true'});
                                                                                            }}
                                                                                            className={cn("px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all", localSettings.vision_model_path === fullPath ? "bg-purple-500/10 text-purple-500" : "bg-primary/10 text-primary hover:bg-primary hover:text-white")}
                                                                                        >
                                                                                            {localSettings.vision_model_path === fullPath ? "Vision" : "+ Vision"}
                                                                                        </button>
                                                                                    </div>
                                                                                </div>
                                                                            );
                                                                        })}
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

                                {isApiMode && (
                                <div className="rounded-2xl border border-border/30 p-4 space-y-3">
                                    <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-1.5">
                                        <Cpu size={10} /> Параметры генерации
                                    </p>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Температура (0.0 - 2.0). Чем выше, тем более креативный ответ.">
                                            Temperature ℹ️
                                            <input type="number" step="0.1" value={localSettings.temperature ?? 0.1} onChange={e => setLocalSettings({...localSettings, temperature: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Top-P (Nucleus sampling). 0.9 - стандарт.">
                                            Top-P ℹ️
                                            <input type="number" step="0.05" value={localSettings.top_p ?? 0.9} onChange={e => setLocalSettings({...localSettings, top_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Минимальный порог вероятности токена.">
                                            Min-P ℹ️
                                            <input type="number" step="0.01" value={localSettings.min_p ?? 0.05} onChange={e => setLocalSettings({...localSettings, min_p: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Максимальное количество токенов в ответе.">
                                            Max Tokens ℹ️
                                            <input type="number" step="64" value={localSettings.max_tokens ?? 8192} onChange={e => setLocalSettings({...localSettings, max_tokens: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за повторения. Помогает избежать циклов.">
                                            Repeat Pen. ℹ️
                                            <input type="number" step="0.1" value={localSettings.repeat_penalty ?? 1.1} onChange={e => setLocalSettings({...localSettings, repeat_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за повторение тем. Делает ответ разнообразнее.">
                                            Presence Pen. ℹ️
                                            <input type="number" step="0.1" value={localSettings.presence_penalty ?? 0.0} onChange={e => setLocalSettings({...localSettings, presence_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Штраф за частотность. Снижает повторение одних и тех же слов.">
                                            Freq Pen. ℹ️
                                            <input type="number" step="0.1" value={localSettings.frequency_penalty ?? 0.0} onChange={e => setLocalSettings({...localSettings, frequency_penalty: parseFloat(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                        </label>
                                        <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Включить режим рассуждений (thinking/CoT).">
                                            Thinking ℹ️
                                            <select value={localSettings.thinking_mode === true || localSettings.thinking_mode === 'true' ? 'true' : 'false'} onChange={e => setLocalSettings({...localSettings, thinking_mode: e.target.value === 'true'})} className="bg-background border border-border rounded px-2 py-1 text-foreground">
                                                <option value="false">Выкл</option>
                                                <option value="true">Вкл</option>
                                            </select>
                                        </label>
                                    </div>
                                    {localSettings.thinking_mode === true || localSettings.thinking_mode === 'true' ? (
                                        <div className="mt-2">
                                            <label className="flex flex-col gap-1 text-[10px] text-muted-foreground" title="Бюджет токенов для thinking (если поддерживается моделью)">
                                                Thinking Budget ℹ️
                                                <input type="number" step="512" value={localSettings.thinking_budget ?? 2048} onChange={e => setLocalSettings({...localSettings, thinking_budget: parseInt(e.target.value)})} className="bg-background border border-border rounded px-2 py-1 text-foreground" />
                                            </label>
                                        </div>
                                    ) : null}
                                </div>
                                )}
                            </div>
                        )}

                        {activeTab === 'rag' && (
                            <div className="p-6 space-y-6">
                                <div className="flex items-center justify-between p-4 rounded-2xl bg-muted/30 border border-border/50">
                                    <div className="space-y-1">
                                        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                            <Zap size={12} className={ragConfig.use_reranker ? "text-amber-500" : ""} /> Использовать реранкер
                                        </label>
                                        <p className="text-[10px] text-muted-foreground/60">Повышает точность, но требует больше VRAM</p>
                                    </div>
                                    <button 
                                        onClick={() => setRagConfig({...ragConfig, use_reranker: !ragConfig.use_reranker})}
                                        className={cn(
                                            "w-10 h-6 rounded-full transition-all relative",
                                            ragConfig.use_reranker ? "bg-primary" : "bg-muted-foreground/30"
                                        )}
                                    >
                                        <div className={cn(
                                            "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                                            ragConfig.use_reranker ? "left-5" : "left-1"
                                        )} />
                                    </button>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                        <Globe size={12} /> Embedding Model (GGUF)
                                    </label>
                                    {(() => {
                                        const allGgufFiles = ggufModels?.flatMap(g => g.gguf_files.map(f => ({ name: f, path: g.dir + '/' + f }))) || [];
                                        return (
                                            <select
                                                value={ragConfig.embedding_model}
                                                onChange={(e) => setRagConfig({...ragConfig, embedding_model: e.target.value})}
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            >
                                                <option value="" disabled>Выберите GGUF модель...</option>
                                                {allGgufFiles.map(f => (
                                                    <option key={f.path} value={f.path}>
                                                        {f.name}
                                                    </option>
                                                ))}
                                            </select>
                                        );
                                    })()}
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                        <RefreshCw size={12} /> Reranker Model (GGUF)
                                    </label>
                                    {(() => {
                                        const allGgufFiles = ggufModels?.flatMap(g => g.gguf_files.map(f => ({ name: f, path: g.dir + '/' + f }))) || [];
                                        return (
                                            <select
                                                value={ragConfig.reranker_model}
                                                onChange={(e) => setRagConfig({...ragConfig, reranker_model: e.target.value})}
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            >
                                                <option value="" disabled>Выберите GGUF модель...</option>
                                                {allGgufFiles.map(f => (
                                                    <option key={f.path} value={f.path}>
                                                        {f.name}
                                                    </option>
                                                ))}
                                            </select>
                                        );
                                    })()}
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                            <Filter size={12} /> Top-K на файл
                                        </label>
                                        <input 
                                            type="number"
                                            value={ragConfig.top_k_per_file}
                                            onChange={(e) => setRagConfig({...ragConfig, top_k_per_file: parseInt(e.target.value)})}
                                            className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                            <Zap size={12} /> Пул реранкера
                                        </label>
                                        <input 
                                            type="number"
                                            value={ragConfig.rerank_pool}
                                            onChange={(e) => setRagConfig({...ragConfig, rerank_pool: parseInt(e.target.value)})}
                                            className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                            <MessageSquare size={12} /> Итоговый топ
                                        </label>
                                        <input 
                                            type="number"
                                            value={ragConfig.final_top_n}
                                            onChange={(e) => setRagConfig({...ragConfig, final_top_n: parseInt(e.target.value)})}
                                            className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                        />
                                    </div>
                                </div>

                                <div className="p-4 rounded-2xl bg-muted/30 border border-border/50 space-y-4">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-1.5">
                                        <Database size={10} /> Серверы Embedding и Reranker
                                    </p>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                                <Cpu size={12} /> Embedding Parallel
                                            </label>
                                            <input 
                                                type="number"
                                                min="1"
                                                max="8"
                                                value={ragConfig.embedding_n_parallel}
                                                onChange={(e) => setRagConfig({...ragConfig, embedding_n_parallel: parseInt(e.target.value) || 1})}
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            />
                                            <p className="text-[9px] text-muted-foreground/60 italic">Параллельные запросы эмбеддингов (1-8). Больше → быстрее, но больше VRAM.</p>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                                <Filter size={12} /> Порог реранкера
                                            </label>
                                            <input 
                                                type="number"
                                                step="0.05"
                                                min="0"
                                                max="1"
                                                value={ragConfig.rerank_score_threshold}
                                                onChange={(e) => setRagConfig({...ragConfig, rerank_score_threshold: parseFloat(e.target.value) || 0})}
                                                className="w-full bg-muted/30 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                            />
                                            <p className="text-[9px] text-muted-foreground/60 italic">Минимальный score реранкера. Ниже этого → фрагмент отбрасывается.</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between">
                                        <div className="space-y-1">
                                            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                                <Search size={12} /> Расширение запроса
                                            </label>
                                            <p className="text-[9px] text-muted-foreground/60 italic">LLM переформулирует запрос для лучшего поиска</p>
                                        </div>
                                        <button 
                                            onClick={() => setRagConfig({...ragConfig, query_expansion: !ragConfig.query_expansion})}
                                            className={cn(
                                                "w-10 h-6 rounded-full transition-all relative",
                                                ragConfig.query_expansion ? "bg-primary" : "bg-muted-foreground/30"
                                            )}
                                        >
                                            <div className={cn(
                                                "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                                                ragConfig.query_expansion ? "left-5" : "left-1"
                                            )} />
                                        </button>
                                    </div>
                                </div>

                                <div className="p-4 rounded-2xl bg-blue-500/10 border border-blue-500/20">
                                    <p className="text-[10px] text-blue-600 dark:text-blue-400 leading-relaxed">
                                        <b>Воронка поиска:</b> Сначала берется по <b>{ragConfig.top_k_per_file}</b> фрагментов из каждого файла. Затем из них выбираются лучшие <b>{ragConfig.rerank_pool}</b> и отправляются реранкеру. В итоге модель получает <b>{ragConfig.final_top_n}</b> самых точных совпадений.
                                    </p>
                                </div>

                                <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/20">
                                    <p className="text-[10px] text-amber-600 dark:text-amber-400 leading-relaxed">
                                        <b>Примечание:</b> Смена GGUF-модели запускает новый сервер на отдельном порту. Убедитесь, что файл .gguf существует локально.
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="p-6 bg-muted/30 flex gap-3 flex-shrink-0">
                        <button 
                            onClick={onClose}
                            className="flex-1 px-4 py-3 rounded-xl border border-border font-bold text-sm hover:bg-muted transition-all"
                        >
                            Отмена
                        </button>
                        <button
                            onClick={async () => {
                                onSave(localSettings);
                                if (activeTab === 'rag') {
                                    await updateRagConfig();
                                }
                                if (ggufConfig.search_dirs !== undefined) {
                                    await updateSearchDirs(ggufConfig.search_dirs);
                                }
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
