// Модалка настроек: LLM, RAG, GGUF-модели, конфигурация.
import React, { useState, useEffect } from 'react';
import { X, Save, Cpu, RefreshCw } from 'lucide-react';
import { cn } from '../lib/utils';
import LLMSettings from './LLMSettings';
import RAGSettings from './RAGSettings';

export default function SettingsModal({ isOpen, closing, onClose, onAnimEnd, settings, llmStatus, onSave }) {
    // ── Состояние компонента ──
    const [localSettings, setLocalSettings] = useState(settings);
    const [activeTab, setActiveTab] = useState('llm');
    const isApiMode = localSettings.use_gguf !== 'true';
    
    const [ggufModels, setGgufModels] = useState([]);
    const [ggufLoading, setGgufLoading] = useState(false);
    const [ggufLoadedModels, setGgufLoadedModels] = useState([]);
    const [expandedDirs, setExpandedDirs] = useState({});
    const [llmLoadState, setLlmLoadState] = useState({ state: 'idle', phase: null, model: null, elapsed: 0, eta: null, error: null });
    const [ggufConfig, setGgufConfig] = useState({ search_dirs: '', gguf_kv_quant: 8, presence_penalty: 0.0, frequency_penalty: 0.0, repeat_penalty: 1.1, top_p: 0.9, min_p: 0.05 });
    const [ragConfig, setRagConfig] = useState({ embedding_model: '', reranker_model: '', embedding_n_parallel: 2, top_k_per_file: 5, rerank_pool: 30, final_top_n: 10, use_reranker: true, query_expansion: true, rerank_score_threshold: 0.1, surya_mode: 'layout_only' });

    useEffect(() => {
        if (isOpen) {
            setLocalSettings(settings);
            fetchGgufModels();
            fetchGgufLoadedModels();
            fetchGgufConfig();
            fetchRagConfig();
        }
    }, [isOpen, settings]);

    // ── Используем llmStatus из пропсов вместо дублирующего EventSource ──
    useEffect(() => {
        if (llmStatus) {
            setLlmLoadState(prev => ({ ...prev, ...llmStatus }));
        }
    }, [llmStatus]);

    // ── API-запросы для управления конфигурацией ──
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

    // ── Выбор и предзагрузка GGUF-модели ──
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

    // ── Рендер модального окна настроек ──
    if (!isOpen && !closing) return null;

    return (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4">
            <div 
                onClick={closing ? undefined : onClose}
                onAnimationEnd={closing ? onAnimEnd : undefined}
                className={`absolute inset-0 bg-black/60 backdrop-blur-md ${closing ? 'animate-fadeOut' : 'animate-fadeIn'}`}
            />
            
            <div
                className={`relative w-full max-w-lg bg-card border border-border shadow-2xl rounded-3xl overflow-hidden max-h-[90vh] flex flex-col ${closing ? 'animate-scaleOut' : 'animate-scaleIn'}`}
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
                            <div key="llm-tab" className="animate-fadeIn">
                            <LLMSettings
                                localSettings={localSettings}
                                setLocalSettings={setLocalSettings}
                                isApiMode={isApiMode}
                                ggufModels={ggufModels}
                                ggufLoading={ggufLoading}
                                ggufLoadedModels={ggufLoadedModels}
                                llmLoadState={llmLoadState}
                                ggufConfig={ggufConfig}
                                setGgufConfig={setGgufConfig}
                                expandedDirs={expandedDirs}
                                selectModel={selectModel}
                                unloadAllModels={unloadAllModels}
                                fetchGgufModels={fetchGgufModels}
                                toggleDir={toggleDir}
                                updateSearchDirs={updateSearchDirs}
                            />
                            </div>
                        )}

                        {activeTab === 'rag' && (
                            <div key="rag-tab" className="animate-fadeIn">
                            <RAGSettings
                                ragConfig={ragConfig}
                                setRagConfig={setRagConfig}
                                ggufModels={ggufModels}
                            />
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
                                await updateRagConfig();
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
                </div>
            </div>
    );
}
