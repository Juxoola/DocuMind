// RAG настройки: конфигурация пайплайна поиска, реранкера, расширения запроса.
import React from 'react';
import { Globe, RefreshCw, Zap, Filter, MessageSquare, Database, Cpu, Search } from 'lucide-react';
import { cn } from '../lib/utils';

export default function RAGSettings({ ragConfig, setRagConfig, ggufModels }) {
    return (
        <div className="p-6 space-y-6">
            {/* Переключатель реранкера */}
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

            {/* Выбор моделей */}
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

            {/* Параметры пайплайна */}
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

            {/* Серверы и параметры */}
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

                {/* Расширение запроса */}
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

            {/* Информационные блоки */}
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
    );
}
