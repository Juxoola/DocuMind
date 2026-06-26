// LLM настройки: API/GGUF режим, параметры генерации, загрузка моделей.
import React from 'react';
import { RefreshCw, Globe, Key, Cpu, Server } from 'lucide-react';
import { cn } from '../lib/utils';
import VisionSettings from './VisionSettings';
import IngestionSettings from './IngestionSettings';

export default function LLMSettings({
    localSettings,
    setLocalSettings,
    isApiMode,
    ggufModels,
    ggufLoading,
    ggufLoadedModels,
    llmLoadState,
    ggufConfig,
    setGgufConfig,
    expandedDirs,
    selectModel,
    unloadAllModels,
    fetchGgufModels,
    toggleDir,
    updateSearchDirs,
}) {
    return (
        <div className="p-6 space-y-6">
            {/* Переключатель API / GGUF */}
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

            {/* Настройки API режима */}
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

            {/* Настройки GGUF режима */}
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
                        {/* Основные параметры GGUF */}
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

                        {/* Тонкие настройки генерации GGUF */}
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

                    {/* Vision модель */}
                    <VisionSettings localSettings={localSettings} setLocalSettings={setLocalSettings} />

                    {/* Браузер моделей */}
                    <IngestionSettings
                        localSettings={localSettings}
                        setLocalSettings={setLocalSettings}
                        ggufModels={ggufModels}
                        ggufLoading={ggufLoading}
                        ggufConfig={ggufConfig}
                        setGgufConfig={setGgufConfig}
                        expandedDirs={expandedDirs}
                        fetchGgufModels={fetchGgufModels}
                        selectModel={selectModel}
                        toggleDir={toggleDir}
                        updateSearchDirs={updateSearchDirs}
                    />
                </div>
            )}

            {/* Параметры генерации API */}
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
    );
}
