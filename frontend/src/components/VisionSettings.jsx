// Vision модель: конфигурация модели для описания изображений/видео.
import React from 'react';
import { Cpu } from 'lucide-react';
import { cn } from '../lib/utils';

export default function VisionSettings({ localSettings, setLocalSettings }) {
    if (!localSettings.vision_model_path) return null;

    return (
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
            {/* Тонкие настройки генерации для Vision */}
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
    );
}
