// Браузер GGUF моделей: выбор моделей, управление директориями поиска.
// ── Импорты ──
import React from 'react';
import { HardDrive, FolderOpen, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

// ── Компонент браузера GGUF моделей ──
export default function IngestionSettings({
    localSettings,
    setLocalSettings,
    ggufModels,
    ggufLoading,
    ggufConfig,
    setGgufConfig,
    expandedDirs,
    fetchGgufModels,
    selectModel,
    toggleDir,
    updateSearchDirs,
}) {
    // ── Рендер списка моделей ──
    return (
        <div className="space-y-2">
            {/* ── Заголовок и кнопка обновления ── */}
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
            {/* ── Управление директориями поиска ── */}
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
            {/* ── Список доступных GGUF моделей ── */}
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
                        {expandedDirs[modelGroup.dir] && (
                        <div
                            className="overflow-hidden animate-fadeIn"
                        >
                                    {/* ── Список моделей в директории ── */}
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
                                </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
