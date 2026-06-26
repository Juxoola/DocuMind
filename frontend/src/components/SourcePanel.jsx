// Панель статистики: время, токены, использование контекста + тултип цитат.
import React from 'react';
import { Clock, Zap, FileText } from 'lucide-react';
import { cn } from '../lib/utils';
import { CitationTooltipPortal } from '../lib/CitationTooltip';

export default function SourcePanel({ stats, contextUsage, hoveredSource, tooltipCoords, tooltipTimeoutRef, setHoveredSource }) {
    return (
        <>
            {/* Строка статистики */}
            <div className="px-6">
                <div className="flex items-center justify-between gap-4 flex-wrap mb-3">
                    <div className="flex items-center gap-3 flex-wrap">
                        {stats && (
                            <>
                                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-muted/30 border border-border/30 text-[11px] font-bold text-muted-foreground">
                                    <Clock size={12} className="text-blue-400" />
                                    <span className="text-blue-400">{stats.elapsed_sec}с</span>
                                </span>
                                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-muted/30 border border-border/30 text-[11px] font-bold text-muted-foreground">
                                    <FileText size={12} className="text-emerald-400" />
                                    <span className="text-emerald-400">~{stats.total_tokens}</span>
                                    <span className="text-muted-foreground/60">ткн</span>
                                </span>
                                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-muted/30 border border-border/30 text-[11px] font-bold text-muted-foreground">
                                    <Zap size={12} className={stats.tokens_per_sec >= 20 ? "text-yellow-400" : stats.tokens_per_sec >= 10 ? "text-orange-400" : "text-red-400"} />
                                    <span className={stats.tokens_per_sec >= 20 ? "text-yellow-400" : stats.tokens_per_sec >= 10 ? "text-orange-400" : "text-red-400"}>{stats.tokens_per_sec}</span>
                                    <span className="text-muted-foreground/60">т/с</span>
                                </span>
                            </>
                        )}
                    </div>
                    {contextUsage && contextUsage.total > 0 && (
                        <div className="flex items-center gap-3 flex-1 max-w-[280px]">
                            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                    className={cn(
                                        "h-full rounded-full transition-all duration-500",
                                        contextUsage.pct > 80 ? "bg-red-500" : contextUsage.pct > 50 ? "bg-yellow-500" : "bg-primary"
                                    )}
                                    style={{ width: `${Math.min(contextUsage.pct, 100)}%` }}
                                />
                            </div>
                            <span className="text-[10px] font-bold tabular-nums text-muted-foreground/70 whitespace-nowrap">
                                {contextUsage.used >= 1000 ? `${(contextUsage.used / 1000).toFixed(1)}k` : contextUsage.used}
                                <span className="text-muted-foreground/40"> / {contextUsage.total >= 1000 ? `${(contextUsage.total / 1000).toFixed(1)}k` : contextUsage.total}</span>
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {/* Тултип цитат */}
            <CitationTooltipPortal
                hoveredSource={hoveredSource && tooltipCoords.x ? { src: hoveredSource, ...tooltipCoords } : null}
                onClose={() => setHoveredSource(null)}
                onCancelClose={() => clearTimeout(tooltipTimeoutRef.current)}
                onResumeClose={() => {
                    tooltipTimeoutRef.current = setTimeout(() => setHoveredSource(null), 100);
                }}
            />
        </>
    );
}
