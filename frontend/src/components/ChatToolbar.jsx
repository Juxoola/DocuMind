// Панель инструментов чата: слайдеры параметров, выбор режима ответа.
import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

import { ChevronDown, SlidersHorizontal, Sparkles, Zap, FileText, Check, ListChecks, ListOrdered, AlignLeft, Scale, GraduationCap, Smile } from 'lucide-react';
import { cn } from '../lib/utils';

// Слайдер настроек — переиспользуется в панели параметров
const SleekSlider = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  icon: Icon,
  presets = [],
  suffix = "",
  description,
  accentColor = "var(--color-primary)",
  disabled = false,
  showUnlimited = false,
  unlimitedChecked = false,
  onUnlimitedChange = null
}) => {
  const [inputValue, setInputValue] = useState(value);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleSliderChange = (e) => {
    if (disabled) return;
    const val = parseInt(e.target.value);
    setInputValue(val);
    onChange(val);
  };

  const handleInputChange = (e) => {
    if (disabled) return;
    let val = e.target.value === '' ? '' : parseInt(e.target.value);
    setInputValue(val);
  };

  const handleInputBlur = () => {
    if (disabled) return;
    let val = parseInt(inputValue);
    if (isNaN(val)) val = min;
    if (val < min) val = min;
    if (val > max) val = max;
    setInputValue(val);
    onChange(val);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleInputBlur();
      e.target.blur();
    }
  };

  const pct = disabled ? 0 : ((value - min) / (max - min)) * 100;

  return (
    <div className={cn(
      "flex flex-col gap-2.5 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all duration-300",
      disabled && "opacity-60"
    )}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 flex-wrap sm:flex-nowrap min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            {Icon && <Icon className="text-muted-foreground shrink-0" size={14} />}
            <span className="text-xs font-bold text-foreground/90 leading-tight">{label}</span>
          </div>
          {showUnlimited && (
            <label className={cn(
              "flex items-center gap-1.5 cursor-pointer select-none text-[9px] font-black uppercase px-2 py-0.5 rounded-md border transition-all pointer-events-auto whitespace-nowrap shrink-0",
              unlimitedChecked
                ? "bg-purple-500/10 border-purple-500/30 text-purple-400"
                : "bg-muted/40 border-border/40 text-muted-foreground hover:border-border/60 hover:text-foreground"
            )}>
              <input
                type="checkbox"
                checked={unlimitedChecked}
                onChange={(e) => onUnlimitedChange(e.target.checked)}
                className="hidden"
              />
              <span className="whitespace-nowrap">Без лимита</span>
            </label>
          )}
        </div>
        
        <div className="flex items-center gap-1.5 bg-muted/60 border border-border/40 rounded-lg px-2 py-0.5">
          <input
            type="number"
            value={disabled ? "" : inputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            onKeyDown={handleKeyDown}
            min={min}
            max={max}
            disabled={disabled}
            className="w-11 bg-transparent border-none text-[11px] font-black focus:ring-0 p-0 text-right appearance-none tuning-number-input"
            style={{ color: accentColor }}
          />
          <span className="text-[9px] font-bold text-muted-foreground uppercase">{disabled ? "∞" : suffix}</span>
        </div>
      </div>

      {description && (
        <p className="text-[9.5px] text-muted-foreground/80 leading-relaxed font-medium">
          {description}
        </p>
      )}

      <div className="relative mt-2.5 flex items-center">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={disabled ? max : value}
          onChange={handleSliderChange}
          disabled={disabled}
          className="tuning-slider"
          style={{
            background: `linear-gradient(to right, ${accentColor} 0%, ${accentColor} ${pct}%, var(--color-border) ${pct}%, var(--color-border) 100%)`
          }}
        />
      </div>

      {presets.length > 0 && (
        <div className="flex items-center justify-between mt-2.5 flex-wrap gap-1">
          {presets.map((p) => {
            const isSelected = !disabled && value === p.value;
            return (
              <button
                key={p.value}
                onClick={() => {
                  if (disabled) return;
                  setInputValue(p.value);
                  onChange(p.value);
                }}
                disabled={disabled}
                className={cn(
                  "px-2 py-0.5 text-[9px] font-extrabold rounded-md transition-all border",
                  isSelected
                    ? "bg-primary/10 border-primary/20 shadow-sm"
                    : "bg-transparent text-muted-foreground border-transparent hover:border-border/60 hover:text-foreground"
                )}
                style={isSelected ? { color: accentColor, backgroundColor: `${accentColor}12`, borderColor: `${accentColor}25` } : {}}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Режимы ответа — ключи и метки. Синхронизировано с config.ANSWER_MODES на бэкенде.
const ANSWER_MODE_OPTIONS = [
  { key: 'concise', label: 'Кратко + пояснение', description: 'Сначала прямой ответ, затем разбор по источникам', Icon: Zap, accent: 'text-amber-400' },
  { key: 'moderate', label: 'Умеренно', description: '2-4 абзаца: суть + ключевые детали, без статьи', Icon: Scale, accent: 'text-yellow-400' },
  { key: 'detailed', label: 'Развёрнуто', description: 'Сразу полный ответ в формате связной статьи', Icon: FileText, accent: 'text-blue-400' },
  { key: 'summary', label: 'TL;DR — 1-3 предложения', description: 'Только суть, без вступлений и пояснений', Icon: AlignLeft, accent: 'text-emerald-400' },
  { key: 'step_by_step', label: 'Пошагово', description: 'Нумерованная инструкция с пояснением к каждому шагу', Icon: ListOrdered, accent: 'text-purple-400' },
  { key: 'checklist', label: 'Чек-лист', description: 'Практический список действий с галочками [ ]', Icon: ListChecks, accent: 'text-rose-400' },
  { key: 'expert', label: 'Экспертный', description: 'Терминология, формулы, нюансы и edge-cases', Icon: GraduationCap, accent: 'text-indigo-400' },
  { key: 'eli5', label: 'Простым языком', description: 'Объяснение через бытовые аналогии, без терминов', Icon: Smile, accent: 'text-pink-400' },
];

const ANSWER_MODE_KEYS = ANSWER_MODE_OPTIONS.map(o => o.key);
const ANSWER_MODE_DEFAULT = ANSWER_MODE_KEYS[0];

const normalizeAnswerMode = (raw) =>
  ANSWER_MODE_KEYS.includes(raw) ? raw : ANSWER_MODE_DEFAULT;

// Селектор режима ответа
function AnswerModeSelect({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ top: 0, left: 0, width: 0 });
  const triggerRef = useRef(null);
  const menuRef = useRef(null);

  const current = ANSWER_MODE_OPTIONS.find(o => o.key === value) || ANSWER_MODE_OPTIONS[0];
  const CurrentIcon = current.Icon;

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => {
      if (triggerRef.current?.contains(e.target)) return;
      if (menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const recalc = () => {
      const r = triggerRef.current?.getBoundingClientRect();
      if (!r) return;
      setMenuPos({ top: r.bottom + 4, left: r.left, width: r.width });
    };
    recalc();
    window.addEventListener('resize', recalc);
    window.addEventListener('scroll', recalc, true);
    return () => {
      window.removeEventListener('resize', recalc);
      window.removeEventListener('scroll', recalc, true);
    };
  }, [open]);

  const handleSelect = (key) => {
    onChange(key);
    setOpen(false);
  };

  return (
    <>
      <button
        ref={triggerRef}
        onClick={() => setOpen(o => !o)}
        title={`Стиль ответа: ${current.label}. Кликните, чтобы сменить.`}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "px-3 py-1.5 text-[10px] font-black uppercase tracking-wider rounded-xl transition-all flex items-center gap-1.5 border",
          open
            ? "bg-primary/10 text-primary border-primary/30 shadow-sm"
            : "bg-muted/40 text-muted-foreground hover:text-foreground border-transparent hover:border-border/60"
        )}
      >
        <CurrentIcon size={11} className={current.accent} />
        <span className="hidden sm:inline">{current.label}</span>
        <span className="sm:hidden">{current.label.split(' ')[0]}</span>
        <ChevronDown size={10} className={cn("transition-transform", open && "rotate-180")} />
      </button>

      {open && createPortal(
          <div
            ref={menuRef}
            role="listbox"
            style={{
              position: 'fixed',
              top: menuPos.top,
              left: menuPos.left,
              minWidth: 280,
              maxWidth: 360,
            }}
            className="z-[100] bg-popover/95 backdrop-blur-md border border-border/60 rounded-xl shadow-2xl overflow-hidden animate-popIn"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="px-3 py-2 text-[9px] font-black uppercase tracking-widest text-muted-foreground/70 border-b border-white/5">
              Стиль ответа
            </div>
            {ANSWER_MODE_OPTIONS.map((opt) => {
              const Icon = opt.Icon;
              const selected = opt.key === value;
              return (
                <button
                  key={opt.key}
                  role="option"
                  aria-selected={selected}
                  onClick={() => handleSelect(opt.key)}
                  className={cn(
                    "w-full px-3 py-2.5 flex items-start gap-2.5 text-left transition-colors",
                    selected ? "bg-primary/15" : "hover:bg-muted/50"
                  )}
                >
                  <Icon size={14} className={cn("mt-0.5 shrink-0", opt.accent)} />
                  <div className="flex-1 min-w-0">
                    <div className={cn(
                      "text-[11px] font-bold leading-tight",
                      selected ? "text-foreground" : "text-foreground/90"
                    )}>
                      {opt.label}
                    </div>
                    <div className="text-[10px] text-muted-foreground/80 leading-snug mt-0.5">
                      {opt.description}
                    </div>
                  </div>
                  {selected && <Check size={12} className="mt-0.5 text-primary shrink-0" />}
                </button>
              );
            })}
          </div>,
        document.body
      )}
    </>
  );
}

// Панель параметров генерации (тоггл-секция)
export function TuningPanel({ isTuningOpen, maxTokens, setMaxTokens, thinkingMode, thinkingBudget, setThinkingBudget, contextStrategy, setContextStrategy }) {
  return (
    <div className={cn(
      "grid transition-all duration-250 ease-out border-b z-10",
      isTuningOpen ? "grid-rows-[1fr] border-border bg-card/25 backdrop-blur-md" : "grid-rows-[0fr] border-transparent"
    )}>
      <div className="overflow-hidden">
          <div className="p-5 flex flex-col gap-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <SleekSlider
                label="Максимальная длина ответа"
                value={maxTokens}
                onChange={setMaxTokens}
                min={100}
                max={32000}
                step={100}
                icon={null}
                suffix="t"
                accentColor="var(--color-primary)"
                description="Регулирует максимальный лимит токенов, генерируемых моделью за один ответ. Большие значения подходят для развернутых эссе."
                presets={[
                  { label: "512", value: 512 },
                  { label: "1k", value: 1024 },
                  { label: "2k", value: 2048 },
                  { label: "4k", value: 4096 },
                  { label: "8k", value: 8192 },
                  { label: "16k", value: 16384 },
                  { label: "32k", value: 32000 }
                ]}
              />

              <SleekSlider
                label="Бюджет рассуждений"
                value={thinkingBudget === -1 ? 32000 : thinkingBudget}
                onChange={(val) => {
                  if (thinkingBudget !== -1) {
                    setThinkingBudget(val);
                  }
                }}
                min={100}
                max={32000}
                step={100}
                icon={null}
                suffix="t"
                accentColor="rgb(168, 85, 247)"
                disabled={!thinkingMode || thinkingBudget === -1}
                showUnlimited={thinkingMode}
                unlimitedChecked={thinkingBudget === -1}
                onUnlimitedChange={(checked) => {
                  setThinkingBudget(checked ? -1 : 1024);
                }}
                description="Сколько токенов думающая модель (например, DeepSeek-R1) может потратить на внутренний процесс рассуждений. Отключение лимита дает модели свободу мысли."
                presets={[
                  { label: "512", value: 512 },
                  { label: "1k", value: 1024 },
                  { label: "2k", value: 2048 },
                  { label: "4k", value: 4096 },
                  { label: "8k", value: 8192 },
                  { label: "16k", value: 16384 }
                ]}
              />

              <div className="flex flex-col gap-2.5 p-4 rounded-xl border border-white/5 bg-white/[0.01] hover:bg-white/[0.03] transition-all duration-300 md:col-span-2">
                <div className="flex items-center gap-2">
                  <SlidersHorizontal className="text-primary" size={14} />
                  <span className="text-xs font-bold text-foreground/90">Стратегия при переполнении контекста</span>
                </div>
                <p className="text-[9.5px] text-muted-foreground/80 leading-relaxed font-medium">
                  Определяет поведение системы, когда размер входного контекста приближается к границам модели.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1.5">
                  <button
                    onClick={() => setContextStrategy('sliding')}
                    className={cn(
                      "flex flex-col text-left p-3.5 rounded-xl border transition-all duration-300",
                      contextStrategy === 'sliding'
                        ? "bg-primary/5 border-primary text-foreground shadow-sm"
                        : "bg-transparent border-border/40 text-muted-foreground hover:border-border/80"
                    )}
                  >
                    <span className="text-xs font-black flex items-center gap-1.5">
                      <span className={cn("w-1.5 h-1.5 rounded-full", contextStrategy === 'sliding' ? "bg-primary animate-pulse" : "bg-muted-foreground")} />
                      «Скользящее окно» (Sliding Window)
                    </span>
                    <span className="text-[9px] font-medium mt-1.5 opacity-80 leading-normal">
                      Автоматически вытесняет старые сообщения из истории чата, освобождая место для новых ответов. Держит RAG-источники в приоритете.
                    </span>
                  </button>

                  <button
                    onClick={() => setContextStrategy('rag_priority')}
                    className={cn(
                      "flex flex-col text-left p-3.5 rounded-xl border transition-all duration-300",
                      contextStrategy === 'rag_priority'
                        ? "bg-purple-500/5 border-purple-500 text-purple-300 shadow-sm"
                        : "bg-transparent border-border/40 text-muted-foreground hover:border-border/80"
                    )}
                  >
                    <span className="text-xs font-black flex items-center gap-1.5">
                      <span className={cn("w-1.5 h-1.5 rounded-full", contextStrategy === 'rag_priority' ? "bg-purple-500 animate-pulse" : "bg-muted-foreground")} />
                      «Приоритет документов» (Document Priority)
                    </span>
                    <span className="text-[9px] font-medium mt-1.5 opacity-80 leading-normal">
                      Жестко удерживает все прикрепленные фрагменты документов в контексте. При нехватке памяти история чата мгновенно очищается.
                    </span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
}

export { SleekSlider, AnswerModeSelect, ANSWER_MODE_OPTIONS, ANSWER_MODE_KEYS, ANSWER_MODE_DEFAULT, normalizeAnswerMode };
