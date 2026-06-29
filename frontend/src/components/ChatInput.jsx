// Область ввода: textarea, прикрепление файлов, кнопка отправки.
import { useRef } from 'react';

import { Send, Square, Image as ImageIcon, Plus, X as XIcon, RefreshCw } from 'lucide-react';
import { cn } from '../lib/utils';

export default function ChatInput({
    input,
    setInput,
    isLoading,
    imagePreview,
    removeImage,
    handleSend,
    handleImageChange,
    handlePaste,
    handleDragOver,
    handleDragLeave,
    isDragging,
    abortController,
    abortControllerRef,
    llmStatus,
    textareaRef,
}) {
    // ── Ссылки на DOM-элементы ──
    const fileInputRef = useRef(null);

    return (
        <>
            {/* Оверлей drag-and-drop */}
            {isDragging && (
                <div
                    className="absolute inset-0 z-50 flex items-center justify-center bg-primary/10 backdrop-blur-[2px] border-2 border-dashed border-primary m-4 rounded-3xl pointer-events-none animate-fadeIn"
                >
                    <div className="flex flex-col items-center gap-3 text-primary">
                        <Plus size={48} className="animate-pulse" />
                        <p className="font-bold text-lg">Отпустите, чтобы прикрепить фото</p>
                    </div>
                </div>
            )}

            <div className="px-6 pb-6">
                {/* ── Превью прикреплённого изображения ── */}
                {imagePreview && (
                    <div
                        className="mb-3 relative inline-block animate-fadeInUp"
                    >
                        <img src={imagePreview} alt="Preview" className="h-20 w-auto rounded-xl border border-primary/30 shadow-lg" />
                        <button
                            onClick={removeImage}
                            className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full shadow-md hover:scale-110 transition-transform"
                        >
                            <XIcon size={12} />
                        </button>
                    </div>
                )}
                <div className="relative">
                    {/* ── Поле ввода и прикрепление файлов ── */}
                    <div className="flex items-end gap-2 bg-muted/20 border border-border/50 rounded-xl p-2 pl-4">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleImageChange}
                            accept="image/*"
                            className="hidden"
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="p-3 text-muted-foreground hover:text-primary transition-colors"
                            title="Прикрепить фото"
                        >
                            <ImageIcon size={20} />
                        </button>
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onPaste={handlePaste}
                            onKeyDown={(e) => {
                                // ── Отправка по Enter без Shift ──
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            placeholder="Спросите что-нибудь об источниках..."
                            className="flex-1 bg-transparent border-none outline-none resize-none py-3 text-sm focus:ring-0 focus:outline-none overflow-y-auto"
                            rows={1}
                        />
                        {/* ── Кнопки действий: остановка, загрузка, отправка ── */}
                        {isLoading ? (
                            <button
                                onClick={() => { if (abortController) { abortController.abort(); abortControllerRef.current = null; } }}
                                className="p-3 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-all shadow-lg shadow-red-500/20"
                                title="Остановить генерацию"
                            >
                                <Square size={18} className="fill-current" />
                            </button>
                        ) : llmStatus.state === 'loading' ? (
                            <button
                                disabled
                                className="p-3 bg-blue-500/50 text-white rounded-xl cursor-not-allowed"
                                title="Дождитесь загрузки модели"
                            >
                                <RefreshCw size={18} className="animate-spin" />
                            </button>
                        ) : (
                            <button
                                onClick={handleSend}
                                disabled={!input.trim() && !imagePreview}
                                className={cn(
                                    "p-3 rounded-xl transition-all",
                                    (input.trim() || imagePreview) ? "bg-primary text-white shadow-lg shadow-primary/20" : "bg-muted text-muted-foreground"
                                )}
                                title="Отправить (Enter)"
                            >
                                <Send size={18} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
