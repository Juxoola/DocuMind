// SSE-стриминг: сборка ответа из чанков, обработка thinking, источников, ошибок.
import { useRef, useCallback } from 'react';

export default function useStreamingHandler({ messages, setMessages, llmSettings, setIsLoading, setStats, setAbortController, selectedSources, notebook, answerMode, thinkingMode, thinkingBudget, contextStrategy, maxTokens, input, setInput }) {
    const abortControllerRef = useRef(null);

    // ── Вспомогательная функция обновления AI-сообщения ──
    const updateAiMessage = useCallback((index, content, sources) => {
        setMessages(prev => {
            if (index >= prev.length + 2) return prev;
            const newMessages = [...prev];
            if (!newMessages[index]) return prev;
            newMessages[index] = { ...newMessages[index], content, sources, loading: false };
            return newMessages;
        });
    }, [setMessages]);

    // ── Формирование и отправка запроса на сервер ──
    const handleSend = useCallback(async () => {
        if (!input.trim() || input.trim() === '') return;
        if (selectedSources.length === 0) {
            alert('Выберите хотя бы один источник в боковой панели!');
            return;
        }

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');


        const currentInput = input;

        setMessages(prev => [...prev, {
            role: 'ai',
            content: '',
            thinkingContent: '',
            thinkingDone: false,
            loading: true,
            _meta: {
                model: llmSettings?.gguf_model_path || llmSettings?.llm_model || '',
                answer_mode: answerMode,
                thinking_mode: thinkingMode,
            },
        }]);

        const controller = new AbortController();
        abortControllerRef.current = controller;
        setAbortController(controller);
        setIsLoading(true);
        setStats(null);

        const aiMsgIndex = messages.length + 1;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    query: currentInput,
                    allowed_files: selectedSources,
                    max_tokens: maxTokens,
                    notebook_id: notebook.id,
                    thinking_mode: thinkingMode,
                    thinking_budget: thinkingBudget,
                    context_strategy: contextStrategy,
                    answer_mode: answerMode,
                    history: messages.slice(0, -1).filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({ role: m.role, content: m.content || '' })),
                    ...llmSettings
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';
            let thinkingContent = '';
            let buffer = '';

            let rafId = null;
            let lastRenderedContent = '';
            let lastRenderedThinking = '';
            const flushStreaming = () => {
                rafId = null;
                if (fullContent === lastRenderedContent && thinkingContent === lastRenderedThinking) return;
                lastRenderedContent = fullContent;
                lastRenderedThinking = thinkingContent;
                setMessages(prev => {
                    const next = prev.slice();
                    if (next[aiMsgIndex]) {
                        next[aiMsgIndex] = { ...next[aiMsgIndex], content: fullContent, thinkingContent, loading: false };
                    }
                    return next;
                });
            };
            const scheduleFlush = () => {
                if (rafId != null) return;
                rafId = requestAnimationFrame(flushStreaming);
            };
            const flushNow = () => {
                if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
                flushStreaming();
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                buffer = lines.pop() || '';

                // ── Обработка SSE-чанков: sources, thinking, content, ошибки ──
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;

                    const payload = trimmedLine.slice(6);
                    if (payload === '[DONE]') break;

                    try {
                        const data = JSON.parse(payload);
                        if (data.type === 'sources') {
                            setMessages(prev => {
                                const next = prev.slice();
                                if (next[aiMsgIndex]) next[aiMsgIndex] = { ...next[aiMsgIndex], sources: data.sources };
                                return next;
                            });
                        } else if (data.type === 'thinking_start') {
                            thinkingContent = '';
                            lastRenderedThinking = '';
                            setMessages(prev => {
                                const next = prev.slice();
                                if (next[aiMsgIndex]) next[aiMsgIndex] = { ...next[aiMsgIndex], loading: false, thinkingContent: '', thinkingDone: false };
                                return next;
                            });
                        } else if (data.type === 'thinking_chunk') {
                            thinkingContent += data.text;
                            scheduleFlush();
                        } else if (data.type === 'thinking_done') {
                            setMessages(prev => {
                                const next = prev.slice();
                                if (next[aiMsgIndex]) next[aiMsgIndex] = { ...next[aiMsgIndex], thinkingDone: true };
                                return next;
                            });
                        } else if (data.type === 'chunk') {
                            fullContent += data.text;
                            scheduleFlush();
                        } else if (data.type === 'stats') {
                            setStats(data);
                        } else if (data.type === 'error') {
                            fullContent = '⚠️ Ошибка: ' + data.text;
                            flushNow();
                        }
                    } catch (e) {
                        console.error('Ошибка парсинга SSE:', e, payload);
                    }
                }
            }
            flushNow();
        } catch (err) {
            if (err.name === 'AbortError') {
                updateAiMessage(aiMsgIndex, '*(Генерация остановлена)*', []);
            } else {
                updateAiMessage(aiMsgIndex, '⚠️ Ошибка связи с сервером.', []);
            }
        } finally {
            setIsLoading(false);
            setAbortController(null);
            abortControllerRef.current = null;
        }
    }, [input, messages, selectedSources, notebook, llmSettings, answerMode, thinkingMode, thinkingBudget, contextStrategy, maxTokens, setMessages, setIsLoading, setStats, setAbortController, updateAiMessage, setInput]);

    return { handleSend, updateAiMessage, abortControllerRef };
}
