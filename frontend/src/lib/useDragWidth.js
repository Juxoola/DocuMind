import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * useDragWidth — custom hook для горизонтального ресайзера.
 *
 * F-fix #29: оригинальный код в MainApp.jsx добавлял mousemove/mouseup
 * на document в onMouseDown, и cleanup был в onMouseUp. Но если компонент
 * unmount-ился во время drag (смена блокнота, закрытие вкладки), listeners
 * оставались висеть на document. При повторном клике мог сработать stale
 * setSidebarWidth с замыканием на старый компонент.
 *
 * Этот хук использует ref-ы для listeners и гарантированно снимает их
 * при unmount компонента через useEffect cleanup.
 *
 * @param {Object} opts
 * @param {number} opts.initial - начальная ширина в px
 * @param {number} opts.min - минимальная ширина в px
 * @param {number} opts.max - максимальная ширина в px
 * @returns {[number, (e: MouseEvent) => void]} [width, onMouseDownHandler]
 */
export function useDragWidth({ initial = 320, min = 240, max = 600 } = {}) {
    const [width, setWidth] = useState(initial);
    const cleanupRef = useRef(null);

    useEffect(() => {
        return () => {
            if (cleanupRef.current) {
                cleanupRef.current();
                cleanupRef.current = null;
            }
        };
    }, []);

    const onMouseDown = useCallback((e) => {
        e.preventDefault();
        const onMouseMove = (ev) => {
            const newWidth = ev.clientX;
            if (newWidth > min && newWidth < max) {
                setWidth(newWidth);
            }
        };
        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            cleanupRef.current = null;
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        cleanupRef.current = onMouseUp;
    }, [min, max]);

    return [width, onMouseDown];
}
