import { useEffect, useRef, useState, useCallback } from 'react';

// Хук для горизонтального ресайза с гарантированным cleanup при unmount

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
