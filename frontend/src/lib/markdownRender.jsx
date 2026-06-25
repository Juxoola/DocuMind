// Рендер markdown-сообщений LLM: цитаты [N], LaTeX, блоки кода.
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { CitationButton, CitationTooltipPortal } from './CitationTooltip';

// Рендер markdown-сообщений LLM: [N] → кликабельные цитаты, LaTeX, блоки кода
export function preProcessMessage(text) {
  if (!text) return '';

  let processed = text
    .replace(/\\\[[\s\S]*?\\\]/g, '$$$$$1$$$$')
    .replace(/\\\([\s\S]*?\\\)/g, '$$$1$$');

  const parts = processed.split(/(```[\s\S]*?```)/g);

  const finalParts = parts.map(part => {
    if (part.startsWith('```')) return part;

    const mathBlocks = [];
    let subPart = part.replace(/(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)/g, (m) => {
      mathBlocks.push(m);
      return `%%MATH_${mathBlocks.length - 1}%%`;
    });

    subPart = subPart.replace(
      /(?<!\\in|\\subset|\\subseteq|\\supset)\[(\d+(?:,\s*\d+)*)\]/g,
      (match, nums) => nums.split(',').map(n => `[${n.trim()}](#cite:${n.trim()})`).join('')
    );

    return subPart.replace(/%%MATH_(\d+)%%/g, (_, idx) => mathBlocks[parseInt(idx)]);
  });

  return finalParts.join('');
}

export function LlmMarkdown({ text, sources = [], onCite, className = '' }) {
  const [hovered, setHovered] = useState(null);
  const timeoutRef = useRef(null);

  const cancelClose = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };
  const scheduleClose = () => {
    cancelClose();
    timeoutRef.current = setTimeout(() => setHovered(null), 120);
  };

  useEffect(() => {
    if (!hovered) return;
    const onKey = (e) => { if (e.key === 'Escape') setHovered(null); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [hovered]);

  return (
    <>
      <div className={`prose prose-invert prose-sm max-w-none ${className}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith('#cite:')) {
                const n = parseInt(href.split(':')[1], 10);
                const src = sources[n - 1];
                if (!src) {
                  return <span className="text-muted-foreground opacity-50 whitespace-nowrap">[{n}]</span>;
                }
                return (
                  <CitationButton
                    n={n}
                    src={src}
                    onClick={(s) => onCite && onCite(n, s)}
                    onHover={(s, coords) => {
                      cancelClose();
                      setHovered({ src: s, x: coords.x, y: coords.y, _n: n });
                    }}
                    onLeave={scheduleClose}
                  />
                );
              }
              return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
            },
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              const lang = match ? match[1] : '';
              if (!inline && match) {
                return (
                  <div className="relative group my-3 rounded-xl overflow-hidden bg-[#0d1117] border border-white/5">
                    <div className="flex items-center justify-between px-3 py-1.5 opacity-40 group-hover:opacity-100 transition-opacity">
                      <span className="text-[9px] font-bold text-white/50 uppercase tracking-[0.2em]">{lang}</span>
                      <button
                        onClick={() => navigator.clipboard?.writeText(String(children).replace(/\n$/, ''))}
                        className="text-[9px] font-bold text-white/50 hover:text-white transition-colors"
                      >
                        COPY
                      </button>
                    </div>
                    <pre className="px-4 pb-3 pt-0 overflow-x-auto custom-scrollbar text-xs font-mono leading-relaxed">
                      <code className={className} {...props}>{children}</code>
                    </pre>
                  </div>
                );
              }
              return <code className="bg-white/10 px-1 py-0.5 rounded text-indigo-300 font-mono text-[0.85em]" {...props}>{children}</code>;
            },
          }}
        >
          {preProcessMessage(text)}
        </ReactMarkdown>
      </div>

      <CitationTooltipPortal
        hoveredSource={hovered}
        onClose={() => setHovered(null)}
        onCancelClose={cancelClose}
        onResumeClose={scheduleClose}
      />
    </>
  );
}
