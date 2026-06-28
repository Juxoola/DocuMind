// Копирование в буфер: plain text, rich text (HTML), fallback через textarea.

// ── Генерация HTML-документа для Word-совместимого копирования ──
export function buildWordHtml(bodyHtml) {
  return `<!DOCTYPE html><html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><title>Exported</title><style>
body{font-family:Calibri,Arial,sans-serif;font-size:11pt;color:#000;line-height:1.5;margin:24pt;}
h1,h2,h3,h4,h5,h6{font-weight:bold;margin:14pt 0 8pt;line-height:1.25;}
h1{font-size:18pt;}h2{font-size:15pt;}h3{font-size:13pt;}h4{font-size:12pt;}
p{margin:0 0 8pt;}
ul,ol{margin:0 0 8pt;padding-left:24pt;}li{margin-bottom:3pt;}
strong{font-weight:bold;}em{font-style:italic;}
code{font-family:Consolas,monospace;font-size:10pt;background:#f4f4f4;padding:1pt 3pt;border-radius:2pt;}
pre{font-family:Consolas,monospace;font-size:10pt;background:#f4f4f4;padding:8pt;border-radius:4pt;margin:0 0 8pt;white-space:pre-wrap;}
blockquote{border-left:3pt solid #bbb;margin:0 0 8pt;padding:0 0 0 10pt;color:#444;}
a{color:#0563C1;text-decoration:underline;}
hr{border:0;border-top:1pt solid #ccc;margin:12pt 0;}
table{border-collapse:collapse;margin:0 0 8pt;}th,td{border:1pt solid #999;padding:4pt 8pt;}
</style></head><body>${bodyHtml}</body></html>`;
}

export async function copyAsRichText({ html, text }) {
  if (!html && !text) return false;
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.write && typeof ClipboardItem !== 'undefined') {
      const htmlBlob = new Blob([buildWordHtml(html || '')], { type: 'text/html' });
      const textBlob = new Blob([text || ''], { type: 'text/plain' });
      await navigator.clipboard.write([
        new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob }),
      ]);
      return true;
    }
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text || '');
      return true;
    }
  } catch (e) {
    console.error('copyAsRichText failed, falling back:', e);
  }
  if (typeof document !== 'undefined' && text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch {}
    document.body.removeChild(ta);
    return true;
  }
  return false;
}

// ── Извлечение чистого контента из DOM-элемента ──
export function extractCleanContent(rootEl) {
  if (!rootEl) return null;
  const prose = rootEl.querySelector?.('.prose') || rootEl;
  if (!prose) return null;
  const clone = prose.cloneNode(true);
  clone.querySelectorAll('[data-copy-skip="1"]').forEach((el) => {
    const prev = el.previousSibling;
    if (prev && prev.nodeType === Node.TEXT_NODE) {
      const t = prev.textContent;
      if (/^[ \t]*[,;:.][ \t]*$/.test(t)) {
        prev.textContent = '';
      } else {
        prev.textContent = t.replace(/[ \t]+$/, '');
      }
    }
    el.remove();
  });
  return {
    html: clone.innerHTML,
    text: (clone.textContent || '').replace(/\u00a0/g, ' ').trim(),
  };
}
