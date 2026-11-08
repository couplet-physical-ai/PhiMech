/* ============================================================================================
   A small markdown renderer for PhiMech's own documentation.

   It lives in its own module for one reason: it is the riskiest code on the docs page, and a
   pure function with no DOM can be tested. `tests/test_home_page.py` runs it under node against
   every real doc in `docs/` and fails if any page renders to something suspicious.

   Rendering happens in the browser, not the server, because `phimech/viewer/server.py` must stay
   stdlib-only (the core boundary) -- and because it keeps the file on disk as the single source of
   truth, so no build step can make the served docs disagree with the repository.

   It covers the subset our docs actually use: ATX headings, fenced code, tables, lists,
   blockquotes, rules, links, inline code, bold and italic. It is NOT CommonMark and does not
   pretend to be. Anything it does not understand is passed through as escaped text, so an
   unsupported construct is visible rather than silently dropped.

   Security: input is HTML-escaped BEFORE any markup is added, so a document cannot inject an
   element. Inline patterns then operate on the escaped string.
   ============================================================================================ */

const escapeHtml = (text) => text
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

const slugify = (text) => text.toLowerCase().replace(/[^\w]+/g, '-').replace(/(^-|-$)/g, '');

/** Allow only URLs that cannot execute.
 *
 *  Escaping the text is not enough on its own: `[x](javascript:alert(1))` produces a perfectly
 *  well-formed anchor whose href runs code on click. Anything that is not an explicit safe scheme,
 *  a fragment, or a relative path is refused and rendered inert.
 */
function safeUrl(href) {
  const trimmed = href.trim();
  // A scheme is only a scheme if the colon comes before any /, ? or #.
  const scheme = trimmed.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):/);
  if (!scheme) return trimmed;                       // relative, fragment or protocol-relative
  return ['http', 'https', 'mailto'].includes(scheme[1].toLowerCase()) ? trimmed : '';
}

/** Inline spans, applied to already-escaped text. Order matters: code first, so nothing else
 *  rewrites the inside of a code span. */
function inline(text) {
  const codes = [];
  let out = text.replace(/`([^`]+)`/g, (_, code) => {
    codes.push(code);
    return '\u0000' + (codes.length - 1) + '\u0000';
  });
  out = out
    .replace(/!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/g, (whole, alt, src) => {
      const url = safeUrl(src);
      return url ? `<img src="${url}" alt="${alt}" loading="lazy" />` : whole;
    })
    .replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g, (whole, label, href) => {
      // Rewrite sibling .md links so they open in this reader rather than downloading a file.
      const internal = /^[^/#?:]+\.md(#.*)?$/.test(href) || /^\.\.?\//.test(href);
      if (internal) {
        return `<a href="?doc=${encodeURIComponent(href.replace(/^\.\//, ''))}">${label}</a>`;
      }
      const url = safeUrl(href);
      // A refused URL keeps its label but loses the link: visibly inert, never silently dropped.
      return url ? `<a href="${url}" rel="noopener">${label}</a>` : whole;
    })
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/(^|[\s(])_([^_\n]+)_/g, '$1<em>$2</em>');
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

const splitRow = (line) => line.replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim());

/** Markdown -> HTML, plus the headings found, for the on-page outline. */
function render(markdown) {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const html = [];
  const headings = [];
  let i = 0;

  const isTableDivider = (line) => /^\|?[\s:-]+\|[\s|:-]*$/.test(line) && line.includes('-');

  while (i < lines.length) {
    const line = lines[i];

    // --- fenced code -----------------------------------------------------------------------
    const fence = line.match(/^\s*```+\s*([\w-]*)/);
    if (fence) {
      const body = [];
      i += 1;
      while (i < lines.length && !/^\s*```+\s*$/.test(lines[i])) { body.push(lines[i]); i += 1; }
      i += 1;
      const lang = fence[1] ? ` data-lang="${escapeHtml(fence[1])}"` : '';
      html.push(`<pre${lang}><code>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }

    // --- heading ---------------------------------------------------------------------------
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2].replace(/\s*#+\s*$/, '');
      const id = slugify(text);
      if (level <= 3) headings.push({ level, text: text.replace(/[`*_]/g, ''), id });
      html.push(`<h${level} id="${id}">${inline(escapeHtml(text))}</h${level}>`);
      i += 1;
      continue;
    }

    // --- table -----------------------------------------------------------------------------
    if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      const th = head.map((c) => `<th>${inline(escapeHtml(c))}</th>`).join('');
      const tb = rows.map((r) =>
        `<tr>${r.map((c) => `<td>${inline(escapeHtml(c))}</td>`).join('')}</tr>`).join('');
      html.push(`<div class="md-table"><table><thead><tr>${th}</tr></thead>` +
                `<tbody>${tb}</tbody></table></div>`);
      continue;
    }

    // --- horizontal rule -------------------------------------------------------------------
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { html.push('<hr />'); i += 1; continue; }

    // --- blockquote ------------------------------------------------------------------------
    if (/^\s*>/.test(line)) {
      const body = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i].replace(/^\s*>\s?/, ''));
        i += 1;
      }
      html.push(`<blockquote>${inline(escapeHtml(body.join(' ')))}</blockquote>`);
      continue;
    }

    // --- list (one level of nesting, which is all our docs use) ----------------------------
    if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && (/^\s*([-*+]|\d+\.)\s+/.test(lines[i]) ||
                                  (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (/^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
          const indent = lines[i].match(/^\s*/)[0].length;
          items.push({ indent, text: lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, '') });
        } else {
          items[items.length - 1].text += ' ' + lines[i].trim();   // lazy continuation
        }
        i += 1;
      }
      const base = Math.min(...items.map((it) => it.indent));
      const parts = [];
      let nested = false;
      items.forEach((it) => {
        const deep = it.indent > base;
        if (deep && !nested) { parts.push('<ul>'); nested = true; }
        if (!deep && nested) { parts.push('</ul>'); nested = false; }
        parts.push(`<li>${inline(escapeHtml(it.text))}</li>`);
      });
      if (nested) parts.push('</ul>');
      const tag = ordered ? 'ol' : 'ul';
      html.push(`<${tag}>${parts.join('')}</${tag}>`);
      continue;
    }

    // --- blank -----------------------------------------------------------------------------
    if (!line.trim()) { i += 1; continue; }

    // --- paragraph: gather until something structural starts -------------------------------
    const para = [];
    while (i < lines.length && lines[i].trim() &&
           !/^(#{1,6}\s|\s*```|\s*>|\s*([-*+]|\d+\.)\s)/.test(lines[i]) &&
           !(lines[i].includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1]))) {
      para.push(lines[i]);
      i += 1;
    }
    html.push(`<p>${inline(escapeHtml(para.join(' ')))}</p>`);
  }
  return { html: html.join('\n'), headings };
}

export { escapeHtml, slugify, inline, render };
