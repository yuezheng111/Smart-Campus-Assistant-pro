/* ============================================================
   main.js — 交互层
   无框架、无构建：原生 DOM + fetch。
   ============================================================ */
(() => {
  'use strict';

  /* ---------------- 工具 ---------------- */

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const escapeHtml = (str) =>
    String(str ?? '').replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );

  const fmtDate = (value) => (value ? String(value).slice(0, 10) : '—');

  const escapeRe = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { detail: text };
    }
    if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
    return data;
  }

  /** 阶段名 → 芯片样式类
   *  注意：键必须加引号。中文可以作标识符，但「报名已截止，待举行」里的全角逗号
   *  不是合法标识符字符，不加引号会直接语法报错。 */
  const STAGE_CLASS = {
    '正在报名': 'chip--open',
    '未开始报名': 'chip--soon',
    '报名已截止，待举行': 'chip--closed',
    '已结束': 'chip--done',
  };

  const stageChip = (stage, extra = '') => {
    if (!stage) return '';
    const cls = STAGE_CLASS[stage] || 'chip--plain';
    return `<span class="chip ${cls}">${escapeHtml(stage)}${extra ? ' · ' + escapeHtml(extra) : ''}</span>`;
  };

  /* ============================================================
     轻量 Markdown 渲染
     只支持回答里真实会出现的语法，不引入外部库
     ============================================================ */

  /** 高亮哨兵：0x01/0x02 不会被 escapeHtml 转义，也不会出现在正文里，
   *  所以可以先在原文上打标记，再走 Markdown 渲染，最后一步才换成 <mark>。 */
  const HL_OPEN = '\u0001';
  const HL_CLOSE = '\u0002';

  function inline(text) {
    let out = escapeHtml(text);

    // 行内代码先抽出来，避免其中的 * 和 [ 被后续规则误伤
    const codes = [];
    out = out.replace(/`([^`]+)`/g, (_, code) => {
      codes.push(code);
      return `\u0000${codes.length - 1}\u0000`;
    });

    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, label, url) => `<a href="${url}" target="_blank" rel="noopener">${label}</a>`);

    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 引用标记 [资料N] → 可点击芯片
    out = out.replace(/\[资料\s*(\d+)\]/g,
      (_, n) => `<button class="cite" data-ref="${n}" type="button" title="跳到参考资料 ${n}">${n}</button>`);

    out = out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);

    out = out.replaceAll(HL_OPEN, '<mark class="hl">').replaceAll(HL_CLOSE, '</mark>');
    return out;
  }

  function renderMarkdown(src) {
    const lines = String(src ?? '').replace(/\r\n/g, '\n').split('\n');
    const html = [];
    let i = 0;

    const isTableLine = (line) => /^\s*\|.*\|\s*$/.test(line);
    const splitRow = (line) =>
      line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) { i += 1; continue; }

      // 表格
      if (isTableLine(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const head = splitRow(line);
        i += 2;
        const body = [];
        while (i < lines.length && isTableLine(lines[i])) { body.push(splitRow(lines[i])); i += 1; }
        html.push(
          `<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead>` +
          `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`
        );
        continue;
      }

      // 标题
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        const level = Math.min(h[1].length + 1, 4);
        html.push(`<h${level}>${inline(h[2])}</h${level}>`);
        i += 1;
        continue;
      }

      // 引用块
      if (/^\s*>\s?/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          buf.push(lines[i].replace(/^\s*>\s?/, ''));
          i += 1;
        }
        html.push(`<blockquote>${buf.map(inline).join('<br>')}</blockquote>`);
        continue;
      }

      // 无序列表
      if (/^\s*[-*+]\s+/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          buf.push(lines[i].replace(/^\s*[-*+]\s+/, ''));
          i += 1;
        }
        html.push(`<ul>${buf.map((t) => `<li>${inline(t)}</li>`).join('')}</ul>`);
        continue;
      }

      // 有序列表
      if (/^\s*\d+[.)]\s+/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          buf.push(lines[i].replace(/^\s*\d+[.)]\s+/, ''));
          i += 1;
        }
        html.push(`<ol>${buf.map((t) => `<li>${inline(t)}</li>`).join('')}</ol>`);
        continue;
      }

      // 段落
      const buf = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^\s*(#{1,4}\s|[-*+]\s|\d+[.)]\s|>)/.test(lines[i]) &&
        !isTableLine(lines[i])
      ) {
        buf.push(lines[i]);
        i += 1;
      }
      if (buf.length) html.push(`<p>${buf.map(inline).join('<br>')}</p>`);
      else i += 1;
    }

    return html.join('');
  }

  /** 在原文里标出被引用的那一段，再交给 Markdown 渲染。
   *  片段里的空白已在入库时被压成了单空格，而原文保留着换行与缩进，
   *  所以两边的空白要当作等价处理。 */
  function markSnippet(content, snippet) {
    const key = String(snippet || '').trim();
    if (key.length < 12) return content;

    try {
      const pattern = key.split(/\s+/).map(escapeRe).join('[\\s\\u3000]*');
      const match = new RegExp(pattern).exec(content);
      if (!match) return content;
      return (
        content.slice(0, match.index) +
        HL_OPEN + match[0] + HL_CLOSE +
        content.slice(match.index + match[0].length)
      );
    } catch {
      return content;
    }
  }

  /* ============================================================
     顶栏状态 + 侧栏统计
     ============================================================ */

  async function loadHealth() {
    const dbDot = $('#dbDot'), dbText = $('#dbText');
    try {
      const data = await api('/health');
      const ok = data.MySQL === 'connected';
      dbDot.className = 'dot ' + (ok ? 'dot--ok' : 'dot--bad');
      dbText.textContent = ok ? '资料库已连接' : '资料库异常';

      const missing = data['缺失的APIKey'] || [];
      if (missing.length) pushAlert(`服务未配置完整，问答功能可能不可用。`);
      return data;
    } catch (err) {
      dbDot.className = 'dot dot--bad';
      dbText.textContent = '服务未启动';
      pushAlert(`无法连接服务：${err.message}`);
      return null;
    }
  }

  async function loadStats() {
    try {
      const s = await api('/api/stats');
      $('#statDocs').textContent = s['文件数'];
      $('#statComps').textContent = s['竞赛记录'];
      $('#statExams').textContent = s['考试记录'];
    } catch { /* 统计失败不影响主流程 */ }
  }

  function pushAlert(message) {
    const box = document.createElement('div');
    box.className = 'alert';
    box.style.margin = 'var(--space-4) var(--space-5) 0';
    box.innerHTML = `<strong>!</strong><span>${escapeHtml(message)}</span>`;
    $('#main').prepend(box);
    setTimeout(() => box.remove(), 12000);
  }

  /* ============================================================
     导航
     ============================================================ */

  const viewLoaded = {};
  const VIEW_LOADERS = {};

  function initNav() {
    const nav = $('#nav');
    nav.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-nav]');
      if (!btn) return;
      const target = btn.dataset.nav;

      $$('.nav__item', nav).forEach((item) =>
        item.setAttribute('aria-current', String(item === btn)));
      $$('.view').forEach((view) =>
        view.setAttribute('data-active', String(view.dataset.view === target)));

      // 首次进入某个视图时才拉数据，避免无意义的请求
      const loader = VIEW_LOADERS[target];
      if (loader && !viewLoaded[target]) {
        viewLoaded[target] = true;
        loader();
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ============================================================
     原文阅读器
     ============================================================ */

  let readerText = '';   // 当前打开文件的全文，复制用
  let readerId = null;   // 用于下载原件

  function showReader() {
    $('#modal').dataset.open = 'true';
    $('#modal').hidden = false;
    $('#modalClose').focus();
  }

  function hideReader() {
    $('#modal').dataset.open = 'false';
    $('#modal').hidden = true;
    readerText = '';
    readerId = null;
  }

  /** 打开某份文件的原文。source 里带 id 才能取到全文，否则退回只显示片段。 */
  async function openReader(source) {
    const title = source.title || '文件原文';
    const meta = [source.department, source.doc_type,
      source.college && source.college !== '全校' ? source.college : null,
      source.publish_date ? fmtDate(source.publish_date) : null]
      .filter(Boolean);

    $('#modalTitle').textContent = title;
    $('#modalMeta').textContent = meta.join(' · ');
    $('#modalCopy').hidden = true;
    readerText = '';
    readerId = source.id || null;

    const download = $('#modalDownload');
    download.hidden = !readerId;

    if (!readerId) {
      $('#modalBody').innerHTML =
        `<div class="skeleton"><div class="skeleton__row"></div><div class="skeleton__row"></div><div class="skeleton__row"></div></div>`;
      showReader();
      $('#modalBody').innerHTML = source.snippet
        ? `<p class="reader__warn">未能定位到这份文件的详情记录，以下是回答参考到的片段：</p>` +
          `<blockquote>${escapeHtml(source.snippet)}</blockquote>`
        : `<p class="reader__warn">未能定位到这份文件的详情记录。</p>`;
      return;
    }

    $('#modalBody').innerHTML =
      `<div class="skeleton"><div class="skeleton__row"></div><div class="skeleton__row"></div><div class="skeleton__row"></div></div>`;
    showReader();

    try {
      const data = await api(`/api/documents/${readerId}/raw`);
      readerText = data.content || '';
      $('#modalBody').innerHTML = readerText
        ? renderMarkdown(markSnippet(readerText, source.snippet))
        : `<p class="reader__warn">这份文件没有可显示的正文。</p>`;
      if (data.truncated) {
        $('#modalBody').insertAdjacentHTML(
          'beforeend',
          `<p class="reader__warn">（文件较长，此处只显示前 ${data.content.length} 字）</p>`
        );
      }
      $('#modalCopy').hidden = !readerText;
    } catch (err) {
      $('#modalBody').innerHTML =
        `<p class="reader__warn">原文读取失败：${escapeHtml(err.message)}</p>`;
    }
  }

  function initReader() {
    $('#modalClose').addEventListener('click', hideReader);
    $('#modal').addEventListener('click', (e) => {
      if (e.target === $('#modal')) hideReader();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !$('#modal').hidden) hideReader();
    });

    $('#modalCopy').addEventListener('click', async () => {
      if (!readerText) return;
      const btn = $('#modalCopy');
      try {
        await navigator.clipboard.writeText(readerText);
        btn.textContent = '已复制';
      } catch {
        // 非安全上下文（http 局域网访问）下 Clipboard API 不可用，退回选中
        const range = document.createRange();
        range.selectNodeContents($('#modalBody'));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = '已选中';
      }
      setTimeout(() => { btn.textContent = '复制全文'; }, 1600);
    });

    $('#modalDownload').addEventListener('click', () => {
      if (readerId) window.location.assign(`/api/documents/${readerId}/download`);
    });
  }

  /* ============================================================
     对话
     ============================================================ */

  const chatHistory = [];

  function appendUserMsg(text) {
    const log = $('#chatLog');
    const el = document.createElement('article');
    el.className = 'msg msg--user';
    el.innerHTML =
      '<div class="msg__who">我</div>' +
      `<div class="msg__body"><div class="msg__bubble">${escapeHtml(text)}</div></div>`;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  function appendAssistantShell() {
    const log = $('#chatLog');
    const el = document.createElement('article');
    el.className = 'msg msg--assistant';
    el.innerHTML =
      '<div class="msg__who">校园智助</div>' +
      '<div class="msg__body"><div class="answer">' +
      '<div class="skeleton">' +
      '<div class="skeleton__row"></div><div class="skeleton__row"></div><div class="skeleton__row"></div>' +
      '</div></div></div>';
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  /** 参考资料：挂在回答正下方，点开即读原文 */
  function refsBlock(sources) {
    const items = sources.map((s, i) => {
      const ref = s.ref ?? i + 1;
      const meta = [fmtDate(s.publish_date), s.department, s.doc_type]
        .filter(Boolean).join(' · ');
      return (
        `<button class="ref" type="button" data-ref="${ref}">` +
        `<span class="ref__n">${ref}</span>` +
        `<span class="ref__body">` +
        `<span class="ref__title">${escapeHtml(s.title)}</span>` +
        `<span class="ref__meta">${escapeHtml(meta)}</span>` +
        (s.snippet ? `<span class="ref__snippet">${escapeHtml(s.snippet)}</span>` : '') +
        `</span>` +
        `<span class="ref__go">读原文</span>` +
        `</button>`
      );
    }).join('');

    return (
      `<section class="refs">` +
      `<div class="refs__head">` +
      `<span class="refs__label">参考资料</span>` +
      `<span class="refs__n">${sources.length} 份学校文件</span>` +
      `</div>` +
      `<div class="refs__list">${items}</div>` +
      `</section>`
    );
  }

  function bindRefs(root, sources) {
    $$('.ref', root).forEach((btn) => {
      const ref = Number(btn.dataset.ref);
      const source = sources.find((s, i) => (s.ref ?? i + 1) === ref);
      if (!source) return;
      btn.addEventListener('click', () => openReader(source));
    });

    // 回答里的 [资料N] 芯片 → 滚到对应卡片并闪一下
    $$('.cite', root).forEach((btn) => {
      btn.addEventListener('click', () => {
        const card = $(`.ref[data-ref="${btn.dataset.ref}"]`, root);
        if (!card) return;
        card.scrollIntoView({ block: 'center', behavior: 'smooth' });
        card.classList.add('is-flash');
        setTimeout(() => card.classList.remove('is-flash'), 1400);
      });
    });
  }

  async function sendQuestion(question) {
    const q = String(question || '').trim();
    if (!q) return;

    const sendBtn = $('#sendBtn');
    const composer = $('#composer');
    sendBtn.disabled = true;
    sendBtn.classList.add('is-loading');
    composer.value = '';
    composer.style.height = 'auto';

    appendUserMsg(q);
    appendAssistantShell();
    $('#chatMeta').textContent = '正在查找…';

    const started = performance.now();
    try {
      const data = await api('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ question: q, history: chatHistory }),
      });
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);

      const shell = $('#chatLog').lastElementChild;
      const sources = data.sources || [];
      const body = $('.msg__body', shell);
      body.innerHTML =
        `<div class="answer">${renderMarkdown(data.answer)}</div>` +
        (sources.length ? refsBlock(sources) : '') +
        `<div class="msg__foot">用时 ${elapsed}s</div>`;

      bindRefs(body, sources);
      $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
      $('#chatMeta').textContent = `完成 · ${elapsed}s`;

      chatHistory.push({ role: 'user', content: q });
      chatHistory.push({ role: 'assistant', content: data.answer });
      // 客户端只保留最近 3 轮，避免请求体无限增长
      while (chatHistory.length > 6) chatHistory.shift();
    } catch (err) {
      const shell = $('#chatLog').lastElementChild;
      $('.msg__body', shell).innerHTML =
        `<div class="alert"><strong>!</strong><span>回答失败：${escapeHtml(err.message)}</span></div>`;
      $('#chatMeta').textContent = '失败';
    } finally {
      sendBtn.disabled = false;
      sendBtn.classList.remove('is-loading');
      composer.focus();
    }
  }

  function initChat() {
    const composer = $('#composer');
    const send = () => sendQuestion(composer.value);

    $('#sendBtn').addEventListener('click', send);

    composer.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    // 输入框随内容长高
    composer.addEventListener('input', () => {
      composer.style.height = 'auto';
      composer.style.height = Math.min(composer.scrollHeight, 168) + 'px';
    });

    $('#quickChips').addEventListener('click', (e) => {
      const chip = e.target.closest('[data-q]');
      if (!chip) return;
      composer.value = chip.dataset.q;
      sendQuestion(chip.dataset.q);
    });

    $('#clearChat').addEventListener('click', () => {
      chatHistory.length = 0;
      $('#chatLog').innerHTML =
        '<p class="empty__text" style="color: var(--color-muted)">已清空。直接提问开始新的对话。</p>';
      $('#chatMeta').textContent = '就绪';
    });

    $('#chatLog').innerHTML =
      '<p class="empty__text" style="color: var(--color-muted)">从上面挑一个问题，或直接在下面输入。</p>';
  }

  /* ============================================================
     文件检索
     ============================================================ */

  const skeleton = (rows = 3) =>
    `<div class="skeleton">${Array.from({ length: rows }, () => '<div class="skeleton__row"></div>').join('')}</div>`;

  const emptyState = (mark, text) =>
    `<div class="empty"><div class="empty__mark">${mark}</div><p class="empty__text">${escapeHtml(text)}</p></div>`;

  function docCard(doc, score) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'doc-card';
    const pct = score != null ? Math.round(Math.max(0, Math.min(1, score)) * 100) : null;
    // 语义检索结果带 snippet（命中的原文片段），列表查询带 summary（文档摘要）
    const excerpt = doc.snippet || doc.summary || '（暂无摘要）';
    card.innerHTML =
      `<div class="doc-card__top">
         <span class="chip chip--plain">${escapeHtml(doc.doc_type || '其他')}</span>
         ${doc.college && doc.college !== '全校' ? `<span class="chip chip--plain">${escapeHtml(doc.college)}</span>` : ''}
       </div>
       <div class="doc-card__title">${escapeHtml(doc.title)}</div>
       <div class="doc-card__sum">${escapeHtml(excerpt)}</div>
       ${pct != null ? `<div class="score-bar"><i style="width:${pct}%"></i></div>` : ''}
       <div class="doc-card__foot">
         <span>${escapeHtml(fmtDate(doc.publish_date))}</span>
         <span>${escapeHtml(doc.department || '—')}${pct != null ? ` · 相关度 ${pct}%` : ''}</span>
       </div>`;
    card.addEventListener('click', () => openReader(doc));
    return card;
  }

  async function loadDocuments() {
    const box = $('#docsResult');
    const q = $('#docsQ').value.trim();
    const params = new URLSearchParams();

    box.innerHTML = skeleton(4);
    $('#docsMeta').textContent = '检索中…';

    try {
      let payload;
      if (q) {
        params.set('q', q);
        params.set('doc_type', $('#docsType').value);
        params.set('college', $('#docsCollege').value);
        payload = await api(`/api/search?${params}`);
      } else {
        params.set('doc_type', $('#docsType').value);
        params.set('college', $('#docsCollege').value);
        params.set('department', $('#docsDept').value);
        params.set('limit', '30');
        payload = await api(`/api/documents?${params}`);
      }

      const docs = payload.documents || payload['文件清单'] || [];
      box.innerHTML = '';
      $('#docsMeta').textContent = q
        ? `语义检索 · ${docs.length} 份${payload.fallback ? ' · 已放宽条件' : ''}`
        : `共 ${docs.length} 份`;

      if (!docs.length) {
        box.innerHTML = emptyState('0', '没有符合条件的文件。试试换关键词，或把筛选条件放宽。');
        return;
      }

      const grid = document.createElement('div');
      grid.className = 'card-grid';
      docs.forEach((doc) => grid.appendChild(docCard(doc, doc.score)));
      box.appendChild(grid);
    } catch (err) {
      box.innerHTML = `<div style="padding:var(--space-5)"><div class="alert"><strong>!</strong><span>${escapeHtml(err.message)}</span></div></div>`;
    }
  }

  function initDocs() {
    $('#docsForm').addEventListener('submit', (e) => {
      e.preventDefault();
      loadDocuments();
    });
    VIEW_LOADERS.docs = loadDocuments;
  }

  /* ============================================================
     竞赛
     ============================================================ */

  async function loadCompetitions() {
    const box = $('#compsResult');
    box.innerHTML = skeleton(5);
    $('#compsMeta').textContent = '查询中…';

    const params = new URLSearchParams({
      category: $('#compCat').value,
      level: $('#compLevel').value,
      year: $('#compYear').value || '0',
      grade: $('#compGrade').value,
      keyword: $('#compKw').value.trim(),
      limit: '50',
    });

    try {
      const data = await api(`/api/competitions?${params}`);
      const rows = data['竞赛列表'] || [];
      $('#compsMeta').textContent = `${rows.length} 条 · 基准日 ${fmtDate(data['今天日期'])}`;

      if (!rows.length) {
        box.innerHTML = emptyState('0', '没有匹配的竞赛。可以试试只保留「类别」这一个条件。');
        return;
      }

      box.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr>
          <th>竞赛名称</th><th>类别 / 级别</th><th>届次</th>
          <th>报名窗口</th><th>比赛时间</th><th>当前阶段</th>
        </tr></thead>
        <tbody>${rows.map((r) => `
          <tr>
            <td class="c-name">${escapeHtml(r.name)}<div class="c-dim">${escapeHtml(r.organizer || '')}</div></td>
            <td>${escapeHtml(r.category || '—')}<div class="c-dim">${escapeHtml(r.level || '')}</div></td>
            <td class="c-time">${r.year ?? '—'}</td>
            <td class="c-time">${fmtDate(r.signup_start)}<br>→ ${fmtDate(r.signup_end)}</td>
            <td class="c-time">${fmtDate(r.contest_start)}<br>→ ${fmtDate(r.contest_end)}</td>
            <td>${stageChip(r['阶段'])}</td>
          </tr>`).join('')}</tbody>
      </table></div>`;
    } catch (err) {
      box.innerHTML = `<div style="padding:var(--space-5)"><div class="alert"><strong>!</strong><span>${escapeHtml(err.message)}</span></div></div>`;
    }
  }

  function initCompetitions() {
    $('#compsForm').addEventListener('submit', (e) => {
      e.preventDefault();
      loadCompetitions();
    });
    VIEW_LOADERS.competitions = loadCompetitions;
  }

  /* ============================================================
     考试
     ============================================================ */

  async function loadExams() {
    const box = $('#examsResult');
    box.innerHTML = skeleton(5);
    $('#examsMeta').textContent = '查询中…';

    const params = new URLSearchParams({
      category: $('#examCat').value,
      year: $('#examYear').value || '0',
      keyword: $('#examKw').value.trim(),
      upcoming_only: $('#examUpcoming').value,
      limit: '50',
    });

    try {
      const data = await api(`/api/exams?${params}`);
      const rows = data['考试列表'] || [];
      $('#examsMeta').textContent = `${rows.length} 条 · 基准日 ${fmtDate(data['今天日期'])}`;

      if (!rows.length) {
        box.innerHTML = emptyState('0', '没有匹配的考试。提示：「四六级」会自动匹配到四级和六级两条记录。');
        return;
      }

      box.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr>
          <th>考试名称</th><th>类别</th><th>年份</th>
          <th>报名窗口</th><th>考试日期</th><th>当前阶段</th><th>报名费</th>
        </tr></thead>
        <tbody>${rows.map((r) => `
          <tr>
            <td class="c-name">${escapeHtml(r['名称'])}<div class="c-dim">${escapeHtml((r['说明'] || '').slice(0, 40))}</div></td>
            <td>${escapeHtml(r['类别'] || '—')}</td>
            <td class="c-time">${r['年份'] ?? '—'}</td>
            <td class="c-time">${fmtDate(r['报名开始'])}<br>→ ${fmtDate(r['报名截止'])}</td>
            <td class="c-time">${fmtDate(r['考试日期'])}</td>
            <td>${stageChip(r['阶段'])}</td>
            <td class="c-dim">${escapeHtml(r['报名费'] || '—')}</td>
          </tr>`).join('')}</tbody>
      </table></div>`;
    } catch (err) {
      box.innerHTML = `<div style="padding:var(--space-5)"><div class="alert"><strong>!</strong><span>${escapeHtml(err.message)}</span></div></div>`;
    }
  }

  function initExams() {
    $('#examsForm').addEventListener('submit', (e) => {
      e.preventDefault();
      loadExams();
    });
    VIEW_LOADERS.exams = loadExams;
  }

  /* ============================================================
     学业规划
     ============================================================ */

  async function loadPlanning() {
    const box = $('#planResult');
    box.innerHTML = skeleton(4);
    $('#planMeta').textContent = '生成中…';

    try {
      const data = await api('/api/recommend', {
        method: 'POST',
        body: JSON.stringify({
          grade: $('#planGrade').value,
          major: $('#planMajor').value.trim(),
          interests: $('#planInterests').value.trim(),
          limit: Number($('#planLimit').value) || 6,
        }),
      });

      const items = data['推荐列表'] || [];
      const dirs = (data['识别到的方向'] || []).join(' / ') || '未识别到明确方向（按通用推荐）';

      if (!items.length) {
        box.innerHTML = emptyState('0', '没有找到匹配的竞赛，试着把兴趣方向写得更具体一些。');
        $('#planMeta').textContent = '0 条';
        return;
      }

      $('#planMeta').textContent = `${items.length} 条 · 基准日 ${fmtDate(data['今天日期'])}`;
      box.innerHTML =
        `<div class="toolbar" style="border-bottom:1px solid var(--color-border)">
           <div class="field field--wide">
             <span class="field__label">识别到的方向</span>
             <span class="chip chip--plain">${escapeHtml(dirs)}</span>
           </div>
           <div class="field">
             <span class="field__label">推荐数量</span>
             <span class="chip chip--plain">${items.length}</span>
           </div>
         </div>
         <div class="card-grid">${items.map((it) => {
           const extra = it['距离报名截止天数'] != null
             ? `还剩 ${it['距离报名截止天数']} 天`
             : (it['距离报名开始天数'] != null ? `${it['距离报名开始天数']} 天后开放` : '');
           return `
           <article class="doc-card doc-card--static">
             <div class="doc-card__top">
               <span class="chip chip--plain">${escapeHtml(it['类别'] || '—')}</span>
               <span class="chip chip--plain">${escapeHtml(it['级别'] || '—')}</span>
               ${stageChip(it['阶段'], extra)}
             </div>
             <h3 class="doc-card__title">${escapeHtml(it['名称'])}</h3>
             <p class="doc-card__sum">${escapeHtml(it['简介'] || '')}</p>
             <dl class="rec__facts">
               <div>报名 <span class="tabular">${fmtDate(it['报名开始'])} → ${fmtDate(it['报名截止'])}</span></div>
               <div>比赛 <span class="tabular">${fmtDate(it['比赛时间'])}</span> · 面向 ${escapeHtml(it['面向年级'] || '—')}</div>
               <div>推荐理由：${escapeHtml(it['推荐理由'] || '')}</div>
               ${it['建议'] ? `<div class="rec__advice">${escapeHtml(it['建议'])}</div>` : ''}
             </dl>
             <div class="doc-card__foot">
               <span>届次 ${it['届次'] ?? '—'}</span>
             </div>
           </article>`;
         }).join('')}</div>`;
    } catch (err) {
      box.innerHTML = `<div style="padding:var(--space-5)"><div class="alert"><strong>!</strong><span>${escapeHtml(err.message)}</span></div></div>`;
    }
  }

  function initPlanning() {
    $('#planForm').addEventListener('submit', (e) => {
      e.preventDefault();
      loadPlanning();
    });
    VIEW_LOADERS.planning = loadPlanning;
  }

  /* ============================================================
     入场动画
     ============================================================ */

  function initReveal() {
    const nodes = $$('.reveal');
    if (!('IntersectionObserver' in window)) {
      nodes.forEach((n) => n.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in');
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    nodes.forEach((n) => io.observe(n));
  }

  /* ============================================================
     启动
     ============================================================ */

  function boot() {
    initNav();
    initChat();
    initDocs();
    initCompetitions();
    initExams();
    initPlanning();
    initReader();
    initReveal();

    loadHealth().then(loadStats);
    viewLoaded.chat = true;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
