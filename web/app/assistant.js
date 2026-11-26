// "VisionX-dən soruş": chat panel that sends the current result as context to /api/chat (Gemini, server-side).
const API = new URL('../api/chat', import.meta.url).href;

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// light formatting: paragraphs, "- " / "* " / "1." lists, **bold**
function format(text) {
  const lines = esc(text).split(/\n/);
  let html = '', list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const l = raw.trim().replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    const ul = l.match(/^[-*•]\s+(.*)/), ol = l.match(/^\d+[.)]\s+(.*)/);
    if (ul || ol) {
      const t = ul ? 'ul' : 'ol';
      if (list !== t) { close(); html += `<${t}>`; list = t; }
      html += `<li>${(ul || ol)[1]}</li>`;
    } else if (l) { close(); html += `<p>${l}</p>`; } else close();
  }
  close();
  return html;
}

export function mountAssistant(root, context, suggestions) {
  const history = [];
  root.innerHTML = `
    <p class="label">VisionX-dən soruş <span class="tag">AI</span></p>
    <div class="chat__log" data-log aria-live="polite"></div>
    <div class="chat__chips" data-chips>${suggestions.map((s) => `<button type="button" class="chip" data-q="${esc(s)}">${esc(s)}</button>`).join('')}</div>
    <form class="chat__form" data-form>
      <input class="input" name="q" maxlength="600" autocomplete="off" placeholder="Sualınızı yazın…" aria-label="Sual">
      <button class="btn btn--primary btn--sm" type="submit">Göndər</button>
    </form>
    <p class="chat__note">Cavabı Gemini AI yazır və səhv edə bilər. Şəxsi məlumat yazmayın.</p>`;
  const log = root.querySelector('[data-log]');
  const form = root.querySelector('[data-form]');
  const input = form.querySelector('input');
  const btn = form.querySelector('button');

  const add = (role, html) => {
    const d = document.createElement('div');
    d.className = 'chat__msg chat__msg--' + role;
    d.innerHTML = html;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  };

  async function ask(q) {
    q = q.trim();
    if (!q || btn.disabled) return;
    add('user', `<p>${esc(q)}</p>`);
    const wait = add('bot', '<p class="chat__typing">Yazır…</p>');
    btn.disabled = true; input.value = '';
    try {
      const r = await fetch(API, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context: typeof context === 'function' ? context() : context, history, question: q }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok || !data.answer) throw new Error(data.error || (r.status === 404 ? 'AI köməkçi bu serverdə yoxdur (yalnız canlı saytda işləyir).' : 'AI cavab vermədi.'));
      wait.innerHTML = format(data.answer);
      history.push({ role: 'user', text: q }, { role: 'model', text: data.answer });
    } catch (e) {
      wait.classList.add('chat__msg--err');
      wait.innerHTML = `<p>${esc(e.message)}</p>`;
    } finally {
      btn.disabled = false; input.focus();
    }
  }

  form.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
  root.querySelectorAll('[data-q]').forEach((b) => b.addEventListener('click', () => ask(b.dataset.q)));
}
