import { analyzeField, drawZoneGrid } from '../assets/js/fieldmap.js';
import { loadModel, classify } from './classifier.js';
import { describe, TYPES, GENERAL_NOTE, fieldAdvice } from './knowledge.js';
import { saveAnalysis, listAnalyses, deleteAnalysis, getAnalysis, newId } from './store.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pctFmt = (p) => (p >= 0.995 ? '>99' : Math.round(p * 100)) + '%';

const SAMPLES = {
  field: [
    { src: '../assets/img/field-vineyard.webp', label: 'Üzüm bağı', name: 'Nümunə: üzüm bağı' },
    { src: '../assets/img/field-canal.webp', label: 'Suvarma sahələri', name: 'Nümunə: suvarılan sahələr' },
    { src: '../assets/img/samples/field-parcels.webp', label: 'Parsellər', name: 'Nümunə: parsellər' },
  ],
  leaf: [
    { src: '../assets/img/samples/leaf-cotton.webp', label: 'Pambıq', name: 'Nümunə: pambıq' },
    { src: '../assets/img/samples/leaf-grape.webp', label: 'Üzüm', name: 'Nümunə: üzüm' },
    { src: '../assets/img/samples/leaf-potato.webp', label: 'Kartof', name: 'Nümunə: kartof' },
  ],
};
const HINTS = {
  field: 'Sahənin üstündən, yuxarıdan çəkilmiş dron şəkli və ya ortofoto yükləyin. Telefonla hündür yerdən çəkilmiş şəkil də olar.',
  leaf: 'Bir yarpağı 20–30 sm məsafədən, gün işığında, kadrın ortasında çəkin. Xəstə hissə aydın görünsün.',
};
const STATUS_TEXT = ['Sağlam', 'Risk', 'Problem'];
const STATUS_DOT = ['ok', 'warn', 'bad'];

const state = { mode: 'field', current: null, filter: 'all' };

// ---------- helpers ----------
function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('Şəkil açılmadı'));
    im.src = src;
  });
}
function canvasFrom(img, maxSide) {
  const w0 = img.naturalWidth || img.width, h0 = img.naturalHeight || img.height;
  const k = Math.min(1, maxSide / Math.max(w0, h0));
  const c = document.createElement('canvas');
  c.width = Math.round(w0 * k); c.height = Math.round(h0 * k);
  const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, c.width, c.height);
  return c;
}
function toast(msg) {
  const t = document.createElement('div');
  t.className = 'toast'; t.textContent = msg; document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}
const MONTHS = ['yanvar', 'fevral', 'mart', 'aprel', 'may', 'iyun', 'iyul', 'avqust', 'sentyabr', 'oktyabr', 'noyabr', 'dekabr'];
const fmtDate = (d) => { const t = new Date(d); const p = (n) => String(n).padStart(2, '0'); return `${t.getDate()} ${MONTHS[t.getMonth()]} ${t.getFullYear()}, ${p(t.getHours())}:${p(t.getMinutes())}`; };

// ---------- model status ----------
const statusEl = $('[data-model-status]');
function setModelStatus(kind, text) {
  statusEl.className = 'appbar__status mono is-' + kind;
  $('span', statusEl).textContent = text;
}
setModelStatus('idle', 'Model: gözləmədə');
function warmModel() {
  setModelStatus('loading', 'Model yüklənir');
  return loadModel((p) => setModelStatus('loading', `Model ${Math.round(p * 100)}%`))
    .then((s) => { setModelStatus('ready', 'Model hazır'); return s; })
    .catch((e) => { console.error(e); setModelStatus('error', 'Model xətası'); throw e; });
}

// ---------- tabs ----------
function showView(v) {
  $$('.tabs__btn').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === v)));
  $$('[data-view-panel]').forEach((p) => { p.hidden = p.dataset.viewPanel !== v; });
  if (v === 'fields') renderHistory();
  window.scrollTo({ top: 0 });
}
$$('.tabs__btn').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
$$('[data-goto]').forEach((b) => b.addEventListener('click', () => showView(b.dataset.goto)));

// ---------- mode ----------
function setMode(m) {
  state.mode = m;
  $(`input[name=mode][value=${m}]`).checked = true;
  $('[data-mode-hint]').textContent = HINTS[m];
  const list = $('[data-samples]');
  list.innerHTML = SAMPLES[m].map((s, i) => `<button class="sample" type="button" data-sample="${i}" title="${esc(s.name)}"><img src="${esc(s.src)}" alt="" loading="lazy"><span>${esc(s.label)}</span></button>`).join('');
  $$('[data-sample]', list).forEach((b) => b.addEventListener('click', async () => {
    const s = SAMPLES[m][+b.dataset.sample];
    const nameInput = $('[data-field-name]');
    if (!nameInput.value) nameInput.value = s.name;
    run(await loadImage(s.src), s.src);
  }));
  if (m === 'leaf') warmModel().catch(() => {});
}
$$('input[name=mode]').forEach((r) => r.addEventListener('change', () => setMode(r.value)));

// ---------- input ----------
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { toast('Zəhmət olmasa, şəkil faylı seçin'); return; }
  const url = URL.createObjectURL(file);
  loadImage(url).then((img) => run(img, file.name)).catch(() => toast('Bu şəkil açılmadı. JPG və ya PNG yoxlayın.'));
}
$('[data-file]').addEventListener('change', (e) => { handleFile(e.target.files[0]); e.target.value = ''; });
$('[data-camera]').addEventListener('change', (e) => { handleFile(e.target.files[0]); e.target.value = ''; });
const drop = $('[data-drop]');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
window.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'));
  if (f) handleFile(f);
});

// ---------- run ----------
function showLoading(text) {
  $('[data-empty]').hidden = true;
  scrollToResult();
  const r = $('[data-result]');
  r.hidden = false;
  r.innerHTML = '';
  r.appendChild($('#tpl-loading').content.cloneNode(true));
  $('[data-loading-text]', r).textContent = text;
}

async function run(img, source) {
  const name = $('[data-field-name]').value.trim();
  try {
    if (state.mode === 'field') {
      showLoading('Sağlamlıq xəritəsi hesablanır…');
      await new Promise((r) => setTimeout(r, 30));
      const res = analyzeField(img, { maxSide: 1400, blocksAcross: 140 });
      const photo = canvasFrom(img, 1400);
      const rec = {
        id: newId(), date: Date.now(), mode: 'field', name: name || 'Adsız sahə', source,
        photo: photo.toDataURL('image/jpeg', 0.82), map: res.map.toDataURL('image/png'),
        stats: res.stats, hot: res.hot, zones: res.zones, grid: res.grid,
      };
      await renderField(rec);
      persist(rec);
    } else {
      showLoading('AI modeli yüklənir və yarpaq təhlil edilir…');
      await warmModel();
      const out = await classify(img);
      const photo = canvasFrom(img, 1000);
      const rec = {
        id: newId(), date: Date.now(), mode: 'leaf', name: name || 'Adsız sahə', source,
        photo: photo.toDataURL('image/jpeg', 0.85),
        top: out.top, ranked: out.ranked, healthyP: out.healthyP, otherP: out.otherP, crop: out.crop,
      };
      await renderLeaf(rec);
      persist(rec);
    }
  } catch (e) {
    console.error(e);
    $('[data-result]').innerHTML = `<div class="empty"><h2>Analiz alınmadı</h2><p class="muted">${esc(e.message || e)}</p></div>`;
  }
}

async function persist(rec) {
  try {
    await saveAnalysis(rec);
    state.current = rec.id;
    refreshCount();
    refreshNames();
  } catch (e) { console.warn('save failed', e); }
}

// ---------- field result ----------
async function renderField(rec) {
  const [photo, map] = await Promise.all([loadImage(rec.photo), loadImage(rec.map)]);
  const s = rec.stats;
  const advice = fieldAdvice(s, rec.hot);
  const ranked = [...rec.zones].filter((z) => z.veg >= 4).sort((a, b) => b.score - a.score).slice(0, 6);
  const r = $('[data-result]');
  r.hidden = false; $('[data-empty]').hidden = true;
  r.innerHTML = `
    <div class="result__head">
      <div class="result__title"><h2>${esc(rec.name)}</h2><p class="mono">Dron / sahə · ${fmtDate(rec.date)}</p></div>
      <div class="result__tools">
        <button class="btn btn--ghost btn--sm" data-download>Xəritəni yüklə</button>
        <button class="btn btn--ghost btn--sm" data-new>Yeni şəkil</button>
      </div>
    </div>
    <div class="result__body">
      <div class="viewer">
        <canvas data-view-canvas></canvas>
        <div class="viewer__switch" role="group" aria-label="Görünüş">
          <button aria-pressed="false" data-layer="photo">Foto</button>
          <button aria-pressed="true" data-layer="overlay">Üst-üstə</button>
          <button aria-pressed="false" data-layer="map">Xəritə</button>
        </div>
        <div class="legend"><span><i class="dot dot--ok"></i>Sağlam</span><span><i class="dot dot--warn"></i>Risk</span><span><i class="dot dot--bad"></i>Problem</span><span><i class="dot dot--soil"></i>Bitkisiz</span><span><i class="dot dot--void"></i>Analiz edilmir</span></div>
      </div>
      <div class="report">
        <div class="report__sec">
          <p class="label">Bitki örtüyünün vəziyyəti</p>
          <div class="bar" role="img" aria-label="Sağlam ${s.ok}%, risk ${s.risk}%, problem ${s.bad}%"><i class="b-ok" style="width:${s.ok}%"></i><i class="b-warn" style="width:${s.risk}%"></i><i class="b-bad" style="width:${s.bad}%"></i></div>
          <div class="kpis">
            <div class="kpi"><b>${s.ok}%</b><span><i class="dot dot--ok"></i>Sağlam</span></div>
            <div class="kpi"><b>${s.risk}%</b><span><i class="dot dot--warn"></i>Risk</span></div>
            <div class="kpi"><b>${s.bad}%</b><span><i class="dot dot--bad"></i>Problem</span></div>
          </div>
          <div class="meta-row"><span>Bitki örtüyü: ${s.vegCover}%</span><span>Bitkisiz: ${s.soil}%</span><span>Analiz edilmir: ${s.void}%</span><span>VARI: ${s.index.toFixed(2)}</span></div>
        </div>
        <div class="report__sec">
          <p class="label">Nə etməli</p>
          <ul class="advice">${advice.map((a) => `<li><i class="dot dot--${STATUS_DOT[a.s]}"></i><div><b>${esc(a.t)}</b><p>${esc(a.d)}</p></div></li>`).join('')}</ul>
        </div>
        <div class="report__sec">
          <p class="label">Zonalar (${rec.grid}×${rec.grid} şəbəkə)</p>
          ${ranked.length ? `<table class="zones"><thead><tr><th>Zona</th><th>Sağlam</th><th>Risk</th><th>Problem</th></tr></thead><tbody>
            ${ranked.map((z) => `<tr><td>${z.id}</td><td>${z.ok}%</td><td>${z.risk}%</td><td>${z.bad}%</td></tr>`).join('')}</tbody></table>`
            : '<p class="muted">Zonaları qiymətləndirmək üçün kifayət qədər bitki örtüyü yoxdur.</p>'}
        </div>
        <div class="report__sec">
          <p class="note">Xəritə adi RGB şəkildən hesablanır: bitki örtüyü ExG, sağlamlıq isə VARI indeksi ilə ölçülür. Rənglər eyni şəkil daxilində müqayisə üçündür. Kölgə, səhər-axşam işığı və payız rəngləri nəticəyə təsir edə bilər. Qırmızı zonanı yerində yoxlayın və yarpaq şəklini telefon rejimində təhlil edin.</p>
        </div>
      </div>
    </div>`;

  const canvas = $('[data-view-canvas]', r);
  canvas.width = map.width; canvas.height = map.height;
  const ctx = canvas.getContext('2d');
  const draw = (layer) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    if (layer !== 'map') ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);
    if (layer !== 'photo') {
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = layer === 'overlay' ? 0.62 : 1;
      ctx.drawImage(map, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
      drawZoneGrid(ctx, canvas.width, canvas.height, rec.grid, rec.hot, canvas.width / Math.max(1, canvas.clientWidth));
    }
    $$('[data-layer]', r).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.layer === layer)));
  };
  draw('overlay');
  let lastW = canvas.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(canvas.clientWidth - lastW) < 40) return;
    lastW = canvas.clientWidth;
    draw($('[data-layer][aria-pressed="true"]', r)?.dataset.layer || 'overlay');
  }).observe(canvas);
  scrollToResult();
  $$('[data-layer]', r).forEach((b) => b.addEventListener('click', () => draw(b.dataset.layer)));
  $('[data-download]', r).addEventListener('click', () => {
    const a = document.createElement('a');
    a.download = `visionx-${(rec.name || 'sahe').replace(/\s+/g, '-')}.png`;
    a.href = canvas.toDataURL('image/png'); a.click();
  });
  $('[data-new]', r).addEventListener('click', resetStage);
}

// ---------- leaf result ----------
async function renderLeaf(rec) {
  const top = rec.top;
  const isOther = top.cls === 'other' || rec.otherP > 0.5;
  const unsure = !isOther && top.p < 0.45;
  const d = describe(isOther ? 'other' : top.cls);
  const st = d.status;
  const statusCls = st == null ? 'sx' : 's' + st;
  const statusLabel = st == null ? 'Nəticə yoxdur' : (unsure ? 'Ehtimal aşağıdır · ' : '') + STATUS_TEXT[st];
  const alts = rec.ranked.filter((x) => x.cls !== top.cls && x.p >= 0.03).slice(0, 3);
  const r = $('[data-result]');
  r.hidden = false; $('[data-empty]').hidden = true;
  r.innerHTML = `
    <div class="result__head">
      <div class="result__title"><h2>${esc(rec.name)}</h2><p class="mono">Telefon / yarpaq · ${fmtDate(rec.date)}</p></div>
      <div class="result__tools"><button class="btn btn--ghost btn--sm" data-new>Yeni şəkil</button></div>
    </div>
    <div class="result__body">
      <div class="viewer viewer--leaf"><img src="${rec.photo}" alt="Yüklənmiş yarpaq şəkli"></div>
      <div class="report">
        <div class="report__sec diag">
          <span class="diag__status ${statusCls}"><i class="dot dot--${st == null ? 'void' : STATUS_DOT[st]}"></i>${esc(statusLabel)}</span>
          <h3>${esc(d.name)}</h3>
          ${d.latin ? `<p class="diag__latin">${esc(d.latin)}</p>` : ''}
          ${isOther ? '' : `<div class="diag__crop"><span class="tag">${esc(d.crop || '')}</span>${d.type ? `<span class="tag">${esc(TYPES[d.type])}</span>` : ''}</div>`}
          ${isOther ? '' : `<div class="conf">
            <div class="conf__row"><span>Modelin əminliyi</span><b>${pctFmt(top.p)}</b></div>
            <div class="conf__track"><i style="width:${Math.round(top.p * 100)}%"></i></div>
            <div class="conf__row"><span>Sağlam olma ehtimalı</span><b>${pctFmt(rec.healthyP)}</b></div>
          </div>`}
          <p class="muted">${esc(d.about)}</p>
          ${unsure ? '<p class="note note--warn">Model bu diaqnozda tam əmin deyil. Yarpağı daha yaxından, gün işığında yenidən çəkin və aşağıdakı alternativlərə də baxın.</p>' : ''}
        </div>
        <div class="report__sec">
          <p class="label">Nə etməli</p>
          <ol class="steps-list">${d.actions.map((a) => `<li>${esc(a)}</li>`).join('')}</ol>
        </div>
        ${alts.length && !isOther ? `<div class="report__sec">
          <p class="label">Digər ehtimallar</p>
          <ul class="alts">${alts.map((a) => { const x = describe(a.cls); return `<li><span>${esc(x.crop ? x.crop + ' · ' : '')}${esc(x.name)}</span><b>${pctFmt(a.p)}</b></li>`; }).join('')}</ul>
        </div>` : ''}
        <div class="report__sec"><p class="note">${esc(GENERAL_NOTE)}</p></div>
      </div>
    </div>`;
  $('[data-new]', r).addEventListener('click', resetStage);
  scrollToResult();
}

function scrollToResult() {
  if (window.matchMedia('(max-width: 860px)').matches) $('.stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetStage() {
  $('[data-result]').hidden = true;
  $('[data-result]').innerHTML = '';
  $('[data-empty]').hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------- history ----------
async function refreshCount() {
  const all = await listAnalyses().catch(() => []);
  const el = $('[data-count]');
  el.hidden = !all.length; el.textContent = all.length;
}
async function refreshNames() {
  const all = await listAnalyses().catch(() => []);
  const names = [...new Set(all.map((a) => a.name).filter(Boolean))];
  $('#field-names').innerHTML = names.map((n) => `<option value="${esc(n)}">`).join('');
}

async function renderHistory() {
  const all = await listAnalyses().catch(() => []);
  const names = [...new Set(all.map((a) => a.name))];
  const filters = [['all', 'Hamısı'], ['field', 'Dron'], ['leaf', 'Yarpaq'], ...names.slice(0, 8).map((n) => ['name:' + n, n])];
  $('[data-filters]').innerHTML = filters.map(([k, l]) => `<button class="chip" aria-pressed="${state.filter === k}" data-filter="${esc(k)}">${esc(l)}</button>`).join('');
  $$('[data-filter]').forEach((b) => b.addEventListener('click', () => { state.filter = b.dataset.filter; renderHistory(); }));
  const f = state.filter;
  const items = all.filter((a) => f === 'all' || a.mode === f || (f.startsWith('name:') && a.name === f.slice(5)));
  $('[data-history-empty]').hidden = !!items.length;
  $('[data-history]').innerHTML = items.map((a) => {
    let sum;
    if (a.mode === 'field') {
      const s = a.stats;
      sum = `<div class="bar"><i class="b-ok" style="width:${s.ok}%"></i><i class="b-warn" style="width:${s.risk}%"></i><i class="b-bad" style="width:${s.bad}%"></i></div>
             <div class="entry__sum"><span>${s.ok}% sağlam</span><span>·</span><span>${s.bad}% problem</span></div>`;
    } else {
      const d = describe(a.top.cls === 'other' || a.otherP > 0.5 ? 'other' : a.top.cls);
      sum = `<div class="entry__sum"><i class="dot dot--${d.status == null ? 'void' : STATUS_DOT[d.status]}"></i><span>${esc(d.crop ? d.crop + ' · ' : '')}${esc(d.name)}</span></div>`;
    }
    return `<article class="entry" data-open="${a.id}" tabindex="0" role="button" aria-label="${esc(a.name)} analizini aç">
      <div class="entry__img"><img src="${a.mode === 'field' ? a.map : a.photo}" alt="" loading="lazy"><span class="tag">${a.mode === 'field' ? 'Dron' : 'Yarpaq'}</span></div>
      <div class="entry__body"><b>${esc(a.name)}</b><small>${fmtDate(a.date)}</small>${sum}
        <button class="btn btn--ghost btn--sm" data-del="${a.id}" style="justify-self:start;margin-top:6px">Sil</button></div>
    </article>`;
  }).join('');
  $$('[data-open]').forEach((el) => {
    const open = async (e) => {
      if (e.target.closest('[data-del]')) return;
      const a = await getAnalysis(el.dataset.open);
      if (!a) return;
      showView('new');
      setMode(a.mode);
      $('[data-field-name]').value = a.name;
      a.mode === 'field' ? renderField(a) : renderLeaf(a);
    };
    el.addEventListener('click', open);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(e); });
  });
  $$('[data-del]').forEach((b) => b.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!confirm('Bu analiz silinsin?')) return;
    await deleteAnalysis(b.dataset.del);
    refreshCount(); renderHistory();
  }));
}

// ---------- boot ----------
setMode(new URLSearchParams(location.search).get('mode') === 'leaf' ? 'leaf' : 'field');
refreshCount();
refreshNames();
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
