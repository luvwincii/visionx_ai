import { analyzeField } from './fieldmap.js';

// Contact e-mail: change here once the real inbox exists.
const CONTACT_EMAIL = 'info@visionx.az';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// Nav border on scroll
const nav = $('.nav');
const onScroll = () => nav.classList.toggle('is-scrolled', window.scrollY > 8);
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

// Mobile menu
const toggle = $('.nav__toggle');
const menu = $('#mobile-menu');
toggle?.addEventListener('click', () => {
  const open = toggle.getAttribute('aria-expanded') === 'true';
  toggle.setAttribute('aria-expanded', String(!open));
  menu.hidden = open;
});
$$('#mobile-menu a').forEach((a) => a.addEventListener('click', () => {
  toggle.setAttribute('aria-expanded', 'false');
  menu.hidden = true;
}));

// Load an image element, resolved when decoded
function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.decoding = 'async';
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

function paint(canvas, mapCanvas) {
  canvas.width = mapCanvas.width; canvas.height = mapCanvas.height;
  canvas.getContext('2d').drawImage(mapCanvas, 0, 0);
}

// Compare slider: computes the live health map in the browser
async function initCompare(el) {
  el.classList.add('is-loading');
  const canvas = $('canvas', el);
  try {
    const img = await loadImage(el.dataset.src);
    const res = analyzeField(img, { maxSide: 1440 });
    paint(canvas, res.map);
  } catch (e) { console.error(e); }
  el.classList.remove('is-loading');
  const range = $('.compare__range', el);
  const set = (v) => el.style.setProperty('--pos', v + '%');
  range.addEventListener('input', () => set(range.value));
  set(range.value);
  // Gentle intro sweep once in view
  const io = new IntersectionObserver((entries) => {
    if (!entries[0].isIntersecting) return;
    io.disconnect();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let t0;
    const step = (t) => {
      t0 ??= t;
      const p = Math.min(1, (t - t0) / 1400);
      const v = 50 + Math.sin(p * Math.PI * 2) * 22 * (1 - p);
      range.value = v; set(v);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, { threshold: 0.5 });
  io.observe(el);
}
$$('[data-compare]').forEach(initCompare);

// Static live map (How it works, step 2)
$$('[data-livemap]').forEach((el) => {
  const io = new IntersectionObserver(async (entries) => {
    if (!entries[0].isIntersecting) return;
    io.disconnect();
    try {
      const img = await loadImage(el.dataset.src);
      const res = analyzeField(img, { maxSide: 1280 });
      paint($('canvas', el), res.map);
    } catch (e) { console.error(e); }
  }, { rootMargin: '300px' });
  io.observe(el);
});

// Model metrics -> stats on the page
fetch('models/metrics.json').then((r) => (r.ok ? r.json() : null)).then((m) => {
  if (!m) return;
  const fmt = (x) => Math.round(x * 100) + '%';
  const map = { acc: m.realworld?.top1 != null ? fmt(m.realworld.top1) : null,
    hd: m.realworld?.healthy_vs_diseased != null ? fmt(m.realworld.healthy_vs_diseased) : null,
    classes: m.num_classes_disease, crops: m.num_crops };
  $$('[data-stat]').forEach((el) => { const v = map[el.dataset.stat]; if (v != null) el.textContent = v; });
}).catch(() => {});

// Plan links preselect the role in the form
$$('.plan__link').forEach((a) => a.addEventListener('click', () => {
  const ta = $('#contact-form textarea');
  if (ta && !ta.value) ta.value = `Maraqlandığım xidmət: ${a.dataset.role}.\n`;
}));

// Contact form: validates, then opens the user's mail app with the message prepared
$$('[data-email]').forEach((a) => { a.href = 'mailto:' + CONTACT_EMAIL; a.textContent = CONTACT_EMAIL; });
const form = $('#contact-form');
form?.addEventListener('submit', (e) => {
  e.preventDefault();
  const status = $('.form__status', form);
  let ok = true;
  $$('[required]', form).forEach((f) => {
    const bad = !f.value.trim() || (f.type === 'email' && !/^\S+@\S+\.\S+$/.test(f.value));
    f.classList.toggle('is-invalid', bad);
    if (bad) ok = false;
  });
  if (!ok) { status.textContent = 'Zəhmət olmasa, ulduzlu xanaları düzgün doldurun.'; return; }
  const d = Object.fromEntries(new FormData(form));
  const body = [`Ad: ${d.name}`, `Təşkilat: ${d.org || '-'}`, `Kim: ${d.role || '-'}`, `E-poçt: ${d.email}`, `Sahə: ${d.ha || '-'} ha`, '', d.msg].join('\n');
  window.location.href = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent('VisionX: ' + (d.role || 'müraciət'))}&body=${encodeURIComponent(body)}`;
  status.textContent = 'E-poçt tətbiqiniz açılır. Açılmasa, birbaşa ' + CONTACT_EMAIL + ' ünvanına yazın.';
});

$$('[data-year]').forEach((el) => { el.textContent = new Date().getFullYear(); });
