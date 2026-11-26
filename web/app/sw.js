// VisionX offline cache: app shell + model are cached after the first visit.
const CACHE = 'visionx-v3';
const SHELL = [
  './', './index.html', './app.js', './classifier.js', './knowledge.js', './store.js', './manifest.webmanifest',
  '../assets/css/base.css', '../assets/css/app.css', '../assets/js/fieldmap.js', '../assets/img/favicon.svg',
  '../assets/fonts/geist-sans-latin-400-normal.woff2', '../assets/fonts/geist-sans-latin-500-normal.woff2',
  '../assets/fonts/geist-sans-latin-600-normal.woff2', '../assets/fonts/geist-mono-latin-400-normal.woff2',
  '../assets/fonts/geist-mono-latin-500-normal.woff2',
];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  const heavy = /\.(onnx|wasm)$/.test(new URL(req.url).pathname);
  // heavy assets: cache-first; everything else: network-first with cache fallback
  e.respondWith(heavy
    ? caches.match(req).then((hit) => hit || fetch(req).then((res) => { if (res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(req, cp)); } return res; }))
    : fetch(req).then((res) => { if (res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(req, cp)); } return res; }).catch(() => caches.match(req)));
});
