// Local history in IndexedDB (stays on this device).
const DB = 'visionx', STORE = 'analyses', VERSION = 1;

function open() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VERSION);
    r.onupgradeneeded = () => {
      const s = r.result.createObjectStore(STORE, { keyPath: 'id' });
      s.createIndex('date', 'date');
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => res(out?.result ?? out);
    t.onerror = () => rej(t.error);
  });
}

export const saveAnalysis = (a) => tx('readwrite', (s) => s.put(a));
export const deleteAnalysis = (id) => tx('readwrite', (s) => s.delete(id));
export const getAnalysis = (id) => tx('readonly', (s) => s.get(id));
export async function listAnalyses() {
  const all = await tx('readonly', (s) => s.getAll());
  return (all || []).sort((a, b) => b.date - a.date);
}

export function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}
