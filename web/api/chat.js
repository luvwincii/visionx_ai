// VisionX assistant: explains a VisionX result and answers follow-up questions in Azerbaijani.
// Runs as a Vercel serverless function. The Gemini key lives only in the server env (GEMINI_API_KEY).
// Free-tier models are tried in order; on a rate limit (429) or overload the next one is used.

const MODELS = (process.env.GEMINI_MODELS || 'gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite')
  .split(',').map((s) => s.trim()).filter(Boolean);
const ENDPOINT = (m) => `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(m)}:generateContent`;

const SYSTEM = `Sən VisionX-in köməkçisisən. VisionX fermerlər üçün dron və telefon şəkillərindən sahə və yarpaq analizi edən tətbiqdir.
Qaydalar:
- Həmişə sadə, düzgün azərbaycan dilində, qısa cavab ver (3–8 cümlə). Lazım olanda qısa siyahı işlət.
- Cavabını aşağıda verilən VisionX nəticəsinə əsaslandır. Nəticədə olmayan faktı uydurma.
- Model əmin deyilsə (aşağı ehtimal), bunu açıq de və şəkli yenidən çəkməyi və ya aqronoma göstərməyi məsləhət gör.
- Kimyəvi preparat barədə yalnız ümumi qrup adını (məs. mis tərkibli fungisid) de; konkret doza vermə. Həmişə qeyd et ki, yalnız Azərbaycanda qeydiyyatdan keçmiş preparat, etiketdəki dozaya uyğun və yerli aqronomla məsləhətləşərək istifadə olunmalıdır.
- Sual kənd təsərrüfatı və ya VisionX nəticəsi ilə bağlı deyilsə, nəzakətlə bildir ki, yalnız bu mövzularda kömək edə bilirsən.
- Şəxsi məlumat istəmə.`;

// best-effort per-instance rate limit
const hits = new Map();
function limited(ip) {
  const now = Date.now(), win = 10 * 60 * 1000, max = 20;
  const arr = (hits.get(ip) || []).filter((t) => now - t < win);
  arr.push(now); hits.set(ip, arr);
  if (hits.size > 5000) hits.clear();
  return arr.length > max;
}

const clip = (s, n) => String(s ?? '').slice(0, n);

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') { res.status(405).json({ error: 'Yalnız POST' }); return; }
  const key = process.env.GEMINI_API_KEY;
  if (!key) { res.status(503).json({ error: 'AI köməkçi hələ qoşulmayıb (server açarı yoxdur).' }); return; }
  const ip = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'local';
  if (limited(ip)) { res.status(429).json({ error: 'Çox sual göndərildi. Bir neçə dəqiqədən sonra yenidən yoxlayın.' }); return; }

  let body = req.body;
  if (typeof body === 'string') { try { body = JSON.parse(body); } catch { body = {}; } }
  const context = clip(body?.context, 4000);
  const history = Array.isArray(body?.history) ? body.history.slice(-8) : [];
  const question = clip(body?.question, 600).trim();
  if (!question) { res.status(400).json({ error: 'Sual boşdur.' }); return; }

  const contents = [];
  for (const h of history) {
    if (!h || (h.role !== 'user' && h.role !== 'model')) continue;
    contents.push({ role: h.role, parts: [{ text: clip(h.text, 1500) }] });
  }
  contents.push({ role: 'user', parts: [{ text: question }] });
  const payload = {
    system_instruction: { parts: [{ text: `${SYSTEM}\n\nVisionX nəticəsi:\n${context || '(nəticə verilməyib)'}` }] },
    contents,
    generationConfig: { maxOutputTokens: 2048, temperature: 0.4 },
  };

  let lastErr = 'AI cavab vermədi.';
  for (const model of MODELS) {
    try {
      const r = await fetch(ENDPOINT(model), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
        body: JSON.stringify(payload),
      });
      if (r.status === 429 || r.status === 503 || r.status === 500 || r.status === 404) {
        lastErr = r.status === 429 ? 'Pulsuz AI limiti dolub. Bir az sonra yenidən yoxlayın.' : 'AI xidməti hazırda məşğuldur.';
        continue; // try the next free model
      }
      const data = await r.json().catch(() => ({}));
      if (!r.ok) { lastErr = data?.error?.message ? 'AI xətası.' : 'AI xətası.'; continue; }
      const cand = data?.candidates?.[0];
      const text = (cand?.content?.parts || []).filter((p) => !p.thought).map((p) => p.text || '').join('').trim();
      if (!text) { lastErr = 'AI bu suala cavab vermədi.'; continue; }
      res.status(200).json({ answer: text, model });
      return;
    } catch (e) {
      lastErr = 'AI xidmətinə qoşulmaq alınmadı.';
    }
  }
  res.status(502).json({ error: lastErr });
};
