# VisionX

AI ilə dron və telefon əsaslı sahə monitorinqi. Sayt və işlək MVP platforma.

- **Sayt**: `web/index.html`. Məhsulun təqdimatı, iş prinsipi, biznes modeli və əlaqə forması.
- **Platforma**: `web/app/`. Şəkil yüklənir, analiz brauzerin içində, cihazda aparılır və nəticələr "Sahələrim" bölməsində saxlanır.
- **ML**: `ml/`. Açıq modellərin qiymətləndirilməsi, VisionX modelinin öyrədilməsi və ONNX eksportu.

## İki analiz rejimi

| Rejim | Giriş | Çıxış | Necə işləyir |
|---|---|---|---|
| **Dron / sahə** | Yuxarıdan çəkilmiş RGB şəkil və ya ortofoto | Yaşıl/sarı/qırmızı xəritə, zonalar (A1–F6), faizlər, tövsiyələr | ExG ilə bitki örtüyü, VARI ilə sağlamlıq; bloklar üzrə aqreqasiya (`web/assets/js/fieldmap.js`) |
| **Telefon / yarpaq** | Yaxından çəkilmiş yarpaq | Bitki, xəstəlik, əminlik, alternativlər, azərbaycanca izah və addımlar | DINOv2-small + VisionX təsnifat başlığı, ONNX Runtime Web (`web/app/classifier.js`) |

Şəkillər serverə göndərilmir. Model ilk istifadədə yüklənir, sonra service worker ilə keşlənir və oflayn işləyir.

## Lokal işə salmaq

```bash
cd web
python3 -m http.server 8080
```

Sonra brauzerdə `http://localhost:8080` (sayt) və `http://localhost:8080/app/` (platforma) ünvanlarını açın. Build addımı yoxdur: hər şey statik fayllardır, istənilən statik hostinqə (GitHub Pages, Netlify, Vercel, Cloudflare Pages) yerləşdirilə bilər.

## Model: seçim və nəticələr

**1. Açıq modellərin yoxlanması.** Hugging Face-də ən çox istifadə olunan açıq çəkili (open-weight) bitki xəstəliyi modelləri test edildi. Hamısı PlantVillage laboratoriya şəkillərində ~95–99% göstərir, lakin fermerin telefonla sahədə çəkdiyi real şəkillərə (PlantDoc) keçəndə kəskin düşür:

| Model (PlantDoc test, n=637, eyni şəkillər) | Top-1 |
|---|---|
| linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification | 27.0% |
| plantdoctor/swin-tiny-patch4-window7-224-plant-doctor | 25.6% |
| rodynaemad/plant-disease-dinov2-small | 38.5% |
| neghani/plantvillage-resnet18 (PlantDoc-də öyrədilib, test sızması ehtimalı var) | 56.4% |
| **VisionX v1 (int8 ONNX, brauzerdə)** | **74.4%** |

**2. VisionX modeli.** Dondurulmuş DINOv2-small (Apache-2.0) xüsusiyyətləri və üzərində MLP təsnifat başlığı. Öyrətmə datası: PlantVillage, PlantDoc (train hissəsi), pambıq (Banqladeş, real sahə), üzüm və "yarpaq deyil" sinfi (Imagenette). Real sahə şəkillərinə daha çox çəki verilib, augmentasiya tətbiq edilib.

| Test dəsti (öyrətmədə istifadə olunmayıb) | Top-1 | Top-3 | Sağlam/xəstə |
|---|---|---|---|
| PlantDoc, real sahə şəkilləri | 74.6% | 91.4% | 96.4% |
| Pambıq | 98.6% | 100% | 99.3% |
| Üzüm | 98.7% | 99.8% | 99.8% |
| PlantVillage | 98.4% | 99.9% | 99.9% |
| Yarpaq deyil | 100% | – | – |

Ümumilikdə 49 sinif var: 15 bitki üzrə 48 diaqnoz və "yarpaq deyil". int8 kvantlaşdırmadan sonra model 25 MB-dır və dəqiqlik itkisi yoxdur (xam şəkildən başlayan tam pipeline ilə yoxlanılıb). Ətraflı nəticələr: `ml/results.json`, `web/models/metrics.json`.

Qeydlər:
- PlantVillage-də eyni yarpağın bir neçə şəkli ola bilər, ona görə 98% nəticə optimistdir. Real göstərici PlantDoc nəticəsidir.
- Buğda xəstəlikləri üçün açıq dataset tapıldı (Project-AgML/wheat_disease_detection), lakin sinif adları nəşr olunmayıb. Etiketləri təxmin etməmək üçün daxil edilmədi. Növbəti addım: Azərbaycanda yerli buğda, pambıq və üzüm şəkilləri toplayıb modeli yenidən öyrətmək.
- Dron rejimi öyrədilmiş model deyil, RGB vegetasiya indekslərinə (ExG, VARI) əsaslanır. Multispektral NDVI qədər dəqiq deyil, amma adi kamera ilə işləyir. Növbəti mərhələdə dron şəkillərindəki bitkiləri seqmentasiya modeli ilə ayırmaq olar.

## Dron AI səbəbləri (seqmentasiya, beta)

Sahə nəticəsində **"AI ilə səbəbləri tap"** düyməsi var. Model SegFormer-dir: ImageNet-də öyrədilmiş MiT-B3 kodlayıcı və yüngül dekoder. Bütöv şəkildə, Kaggle-da 2×T4 GPU üzərində 5 saat (3 epoxa, ~57 min plitə) öyrədilib. 512×512 plitə üçün hər sinfin 128×128 ehtimal xəritəsini verir. Model int8 ONNX formatındadır (49 MB), brauzerdə işləyir, bir şəkli ~4–10 saniyəyə analiz edir. int8 və tam (fp32) versiyanın dəqiqliyi eynidir.

- **Data:** Agriculture-Vision 2021 (Chiu et al., CVPR 2020), yalnız RGB. Lisenziya yalnız qeyri-kommersiya tədqiqata icazə verir. Burada yalnız tələbə müsabiqəsi və sərgi demosu üçün istifadə olunub. Xam data repoda yoxdur. SegFormer çəkilərinin lisenziyası da qeyri-kommersiyadır.
- **Eşiklər:** iki ayrı dəstdə (val-00001 və val-00000-ın birinci yarısı) seçilib, ortalaması götürülüb. Test plitələri eşik seçimində istifadə olunmayıb.
- **Test, modelin görmədiyi plitələrdə (piksel IoU):**

| Sinif | Köhnə model (DINOv2 başlığı) | Yeni model, test A (240 plitə) | Yeni model, test B (800 plitə) |
|---|---|---|---|
| Quruma | 0.44 | 0.82 | 0.61 |
| Su yığılması | 0.55–0.60 | 0.80 | 0.88 |
| Alaq | 0.19–0.26 | 0.57 | 0.32 |
| Qida çatışmazlığı | 0.26–0.31 | 0.35 | 0.40 |
| Əkin boşluğu | 0.15–0.21 | 0.21 | 0.41 |
| **Orta (5 sinif)** | **0.34** | **0.55** | **0.53** |
| Fırtına zədəsi | – | söndürülüb | söndürülüb (nümunə azdır) |

Müqayisə üçün: Agriculture-Vision yarışlarında ən yaxşı nəticələr RGB+infraqırmızı ilə və bütün siniflər üzrə mIoU ~0.55–0.58-dir. Datanın şəkilləri ABŞ-da təyyarədən çəkilib (10–20 sm/piksel). Azərbaycan dron şəkillərində dəqiqliyi yoxlamaq üçün yerli etiketli şəkillər lazımdır.

Skriptlər: `ml/kaggle/train_seg.py` (öyrətmə), `ml/kaggle/export_local.py` (ONNX), `ml/seg_eval_big.py`, `ml/seg_thr_sweep.py` (test və eşiklər). Nəticələr: `ml/seg_compare_*.json`, `ml/seg_thr_sweep.json`. Köhnə model skriptləri: `ml/agv_*.py`.

## VisionX köməkçisi (Gemini)

Nəticə ekranındakı "VisionX-dən soruş" bölməsi `web/api/chat.js` Vercel funksiyası vasitəsilə Gemini-nin pulsuz modellərinə müraciət edir: əvvəl sürətli `gemini-3.5-flash-lite` (~1.5 san), limit dolanda və ya cavab gecikəndə `gemini-3.1-flash-lite`, sonra `gemini-3.8-flash`. Hər cəhdə 15 saniyə limit qoyulub (pulsuz planda `gemini-3.8-flash` testlərdə 20 saniyə ərzində cavab vermədi). Açar yalnız Vercel-də `GEMINI_API_KEY` mühit dəyişənində saxlanılır. Pulsuz planda göndərilən mətn Google tərəfindən məhsullarını yaxşılaşdırmaq üçün istifadə olunur.

## Modeli yenidən öyrətmək

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r ml/requirements.txt
cd ml
# datasetləri və modelləri data/ və models/ qovluqlarına yükləyin (feats.py-də siyahı var), sonra:
python feats.py dinov2-small          # xüsusiyyətlərin çıxarılması
python train_head.py dinov2-small     # təsnifat başlığının öyrədilməsi
python export.py dinov2-small         # ONNX + int8 kvantlaşdırma
python eval_onnx.py visionx-leaf-int8.onnx   # real şəkillər üzərində yekun yoxlama
```

## Dəyişdirilməli olanlar

- `web/assets/js/site.js` faylındakı `CONTACT_EMAIL` hazırda nümunə ünvandır (`info@visionx.az`). Real ünvanı yazın.
- Əlaqə forması hazırda e-poçt tətbiqini açır. Serverə göndərmək üçün Formspree, öz API-niz və ya buna bənzər xidmət qoşulmalıdır.

## Mənbələr

Datasetlər, şəkillər və lisenziyalar: `web/credits.html`.
