// VisionX knowledge base (Azerbaijani). Keys match the model's class names "crop:condition".
// status: 0 healthy (green) · 1 risk (yellow) · 2 problem (red)
// Advice is general agronomic practice; product choice and dose must follow the label of a
// preparation registered in Azerbaijan and, ideally, a local agronomist's confirmation.

export const CROPS = {
  apple: 'Alma', blueberry: 'Qaragilə', cherry: 'Albalı / gilas', corn: 'Qarğıdalı', grape: 'Üzüm',
  orange: 'Sitrus', peach: 'Şaftalı', pepper: 'Bibər', potato: 'Kartof', raspberry: 'Moruq',
  soybean: 'Soya', squash: 'Balqabaq', strawberry: 'Çiyələk', tomato: 'Pomidor', cotton: 'Pambıq',
};

export const TYPES = {
  fungal: 'Göbələk xəstəliyi', oomycete: 'Göbələkəbənzər (oomiset) xəstəlik', bacterial: 'Bakterial xəstəlik', viral: 'Virus xəstəliyi',
  pest: 'Zərərverici', abiotic: 'Fizioloji / qeyri-yoluxucu', healthy: 'Sağlam',
};

const HEALTHY = (crop) => ({
  name: 'Sağlam', type: 'healthy', status: 0,
  about: `Yarpaqda ${crop.toLowerCase()} üçün xarakterik xəstəlik əlaməti görünmür.`,
  actions: [
    'Sahəni həftədə bir dəfə müşahidə etməyə davam edin.',
    'Suvarmanı və gübrələməni mövsümə uyğun saxlayın.',
    'Şübhəli ləkə görsəniz, həmin yarpağı yenidən çəkin.',
  ],
});

const SANITATION = 'Yoluxmuş yarpaq və bitki qalıqlarını toplayıb sahədən kənarda məhv edin.';
const NO_WET = 'Yarpaqları islatmayın: mümkünsə damcı suvarmaya keçin, suvarmanı səhər edin.';
const ROTATION = 'Növbəli əkin edin: eyni bitkini 2–3 il eyni yerə əkməyin.';
const AIRFLOW = 'Budama və düzgün sıxlıqla bitkilər arasında hava dövranını artırın.';
const COPPER = 'Mis tərkibli qeydiyyatlı preparatla profilaktik çiləmə edin (etiketdəki dozaya əməl edin).';
const FUNGICIDE = 'Yayılma varsa, bu xəstəliyə qarşı qeydiyyatlı fungisidlə çiləyin; təsir mexanizmini növbələyin.';
const VECTOR = 'Daşıyıcı həşəratlara (ağqanad, mənənə) qarşı sarı yapışqan tələlər qoyun və sayını izləyin.';
const REMOVE = 'Güclü yoluxmuş bitkiləri sahədən çıxarın ki, xəstəlik qonşu bitkilərə keçməsin.';

export const KB = {
  // Apple
  'apple:scab': { name: 'Alma qoturu', latin: 'Venturia inaequalis', type: 'fungal', status: 2,
    about: 'Yarpaqda zeytuni-qəhvəyi, məxməri ləkələr; sonra meyvədə çatlayan qotur ləkələri. Yaz yağışlarında sürətlə yayılır.',
    actions: ['Payızda tökülmüş yarpaqları yığın və ya basdırın: xəstəlik onlarda qışlayır.', 'Tumurcuq açılandan çiçəkləmədən sonraya qədər yağışlardan əvvəl qoruyucu çiləmə edin.', AIRFLOW, 'Yeni bağda davamlı sortlara üstünlük verin.'] },
  'apple:black_rot': { name: 'Qara çürümə', latin: 'Botryosphaeria obtusa', type: 'fungal', status: 2,
    about: 'Yarpaqda bənövşəyi haşiyəli "qurbağa gözü" ləkələri, meyvədə qara çürümə, budaqlarda xərçəng yaraları.',
    actions: ['Quru budaqları, xərçəngli hissələri və mumiyalaşmış meyvələri kəsib məhv edin.', 'Budama yaralarını təmiz alətlə kəsin və qoruyun.', FUNGICIDE, 'Ağacı quraqlıq və günvurmadan qoruyun.'] },
  'apple:rust': { name: 'Alma pası', latin: 'Gymnosporangium spp.', type: 'fungal', status: 1,
    about: 'Yarpağın üzündə parlaq narıncı-sarı ləkələr. Göbələk həyat dövrünü ardıc (Juniperus) kolları ilə alma arasında keçirir.',
    actions: ['Bağın yaxınlığındakı ardıc kollarını yoxlayın, üzərindəki şişləri kəsin.', 'Yazda yarpaq açılarkən qoruyucu fungisid çiləməsi edin.', 'Yoluxmuş yarpaqları toplayın.'] },
  'apple:healthy': HEALTHY('Alma'),
  // Blueberry, raspberry, soybean (healthy only)
  'blueberry:healthy': HEALTHY('Qaragilə'),
  'raspberry:healthy': HEALTHY('Moruq'),
  'soybean:healthy': HEALTHY('Soya'),
  // Cherry
  'cherry:powdery_mildew': { name: 'Unlu şeh', latin: 'Podosphaera clandestina', type: 'fungal', status: 1,
    about: 'Cavan yarpaqlarda ağ, unabənzər örtük; yarpaqlar qıvrılır, böyümə zəifləyir. İsti və rütubətli havada güclənir.',
    actions: [AIRFLOW, 'Azotlu gübrəni artıq verməyin: zərif cavan zoğlar xəstəliyə həssasdır.', 'Erkən mərhələdə kükürd tərkibli və ya qeydiyyatlı fungisidlə çiləyin.', SANITATION] },
  'cherry:healthy': HEALTHY('Albalı'),
  // Corn
  'corn:gray_leaf_spot': { name: 'Boz yarpaq ləkəsi', latin: 'Cercospora zeae-maydis', type: 'fungal', status: 2,
    about: 'Damarlar boyunca uzunsov, düzbucaqlı, boz-qəhvəyi ləkələr. Uzun müddət rütubətli havada və qalıqlar üzərində yayılır.',
    actions: [ROTATION, 'Bitki qalıqlarını şumlayaraq basdırın.', 'Davamlı hibridlərə üstünlük verin.', 'Süpürgələnmə dövründə ləkələr yuxarı yarpaqlara çatırsa, fungisid tətbiqini aqronomla qərarlaşdırın.'] },
  'corn:common_rust': { name: 'Adi pas', latin: 'Puccinia sorghi', type: 'fungal', status: 1,
    about: 'Yarpağın hər iki üzündə qəhvəyi-qırmızı, toz verən pustullar. Sərin və rütubətli havada yayılır.',
    actions: ['Davamlı hibridlər seçin.', 'Yarpaqların 5–10%-dən çoxu yoluxubsa, fungisid tətbiqini nəzərdən keçirin.', 'Pustulların yuxarı yarpaqlara keçməsini həftəlik izləyin.'] },
  'corn:leaf_blight': { name: 'Yarpaq yanığı (helmintosporioz)', latin: 'Exserohilum turcicum', type: 'fungal', status: 2,
    about: 'Uzun (2,5–15 sm), siqarşəkilli, boz-yaşıl, sonra qəhvəyiləşən ləkələr. Məhsulu ciddi azalda bilər.',
    actions: [ROTATION, 'Qalıqları basdırın: göbələk onlarda qışlayır.', 'Davamlı hibrid əkin.', FUNGICIDE] },
  'corn:healthy': HEALTHY('Qarğıdalı'),
  // Grape
  'grape:black_rot': { name: 'Qara çürümə', latin: 'Guignardia bidwellii', type: 'fungal', status: 2,
    about: 'Yarpaqda qara nöqtəli, qəhvəyi dairəvi ləkələr; giləmeyvə büzüşüb qara "mumiyaya" çevrilir.',
    actions: ['Mumiyalaşmış salxımları və yoluxmuş zoğları kəsib məhv edin.', AIRFLOW, 'Çiçəkləmədən əvvəl və sonra qoruyucu fungisid çiləməsi edin.', SANITATION] },
  'grape:esca': { name: 'Eska (qara qızılca)', latin: 'Phaeomoniella, Phaeoacremonium və s.', type: 'fungal', status: 2,
    about: 'Gövdə odununun xroniki xəstəliyi. Yarpaqda damarlar arasında "pələng zolağı" kimi saralma-qızarma; kol qəfil quruya bilər.',
    actions: ['Xəstə kolları işarələyin və budamanı onlardan sonra, ayrıca alətlə edin.', 'Böyük budama yaralarını quru havada açın və qoruyun.', 'Güclü yoluxmuş qolları sağlam oduna qədər kəsin; quru kolları sahədən çıxarın.', 'Bu xəstəliyin effektiv kimyəvi müalicəsi yoxdur: profilaktika əsasdır.'] },
  'grape:leaf_blight': { name: 'Yarpaq ləkəliyi (izariopsis)', latin: 'Pseudocercospora vitis', type: 'fungal', status: 1,
    about: 'Mövsümün sonunda yaşlı yarpaqlarda tünd-qəhvəyi, bucaqlı ləkələr; yarpaqlar vaxtından əvvəl tökülür.',
    actions: [SANITATION, AIRFLOW, 'Mildiu və oidiuma qarşı çiləmə proqramı adətən bu xəstəliyi də məhdudlaşdırır.'] },
  'grape:bacterial_leaf_spot': { name: 'Bakterial yarpaq ləkəsi', latin: 'Xanthomonas spp.', type: 'bacterial', status: 2,
    about: 'Kiçik, yağlı görünüşlü, sonra qaralan bucaqlı ləkələr; ətrafında sarı haşiyə ola bilər.',
    actions: [SANITATION, 'Alətləri kollar arasında dezinfeksiya edin.', COPPER, 'Yağışlı havada bağda iş görməyin: bakteriya su damcıları ilə yayılır.'] },
  'grape:downy_mildew': { name: 'Mildiu (yalançı unlu şeh)', latin: 'Plasmopara viticola', type: 'oomycete', status: 2,
    about: 'Yarpaq üzündə sarımtıl "yağ ləkələri", alt tərəfində ağ pambığabənzər örtük. Yağışdan sonra çox sürətlə yayılır, salxımı məhv edə bilər.',
    actions: ['Təcili: yağışlı dövrdə 7–10 gündən bir qoruyucu çiləmə edin (mis tərkibli və ya qeydiyyatlı sistem preparat).', 'Torpağa yaxın aşağı zoğları və artıq yarpaqları təmizləyin.', AIRFLOW, 'Yağışdan sonra 2–3 gün ərzində bağı yenidən yoxlayın.'] },
  'grape:powdery_mildew': { name: 'Oidium (unlu şeh)', latin: 'Erysiphe necator', type: 'fungal', status: 2,
    about: 'Yarpaq və gilələrdə boz-ağ unlu örtük; gilələr çatlayır. İsti, quru, lakin rütubətli gecələri olan havada yayılır.',
    actions: ['Kükürd tərkibli və ya qeydiyyatlı fungisidlə çiləmə proqramına başlayın.', AIRFLOW, 'Salxım ətrafındakı yarpaqları seyrəldin ki, günəş və hava çatsın.', SANITATION] },
  'grape:healthy': HEALTHY('Üzüm'),
  // Citrus
  'orange:hlb': { name: 'Sitrus yaşıllaşması (HLB)', latin: 'Candidatus Liberibacter spp.', type: 'bacterial', status: 2,
    about: 'Yarpaqda asimmetrik ləkəli saralma, kiçik və eybəcər, acı meyvələr. Sitrus psillidi daşıyır. Müalicəsi yoxdur.',
    actions: ['Diaqnozu laboratoriyada təsdiqlədin və fitosanitar xidmətə məlumat verin.', 'Təsdiqlənmiş ağacları çıxarın: digər ağaclar üçün mənbədir.', 'Psillidə qarşı mübarizə aparın.', 'Yalnız sertifikatlı, sağlam tinglər əkin.'] },
  // Peach
  'peach:bacterial_spot': { name: 'Bakterial ləkə', latin: 'Xanthomonas arboricola pv. pruni', type: 'bacterial', status: 2,
    about: 'Yarpaqda kiçik bucaqlı ləkələr, ortası töküldükdə "güllə dəliyi" görünüşü; meyvədə çatlar.',
    actions: [COPPER, 'Küləkli, qumlu yerlərdə küləkkəsən zolaq salın.', 'Azotu artıq verməyin.', 'Davamlı sortlara üstünlük verin.'] },
  'peach:healthy': HEALTHY('Şaftalı'),
  // Pepper
  'pepper:bacterial_spot': { name: 'Bakterial ləkə', latin: 'Xanthomonas spp.', type: 'bacterial', status: 2,
    about: 'Yarpaqda kiçik, su hopmuş, sonra qəhvəyiləşən ləkələr; yarpaqlar saralıb tökülür.',
    actions: ['Sertifikatlı, sağlam toxum və şitil istifadə edin.', NO_WET, COPPER, ROTATION] },
  'pepper:healthy': HEALTHY('Bibər'),
  // Potato
  'potato:early_blight': { name: 'Alternarioz (erkən yanıq)', latin: 'Alternaria solani', type: 'fungal', status: 1,
    about: 'Aşağı yarpaqlarda konsentrik halqalı ("hədəf" kimi) qəhvəyi ləkələr. Zəifləmiş, qidası az bitkilərdə güclənir.',
    actions: ['Bitkinin qidalanmasını (xüsusən azot və kalium) yoxlayın.', ROTATION, FUNGICIDE, SANITATION] },
  'potato:late_blight': { name: 'Fitoftoroz (gec yanıq)', latin: 'Phytophthora infestans', type: 'oomycete', status: 2,
    about: 'Yarpaq kənarında tünd, su hopmuş ləkələr, alt tərəfdə ağ örtük. Sərin və rütubətli havada bir neçə gündə bütün sahəni məhv edə bilər.',
    actions: ['Təcili: yoluxmuş ocağı işarələyin və qonşu sahəni də yoxlayın.', 'Dərhal qeydiyyatlı fitoftoroz əleyhinə preparatla çiləyin; rütubətli havada intervalı qısaldın.', 'Güclü yoluxmuş kolları çıxarın; yumruları yığmadan 2 həftə əvvəl gövdələri məhv edin.', 'Yığılan yumruları çeşidləyin, xəstələri anbara qoymayın.'] },
  'potato:healthy': HEALTHY('Kartof'),
  // Squash
  'squash:powdery_mildew': { name: 'Unlu şeh', latin: 'Podosphaera xanthii', type: 'fungal', status: 1,
    about: 'Yarpaqların üzündə ağ unlu ləkələr, sonra bütün yarpaq örtülür və quruyur.',
    actions: ['Erkən mərhələdə kükürd tərkibli və ya qeydiyyatlı fungisidlə çiləyin.', AIRFLOW, 'Ən çox yoluxmuş yarpaqları kəsib çıxarın.', 'Davamlı sortlar seçin.'] },
  // Strawberry
  'strawberry:leaf_scorch': { name: 'Yarpaq yanığı', latin: 'Diplocarpon earlianum', type: 'fungal', status: 1,
    about: 'Yarpaqda çoxlu kiçik bənövşəyi ləkələr birləşir və yarpaq yanmış kimi görünür.',
    actions: ['Köhnə və xəstə yarpaqları məhsul yığımından sonra kəsin.', NO_WET, 'Plantasiyanı 3–4 ildən bir yeniləyin.', FUNGICIDE] },
  'strawberry:healthy': HEALTHY('Çiyələk'),
  // Tomato
  'tomato:bacterial_spot': { name: 'Bakterial ləkə', latin: 'Xanthomonas spp.', type: 'bacterial', status: 2,
    about: 'Yarpaq və meyvədə kiçik, tünd, qabıqlı ləkələr; yarpaqlar saralıb tökülür.',
    actions: ['Sağlam, sertifikatlı toxum və şitil istifadə edin.', NO_WET, COPPER, ROTATION] },
  'tomato:early_blight': { name: 'Alternarioz (erkən yanıq)', latin: 'Alternaria solani', type: 'fungal', status: 1,
    about: 'Aşağı yarpaqlarda konsentrik halqalı qəhvəyi ləkələr, ətrafında saralma. Yuxarıya doğru yayılır.',
    actions: ['Aşağı, torpağa dəyən yarpaqları kəsin və mulça çəkin.', NO_WET, FUNGICIDE, ROTATION] },
  'tomato:late_blight': { name: 'Fitoftoroz (gec yanıq)', latin: 'Phytophthora infestans', type: 'oomycete', status: 2,
    about: 'Yarpaqda tünd yaşıl-qəhvəyi, sürətlə böyüyən ləkələr, rütubətdə alt tərəfdə ağ örtük; meyvədə qəhvəyi bərk çürümə.',
    actions: ['Təcili: yoluxmuş bitkiləri çıxarın və sahədən uzaqda məhv edin.', 'Qalan bitkiləri dərhal qeydiyyatlı preparatla çiləyin.', 'İstixanada havalandırmanı artırın, rütubəti azaldın.', 'Kartof sahəsini də yoxlayın: eyni xəstəlikdir.'] },
  'tomato:leaf_mold': { name: 'Yarpaq kifi (kladosporioz)', latin: 'Passalora fulva', type: 'fungal', status: 1,
    about: 'Yarpağın üzündə solğun sarı ləkələr, alt tərəfində zeytuni-qəhvəyi məxməri örtük. Əsasən istixanada, yüksək rütubətdə.',
    actions: ['İstixanada nisbi rütubəti 85%-dən aşağı saxlayın, gecə havalandırın.', 'Aşağı yarpaqları kəsərək hava dövranını artırın.', FUNGICIDE, 'Davamlı sortlar seçin.'] },
  'tomato:septoria': { name: 'Septorioz (ağ ləkəlik)', latin: 'Septoria lycopersici', type: 'fungal', status: 1,
    about: 'Aşağı yarpaqlarda çoxsaylı kiçik, ortası boz-ağ, kənarı tünd ləkələr; yarpaqlar aşağıdan yuxarı quruyur.',
    actions: ['Ləkəli aşağı yarpaqları kəsib çıxarın.', NO_WET, 'Mulça çəkin ki, torpaqdan su sıçramasın.', FUNGICIDE] },
  'tomato:spider_mites': { name: 'Hörümçək gənəsi', latin: 'Tetranychus urticae', type: 'pest', status: 1,
    about: 'Yarpaqda çoxlu xırda açıq nöqtələr, mərmər görünüş, alt tərəfdə incə tor. İsti və quru havada sürətlə çoxalır.',
    actions: ['Yarpağın alt tərəfini lupa ilə yoxlayın: gənələr çox kiçikdir.', 'Bitkilərin su stresinə düşməsinə imkan verməyin.', 'Ocaq halındadırsa, yalnız ocağı qeydiyyatlı akarisidlə işləyin.', 'Faydalı yırtıcı gənələri qoruyun: geniş spektrli insektisidlərdən çəkinin.'] },
  'tomato:target_spot': { name: 'Hədəf ləkəsi', latin: 'Corynespora cassiicola', type: 'fungal', status: 1,
    about: 'Yarpaqda halqalı, ortası açıq qəhvəyi ləkələr; ağır halda yarpaq tökülməsi.',
    actions: [AIRFLOW, SANITATION, FUNGICIDE, ROTATION] },
  'tomato:ylcv': { name: 'Sarı yarpaq qıvrımlığı virusu', latin: 'TYLCV', type: 'viral', status: 2,
    about: 'Cavan yarpaqlar kiçilir, kənarları yuxarı qıvrılır və saralır; bitki bodur qalır, meyvə az olur. Ağqanad daşıyır.',
    actions: [REMOVE, VECTOR, 'İstixana girişlərinə həşərat toru çəkin.', 'Gələn mövsüm davamlı sortlar və təmiz şitil istifadə edin.'] },
  'tomato:mosaic': { name: 'Mozaika virusu', latin: 'ToMV', type: 'viral', status: 2,
    about: 'Yarpaqda açıq və tünd yaşıl mozaik naxış, yarpaq deformasiyası. Əl, alət və toxumla asanlıqla keçir.',
    actions: [REMOVE, 'Xəstə bitkiyə toxunduqdan sonra əlləri və alətləri yuyun, dezinfeksiya edin.', 'Gələn mövsüm sertifikatlı toxum və davamlı sort seçin.', 'Virusun kimyəvi müalicəsi yoxdur.'] },
  'tomato:healthy': HEALTHY('Pomidor'),
  // Cotton
  'cotton:bacterial_blight': { name: 'Bakterial yanıq (qommoz)', latin: 'Xanthomonas citri pv. malvacearum', type: 'bacterial', status: 2,
    about: 'Yarpaqda bucaqlı, su hopmuş, sonra qaralan ləkələr; damarlar boyunca yayılır, qozalar da yoluxa bilər.',
    actions: ['Gələn mövsüm turşu ilə işlənmiş, sağlam toxum istifadə edin.', ROTATION, 'Qalıqları şumla basdırın.', 'Yağışlı dövrdə yayılmanı izləyin; tədbiri aqronomla qərarlaşdırın.'] },
  'cotton:curl_virus': { name: 'Yarpaq qıvrımlığı virusu', latin: 'CLCuD', type: 'viral', status: 2,
    about: 'Yarpaqlar yuxarı və ya aşağı qıvrılır, damarlar qalınlaşır, alt tərəfdə yarpaqcıq çıxıntıları. Ağqanad daşıyır.',
    actions: [VECTOR, 'İlk yoluxmuş bitkiləri erkən çıxarın.', 'Sahə ətrafındakı alaq otlarını (virus mənbəyi) təmizləyin.', 'Davamlı sortlar seçin.'] },
  'cotton:herbicide_damage': { name: 'Herbisid zədəsi', latin: '', type: 'abiotic', status: 1,
    about: 'Yarpaqlar deformasiyaya uğrayır, büzüşür və ya rəngini itirir; adətən çiləmə xəttinə uyğun, sahədə düz zolaqlarla görünür.',
    actions: ['Son çiləmələrin tarixini, preparatını və dozasını yoxlayın.', 'Qonşu sahədən küləklə sürüklənmə ehtimalını araşdırın.', 'Çiləyicini preparat dəyişərkən yaxşıca yuyun.', 'Bitki adətən bərpa olunur: suvarma və qidalanma ilə dəstəkləyin.'] },
  'cotton:jassids': { name: 'Yaşıl sikadka (jassid)', latin: 'Amrasca spp.', type: 'pest', status: 1,
    about: 'Yarpaq kənarlarından saralma, sonra qızarma və qıvrılma ("hopper burn"). Yarpağın alt tərəfində yaşıl, çəp hərəkət edən həşəratlar.',
    actions: ['Yarpağın alt tərəfində həşəratları sayın: hər yarpaqda 2 və çoxdursa, tədbir lazımdır.', 'Tüklü yarpaqlı davamlı sortlara üstünlük verin.', 'Lazım olarsa qeydiyyatlı insektisid tətbiq edin; faydalı həşəratları qoruyun.'] },
  'cotton:leaf_reddening': { name: 'Yarpaq qızarması', latin: '', type: 'abiotic', status: 1,
    about: 'Yarpaqların qırmızımtıl-bənövşəyi rəngə çalması. Çox vaxt maqnezium və ya azot çatışmazlığı, su stresi və ya sərin gecələrlə bağlıdır.',
    actions: ['Torpaq və yarpaq analizi edin (N, Mg, K).', 'Suvarma rejimini yoxlayın: həm quraqlıq, həm artıq su səbəb ola bilər.', 'Çatışmazlıq təsdiqlənərsə, yarpaqdan qidalandırma edin.'] },
  'cotton:leaf_variegation': { name: 'Yarpaq alabəzəkliyi', latin: '', type: 'abiotic', status: 1,
    about: 'Yarpaqda ağımtıl-sarı və yaşıl sahələrin növbələşməsi. Çox vaxt genetik və ya fizioloji səbəblidir və zərərsizdir, lakin virusla qarışdırıla bilər.',
    actions: ['Əlamətin neçə bitkidə olduğunu və yayılıb-yayılmadığını izləyin.', 'Yeni yarpaqlarda qıvrılma və damar qalınlaşması varsa, virus ehtimalı üçün aqronoma göstərin.', 'Qidalanmanı (xüsusən mikroelementləri) yoxlayın.'] },
  'cotton:healthy': HEALTHY('Pambıq'),
  // Not a leaf
  other: { name: 'Bitki yarpağı tanınmadı', type: null, status: null,
    about: 'Şəkildə tanıdığımız bitkilərin yarpağı görünmür. Dron və ya sahə şəkli yükləmisinizsə, "Dron / sahə" rejimini seçin.',
    actions: ['Bir yarpağı kadrın ortasında, 20–30 sm məsafədən çəkin.', 'Gün işığında, kölgəsiz çəkin; fokusun dəqiq olduğuna əmin olun.', 'Xəstə və sağlam hissə eyni kadrda görünsün.'] },
};

export function describe(cls) {
  const [crop] = cls.split(':');
  const k = KB[cls] || KB.other;
  return { ...k, crop: CROPS[crop] || null, cropKey: crop, key: cls };
}

export const GENERAL_NOTE = 'Bu, ilkin AI qiymətləndirməsidir. Kimyəvi preparat tətbiq etməzdən əvvəl yerli aqronomla məsləhətləşin və yalnız Azərbaycanda qeydiyyatdan keçmiş preparatlardan, etiketdəki dozaya uyğun istifadə edin.';

// Field (drone) advice by share of the map
export function fieldAdvice(stats, hot) {
  const out = [];
  // Guard: the method needs a top-down view of a field
  if (stats.vegCover < 5) {
    return [{ s: 1, t: 'Şəkildə bitki örtüyü demək olar ki, yoxdur', d: 'Bu şəkildən sahənin vəziyyətini qiymətləndirmək olmur. Şəkli sahənin üstündən, yuxarıdan çəkin. Sahə yeni əkilibsə, xəritə cücərtilər böyüdükcə məlumatlı olacaq. Tək yarpağı yoxlamaq üçün "Telefon / yarpaq" rejimini seçin.' }];
  }
  if (stats.void >= 35) {
    out.push({ s: 1, t: 'Şəkil yuxarıdan çəkilməyib', d: `Şəklin ${stats.void}%-i səma, su, kölgə və ya yoldur. Yandan çəkilmiş şəkildə rənglər etibarlı deyil. Ən dəqiq nəticə üçün dronla, sahənin düz üstündən çəkin.` });
  }
  if (stats.vegCover < 15) {
    out.push({ s: 1, t: 'Şəkildə bitki örtüyü azdır', d: 'Şəkil sahənin üstündən, yuxarıdan çəkilməlidir. Sahə yeni əkilibsə, xəritə cücərtilər böyüdükcə daha informativ olacaq.' });
  }
  if (stats.bad >= 15) {
    out.push({ s: 2, t: 'Ciddi problemli sahələr var', d: `Bitki örtüyünün ${stats.bad}%-i zəif və ya zədələnmiş görünür. Aşağıdakı zonalara ilk növbədə gedin, yarpaq şəklini telefon rejimində yoxlayın.` });
  } else if (stats.bad >= 5) {
    out.push({ s: 1, t: 'Lokal problem ocaqları', d: `Bitki örtüyünün ${stats.bad}%-i problemli görünür. Ocaqlar böyümədən yerində yoxlayın.` });
  }
  if (stats.risk >= 25) {
    out.push({ s: 1, t: 'Erkən stress əlamətləri', d: `Örtüyün ${stats.risk}%-i sarı zonadadır. Suvarmanın bərabər paylandığını, damcı xətlərinin tıxanmadığını və qidalanmanı yoxlayın.` });
  }
  if (!out.length || (stats.ok >= 70 && stats.bad < 5)) {
    out.push({ s: 0, t: 'Sahə ümumilikdə sağlamdır', d: `Bitki örtüyünün ${stats.ok}%-i sağlam görünür. Növbəti uçuşu 7–10 gün sonra planlaşdırın və dəyişikliyi müqayisə edin.` });
  }
  if (hot.length) {
    out.push({ s: hot[0].bad >= 25 ? 2 : 1, t: 'Yoxlanılacaq zonalar: ' + hot.map((z) => z.id).join(', '), d: 'Bu zonalarda sarı və qırmızı payı ən yüksəkdir. Hər birində 2–3 bitkinin yarpağını yaxından çəkib diaqnoz alın.' });
  }
  return out;
}
