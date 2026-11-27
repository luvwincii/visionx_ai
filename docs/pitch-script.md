# VisionX: pitch mətni

Təxminən 4 dəqiqə. Mötərizədəki yerləri öz məlumatınızla doldurun.

## 1. VisionX

Salam, mənim adım [ad], VisionX-in təsisçisiyəm. Təsəvvür edin ki, 50 hektar pomidor sahəniz var. Bir səhər sahənin bir küncündə yarpaqlarda qəhvəyi ləkələr görürsünüz. İlk sual budur: bu ləkələr nə vaxtdan oradadır və artıq nə qədər sahəyə yayılıb? VisionX məhz bu suala cavab verir. Dron və ya telefonla çəkilmiş adi şəkli sahənin sağlamlıq xəritəsinə çevirir və problemin harada olduğunu göstərir.

## 2. Problem

Fermerin əsas problemi məlumatın gec gəlməsidir. Onlarla hektarlıq sahəni hər gün piyada gəzib yoxlamaq mümkün deyil. Xəstəlik, zərərverici və ya susuzluq adətən sahənin kiçik bir hissəsində başlayır və fermer onu çox vaxt artıq yayıldıqdan sonra görür. Problemin dəqiq yeri bilinmədiyi üçün dərman bütün sahəyə vurulur. Nəticədə fermer üç tərəfdən itirir: məhsulun bir hissəsi gedir, dərmana artıq pul xərclənir, torpaq və ətraf mühit isə lazımsız kimyəvi yük alır.

## 3. Həll

Biz bu prosesi tərsinə çeviririk. Fermer problemi axtarmır, problem özü onun telefonuna gəlir. Dron sahənin üstündən uçur, sistemimiz şəkli analiz edir və sahəni üç rəngə bölür. Yaşıl sağlam hissələri, sarı risk altında olan hissələri, qırmızı isə dərhal baxılmalı olan yerləri göstərir. Fermer bütün sahəni deyil, yalnız qırmızı zonaları yoxlayır və dərmanı da yalnız orada istifadə edir.

## 4. Necə işləyir

İş prosesi üç addımdan ibarətdir. Birinci addımda dron sahənin üstündən uçur. Dronu olmayan fermer isə xəstə görünən yarpağı telefonla çəkə bilər. İkinci addımda süni intellekt modelimiz şəkli bir neçə saniyə ərzində təhlil edir. Üçüncü addımda tətbiq nəticəni azərbaycan dilində izah edir: problem nədir, harada yerləşir və fermer nə etməlidir. Fermerin əlavə sualı olsa, nəticə ekranındakı AI köməkçidən soruşa bilər. Köməkçi Gemini modeli ilə işləyir və VisionX-in nəticəsinə əsaslanaraq azərbaycan dilində cavab verir.

## 5. Məhsul

Platformanı artıq qurmuşuq və o işləyir. Dron rejimində fermer sahənin xəritəsini görür. Sahə zonalara bölünür və hər zonada problemin neçə faiz olduğu göstərilir. AI düyməsi ilə model problemin səbəbini də tapır: quruma, su yığılması, alaq, qida çatışmazlığı və ya əkin boşluğu. Telefon rejimində fermer yarpağın şəklini çəkir və xəstəliyin adını, modelin nə dərəcədə əmin olduğunu və görülməli addımları alır. Hər iki rejimdə fermer nəticə barədə AI köməkçiyə sual verə bilər. Bir vacib məqam da var: analiz birbaşa telefonda aparılır. Şəkil serverə göndərilmir və internet olmayan sahədə də sistem işləyir. Kənd yerləri üçün bu, xüsusilə önəmlidir.

## 6. Sübut: yarpaq

Modelin keyfiyyətini real şəkillər üzərində yoxladıq. Açıq yayılan modellər laboratoriyada çəkilmiş şəkillərdə yaxşı nəticə göstərir. Lakin fermerin sahədə çəkdiyi şəkillərdə onların dəqiqliyi kəskin düşür. Biz sahədə çəkilmiş 637 şəkil üzərində test apardıq. Bu şəkillər modelin öyrədilməsində istifadə olunmayıb. Ən yaxşı müstəqil açıq model bu şəkillərin 39 faizində düzgün diaqnoz qoydu, bizim model isə 75 faizində. Üç ən ehtimallı diaqnoz arasında düzgün cavab 94 faiz hallarda olur. Hazırda model 15 bitki üzrə 48 diaqnozu tanıyır. Bunlara Azərbaycan üçün vacib olan pambıq və üzüm də daxildir.

## 7. Sübut: dron AI

Dron rejimində yalnız sağlam və problemli zonaları göstərmək kifayət deyil. Fermer problemin səbəbini də bilmək istəyir. Bunun üçün ayrıca seqmentasiya modeli öyrətdik. Model şəkildə beş problem növünü axtarır: quruma, su yığılması, alaq, qida çatışmazlığı və əkin boşluğu. Modeli 57 min sahə şəkli üzərində, Kaggle-da yeddi saata yaxın öyrətdik. Modelin heç görmədiyi 800 şəkil üzərində yoxladıq: dəqiqlik göstəricisi əvvəlki modelimizdə 0,34 idi, yeni modeldə 0,56-dır. Sahəni zonalara bölüb baxanda model problemli zonaların 76 faizini düzgün tapır. Su yığılması və quruma ən etibarlı siniflərdir: model bu zonaları işarələyəndə hallarının 75–95 faizində həqiqətən problem olur. Analiz birbaşa telefonda, təxminən beş saniyəyə aparılır. Növbəti addım Azərbaycan sahələrindən toplanan şəkillərlə modeli yerli şəraitə uyğunlaşdırmaqdır.

## 8. Biznes model

Gəlir modelimiz üç istiqamətdən ibarətdir. Birincisi, dron paketidir. Fermerə və aqroparka ucuz RGB dron, quraşdırma və təlim satırıq. Paketin qiyməti 2 500 manatdır, dronun özü təxminən 2 000 manata başa gəlir. İkincisi və əsas gəlir mənbəyi analiz abunəsidir: mövsüm üçün hektar başına 15 manat. Məsələn, 100 hektarlıq təsərrüfat mövsümdə 1 500 manat ödəyir. Üçüncüsü dron xidmətidir. Dron almaq istəməyən kiçik fermer üçün özümüz uçuruq: mövsümdə 12 uçuş və analiz, hektar başına 30 manat. Bir dronla on təsərrüfata xidmət edə bilirik. Müqayisə üçün: ABŞ-da dron xidməti bir uçuş üçün hektar başına 20–37 manatdır, Taranis kimi platformanın abunəsi isə hektar başına təxminən 63 manatdır. Peyk xidmətləri daha ucuzdur, amma kiçik problemləri görmür.

## 9. Bazar

Dövlət Statistika Komitəsinin məlumatına görə, 2023-cü ildə Azərbaycanda əkin sahəsi bir yarım milyon hektar olub. Bunun 215 min hektarı Qarabağ və Şərqi Zəngəzur iqtisadi rayonlarının payına düşür. Ölkədə 1 203 kənd təsərrüfatı müəssisəsi fəaliyyət göstərir. Hektar başına 15 manatlıq abunə ilə hesablasaq, bütün Azərbaycan üzrə illik bazar 22,5 milyon manat, təkcə Qarabağda isə 3,2 milyon manatdır. Biz real hədəf qoyuruq: üç il ərzində Qarabağdakı sahələrin təxminən 5 faizini, yəni 10 min hektarı əhatə etmək. Bu, yalnız abunədən ildə təxminən 150 min manat gəlir deməkdir, dron paketləri və xidmət bundan əlavədir. İşə Qarabağdan başlayırıq, çünki orada yeni sahələr və aqroparklar salınır, kənd təsərrüfatı yenidən qurulur. Dövlət də ağıllı texnologiyalar üçün subsidiya mexanizmi hazırlayır. Yeni texnologiyanı ilk gündən tətbiq etmək üçün ən uyğun yer oradır.

## 10. Rəqabət

Bu gün fermerin sahəsini izləmək üçün bir neçə yolu var, amma hər birinin çatışmazlığı var. Peyk şəkilləri gecikmə ilə gəlir və kiçik detalları göstərmir. Multispektral dronlar dəqiqdir, amma belə dronun qiyməti 8 000 manatdan başlayır. Aqronom xəstəliyi tanıya bilir, lakin hər səfər vaxt və pul tələb edir. VisionX isə təxminən 2 000 manatlıq adi dron və ya telefonla işləyir, yəni avadanlıq dörd-beş dəfə ucuzdur. Həm də iki suala birdən cavab verir: problem haradadır və nədir. Yarpaqda xəstəliyin adını, dron şəklində isə problemin səbəbini göstərir.

## 11. Yol xəritəsi

Hazırda MVP tam hazırdır: sayt və platforma işləyir, yarpaq diaqnozu və dron səbəb modelləri öyrədilib və test olunub, AI köməkçi işləyir. Növbəti mərhələdə Qarabağda pilot fermerlərlə işləməyə başlayırıq. Onların sahələrindən toplanan yerli şəkillərlə modeli yenidən öyrədəcəyik ki, Azərbaycan şəraitində daha dəqiq işləsin. Bundan sonra buğda üçün ayrıca model əlavə edəcəyik, abunə və dron paketlərinin satışına başlayacağıq.

## 12. Təklif

Bizim təklifimiz kiçik və ölçülə bilən bir pilot mövsümüdür. Qarabağda 300 hektar sahəni aprel-sentyabr ərzində hər iki həftədən bir izləmək istəyirik. Bunun üçün 12 000 manat lazımdır: iki ucuz dron dəsti təxminən 4 000 manat, Azərbaycan sahələrindən şəkillərin toplanması və aqronomla etiketlənməsi 3 000 manat, sahə səfərləri 3 000 manat, server və digər xərclər 2 000 manat. Bir ucuz dron gündə 60–100 hektar çəkə bildiyi üçün 300 hektarı bir-iki günə tam əhatə edirik. Pilotda dörd fermer və bir aqroparkla işləmək istəyirik. Məqsədimiz bir mövsümdə real nəticə göstərmək: dərmana qənaət, problemin erkən tapılması və yerli şəkillərlə daha dəqiq model. Maliyyədən əlavə üç şey istəyirik: Çindəki sərgidə iştirak, orada dron istehsalçıları və tərəfdaşlarla görüş; Qarabağ aqroparkları ilə tanışlıq; və ağıllı texnologiyalar üçün planlaşdırılan subsidiya mexanizminə qoşulmaq üçün nazirliklə əlaqə.

## 13. Komanda

VisionX dörd nəfərlik komandadır. Elxan Ağakişiyev Product Owner və AI arxitektorudur: məhsulun istiqamətini müəyyən edir və AI modellərini qurur. İnci Abdulazizova Project Manager-dir: planı, pilotu və müddətləri idarə edir. Əli Səfərov Business Analyst-dir: bazarı, qiymət modelini və tərəfdaşlıqları hazırlayır. Ömər Sultanlı mühəndisimizdir: tətbiqi və dron axınını texniki tərəfdən qurur. Hər birimizin LinkedIn profili slayddadır. Diqqətinizə görə təşəkkür edirik. İstəsəniz, platformanı indi canlı göstərə bilərik.
