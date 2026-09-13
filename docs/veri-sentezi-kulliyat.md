# Multi-turn agentic RL icin veri kurasyonu ve sentezi

Derleme tarihi: 2026-09-03. Bu depodaki uretim hattinin (`generate.py` ->
`gates.py` -> `derive.py` -> `prompts.py` -> `sweep.py`) literaturdeki
karsiliklari.

Dogrulama notu: asagidaki iddialari makale sayfalarindan teyit ettim.
Iki istisna isaretli -- **AutoForge** ve **SPADE**'den yalnizca ozet
duzeyinde bilgi cekilebildi (PDF metin cikarimi basarisiz oldu), o
ikisinin ayrintilarini kullanmadan once kendiniz dogrulayin.

Okuma sirasi onerisi: R2E-Gym -> Prime Intellect blogu -> TaskCraft ->
APIGen-MT -> Practitioner's Guide. Ilk ikisi bu hafta uygulanabilir,
kalani sonraki genisletme icin.

---

## A. Gorev/spec sentezi -- `generate.py`'nin akrabalari

### 1. TaskCraft: Automated Generation of Agentic Tasks (ICLR 2026)

<https://arxiv.org/abs/2506.10055> · kod: <https://github.com/OPPO-PersonalAI/TaskCraft>

En yakin akraba. Atomik gorevleri **derinlik** (bir gorevin ciktisini
baska bir gorevin girdisi yapmak) ve **genislik** (bagimsiz alt gorevleri
birlestirmek) ekseninde buyuterek zorlugu olceklenebilir kiliyor.
Dogrulama artimli: rejection sampling + LLM'e dilsel analiz yaptirma.
41k arac-yogun gorev, 12.6k etkilesim yorungesi, 5k cok adimli ayristirma
yayinlanmis.

**Bagi:** `seeds.py`'deki `adim` ekseni TaskCraft'in derinlik
genisletmesinin ilkel hali. Onlar bunu uretilen gorevden gorev tureterek
yapiyor, biz seed'de sabit sayi olarak veriyoruz. Zorluk bandi
`OLU-kolay` tarafinda siserse dogrudan uygulanabilir.

### 2. APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation (NeurIPS 2025 D&B)

<https://arxiv.org/abs/2504.03601> · site: <https://apigen-mt.github.io/>

Iki fazli: once **ground-truth eylemleri olan gorev planlari
(blueprint)** uretiliyor, bunlar bir **LLM hakem komitesinden** ve
yinelemeli geri besleme dongusunden geciyor; sonra onaylanmis planlar
simule edilmis insan-ajan etkilesimiyle tam yorungeye cevriliyor. xLAM-2
modelleri tau-bench ve BFCL'de GPT-4o ve Claude 3.5'i gecmis, 5K yorunge
acik.

**Bagi:** Yapisal olarak bizim hattimizla ayni -- plan uret, denetle,
sonra dogal dile cevir. Kritik fark bizim lehimize: onlarin denetimi LLM
hakem komitesi, bizimki `gates.py`'de **model cagirmayan deterministik
kapilar**. Bizimki hem ucuz hem de hakemin kendi hatasini tasimiyor.

### 3. COVERT: Controllable and Verifiable Tool-Use Data Synthesis for Agentic RL

<https://arxiv.org/html/2604.09813v1>

Iki asamali. Once guvenilir taban yorungeler: LLM uretici seed
havuzundan orneklerle aday uretiyor, **uc dogrulama katmani**
(deterministik format kontrolu, kural tabanli sema dogrulama, LLM
yargiciyla anlamsal degerlendirme) gecenler seed havuzuna geri
besleniyor. Sonra **oracle-koruyan bozmalar**: sorgu, arac seti ve
ciktilar alti boyutta bozuluyor ama odul dogrulamasi icin gereken oracle
alanlari korunuyor -- dikkat dagitici araclar, sorgu yeniden yazimi,
coklu format, gurultulu cikti, hatali cikti, muglak sorgu.

**Bagi:** `seeds.py`'deki `direnc` ve `bukulme` eksenleri
(`tuzak-dosya`, `baslik-yok`) tam olarak COVERT'in bozma boyutlarinin
karsiligi. "Oracle'i koru, cevreyi boz" ilkesi bizde de gecerli: check
turetildikten sonra setup'i zorlastirmak check'i bozmuyor.

### 4. AutoForge: Automated Environment Synthesis for Agentic RL

<https://arxiv.org/abs/2512.22857>

Ortam sentezi -> arac dizisi uretimi -> gorev uretimi seklinde birlesik
hat, ustune ortam-seviyesi RL algoritmasi.

> Uyari: yalnizca ozet duzeyinde bilgi cekilebildi.

---

## B. Calistirilabilir ortami olcekleme -- `derive.py` ve Docker tarafi

### 5. R2E-Gym: Procedural Environments and Hybrid Verifiers (COLM 2025)

<https://arxiv.org/abs/2504.07164> · kod: <https://github.com/R2E-Gym/R2E-Gym>

Bu kulliyattaki **en onemli fikir**: insan yazimi issue ve unit test'e
bel baglamak yerine, **commit'lerden test uretimi ve geri-ceviri
(back-translation)** ile calistirilabilir gorev kurasyonu. 8K+
calistirilabilir gorev.

Hibrit dogrulayici: calistirma-tabanli dogrulayicilar ayirt ediciligi
dusuk, calistirma-serbest olanlar yanli; ikisi birlesince SWE-Bench
Verified'da %51, tek baslarina ~%42-43.

**Bagi:** `derive.py` zaten geri-ceviri yapiyor -- referans cozumu
calistirip sonucundan check turetiyoruz. Ayni fikir, farkli alan.

### 6. Prime Intellect: Scaling Agentic RL -- 365,000+ Environments

<https://www.primeintellect.ai/blog/scaling-agentic-rl>

Makale degil ama muhendislik acisindan en ogretici parca, ve **terminal
gorevleri** (~28.6k) bizim alanimiz. Dort asamali dogrulama hatti:

1. **No-op dogrulamasi** -- gorev, hicbir duzeltme yapilmadan testlerden
   dusmeli
2. **Altin yama dogrulamasi** -- referans cozum testleri gecmeli,
   **10x tekrarla**, boylece kirik gorevle kararsiz (flaky) gorev ayrilir
3. Bagimsiz ikinci dogrulama gecisi
4. Dislananlarin kaydini tutarak yeniden yukleme

Sonuclar acimasiz: SWE-rebench-V2 32.079 satirdan 6.275'e dusmus;
Multi-SWE 4.703'ten 2.232'ye ("sifir duzenlemeyle cozulmus sayilan"
gorevler atilarak); Scale-SWE 20.181'den 17.202'ye.

Odul guvenilirligi uzerine iki baslik:

- **Yanlis pozitifler:** ajan grading makinesiyle ayni sandbox'ta, "RL
  baskisi altindaki bir politika eninde sonunda kalan dikisi bulur".
  Grading malzemesini puanlama anina kadar sakliyorlar; bunun "garanti
  degil, azaltma" oldugunu kendileri soyluyor.
- **Yanlis negatifler:** PR'lardan cikarilan testler dogru davranisi
  degil implementasyon ayrintisini siniyor; esit derecede gecerli
  alternatif cozumler dusuyor. Olcekte "egitim sinyalinde saf gurultu".

**Bagi uc yerden:** `runner.py:153`'teki `_assert_not_pre_solved` tam
olarak onlarin 1. asamasi. Referansin gecmesi sarti 2. asama -- ama biz
**tek kez** kosuyoruz, onlar 10x. Yanlis negatif sorunu icin asagiya
bakin.

### 7. SWE-smith

Prime Intellect blogunda ve R2E-Gym karsilastirmalarinda geciyor.

Kod tabanlarindan dogrudan hata-duzeltme gorevi sentezleyen
olceklenebilir hat; 128 depodan 50K ornek. Olcek referansi olarak
yararli: bizim 154'umuz bu baglamda cok kucuk.

---

## C. Kendi mufredatini ureten sistemler

### 8. Absolute Zero: Reinforced Self-play Reasoning with Zero Data

<https://arxiv.org/abs/2505.03335>

Tek model hem **oneren** hem **cozen**. Kendi ogrenme ilerlemesini
maksimize eden gorevleri oneriyor, bir kod calistirici hem onerilen
gorevi dogruluyor hem cevabi denetliyor. Hic dis veri yok.

**Bagi:** Bizde oneren (uretici model) ve cozen (politika) ayri ve oneren
sabit. Absolute Zero'nun katkisi bu donguyu kapatmak -- band egitim
ilerledikce `OLU-kolay`a kayacagi icin er gec ihtiyac olacak.

### 9. SPADE: Self-Play in Adaptive Synthetic Executable Environments

<https://arxiv.org/pdf/2608.19197>

Tek LLM iki rol: **Ortam Tasarimcisi** Gym arayuzlu calistirilabilir kod
olarak uzun ufuklu ortamlar yaziyor, **Akil Yuruten Ajan** onlarda
ogreniyor. Zorluk, cozucunun performansina gore uyarlaniyor.

> Uyari: yalnizca yuzeysel ozet cekilebildi.

### 10. R-Zero: Self-Evolving Reasoning LLM

<https://arxiv.org/pdf/2508.05004>

Meydan okuyan-cozucu ko-evrimi. Absolute Zero ile birlikte okunacak
eslik makalesi.

---

## D. Zorluk kalibrasyonu ve filtreleme -- `sweep.py`

### 11. A Practitioner's Guide to Multi-turn Agentic RL

<https://arxiv.org/abs/2510.01132>

Ortam, odul ve politika uc diregi; TextWorld, ALFWorld, SWE-Gym
ortamlari. PPO ve GRPO (yanli) ile RLOO'yu (yansiz) odul seyrekligiyle
etkilesim icinde degerlendiriyorlar.

- **Yogun tur-seviyesi oduller egitimi hizlandiriyor ama kararlilik
  algoritma secimine bagli.**
- "Bir alandaki basit ortamlar bile ajanin daha karmasik gorevlere ne
  kadar genelleyecegi konusunda sinyal verir" -- 30'luk orneklem
  yaklasimimizin dogrudan gerekcesi.

### Sifir-varyans filtrelemesi

Ayri bir makale degil, alanin yerlesmis pratigi: sorgu basina K=8 rollout
uretip yalnizca 1<=c<=7 dogru olanlari tutmak, c=8'i (bedava, sifir
gradyan) ve c=0'i (imkansiz) atmak.

**Bagi:** `sweep.py:27`'de `BAND_LOW, BAND_HIGH = 0.125, 0.875` yaziyor.
0.125 = 1/8, 0.875 = 7/8. Yani zaten tam olarak bu kural uygulanmis,
literaturdeki esiklerle birebir ayni.

---

## Bu kulliyatin depoda isaret ettigi iki somut bosluk

### 1. Denklik kapisi hic calismiyor

`envs_gen`'deki **154 task'in 0'inda** `alt_solutions` alani var
(154'unde gecen sey sadece "sonraki asamada eklenecek" yorumu). Elle
yazilmis `envs/` tarafinda 8/8 dolu. `gates.py:227` `alt_solutions`
yoksa denklik kapisini `ATLA` diye geciyor -- yani uretilmis task'larin
hicbiri bu kapidan gecmemis.

Bu, Prime Intellect'in "yanlis negatif" basliginin ta kendisi: check
referansin urettigi tek bir ciktiyi siniyor, esit derecede dogru bir
alternatif cozum dusuyor ve model dogru davrandigi halde ceza yiyor.
`equiv.py` bu is icin yazilmis ve bos duruyor.

Gozlem: 2026-09-03 kosusunda `gen-s-0006-toplu-yeniden-adlandir`
`kismi=0.80` ile 0/2 dustu -- tam bu suphe. Model isi yapmis olabilir,
check bicimi tutturamamis olabilir.

### 2. Referans tek kez kosuyor

Prime Intellect kararsiz gorevi kirik gorevden ayirmak icin altin cozumu
10x kosuyor. Bizde `--reference` bir kez kosuyor; `gates.py`'deki
determinizm kapisi bir kismini yakaliyor ama ayni sey degil.

### 3. Ince taneli olmayan odul

Check sayisi task basina degisiyor (cogunda 1, bazilarinda 3-6), yani
`partial` bir miktar bilgi tasiyor -- ama `task.py`'de `reward` yine de
ikili, `partial` yalnizca teshis icin tutuluyor. Practitioner's Guide'in
"yogun tur-seviyesi odul" bulgusuna gecmek icin `partial`'i odule
baglamak ve check turetmeyi coklamak gerekiyor;
`data/yeniden_turet.log`'da "elle yazilanlardan 87 check kacirildi"
yaziyor.

### 4. Uretilen referans cozumun DOGRULUGU hic denetlenmiyor

2026-09-03 olcumu. Gauntlet referansin *calistigini*, *deterministik*
oldugunu ve *baslangicta cozulmus olmadigini* dogruluyor; ama ciktisinin
prompt'un istedigi sey olup olmadigini hicbir kapi sormuyor. Uretici
model klasik bash hatalari yapiyor ve ortaya cikan cop ground truth
oluyor:

- `gen-s-0016-ayikla`: referans `head`/`tail`'i coklu dosyada `-q`'suz
  cagiriyor, `==> dosya <==` basliklari ground truth'a sizmis
- `gen-s-0030-bol`: `cut -d '"' -f4,12` yanlis alani kesiyor, beklenen
  cikti `SH101"destination` (alanin degeri degil ADI)
- `gen-s-0015-donustur`: "sayisal degerlerin toplami" isteniyor, gercek
  cevap 2189.8; ground truth 105562
- `gen-s-0012-donustur`: ground truth gecersiz JSON (nesneler arasi
  virgul yok), dogru JSON ureten cozum duser

Statik tarama 154 task'in 35'inde (%23) bu tur belirti buluyor. Eksik
olan kapi tam olarak `equiv.py`/`alt_solutions`: referansi gormeyen
bagimsiz bir cozucunun ciktisi ground truth'tan farkliysa task
supheliidir.
