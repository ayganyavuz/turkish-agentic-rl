# VERI-URETIMI

Turkce dogrulanabilir gorev korpusunun uretimi, kalite kapilari ve
dondurulmus split. Bu ayak **lokalde, Docker'li makinede** kosuyor.

Kardes dosya: `EGITIM-ORTAMI.md` (sandbox, GRPO, olcum).
Tarihce: `PROJECT-TRACE.md` (Bolum 0-8, append-only gunluk; bu dosya oradan
turetildi, oras silinmedi).

> Yeni bulgular bu dosyaya **append** edilir; gecmis yeniden yazilmaz.

---

## Amac

Her gorev bir Linux sandbox'inda coklu tur arac cagrisiyla cozulur ve sonuc
**host tarafinda deterministik check'lerle** notlandirilir. "Dogrulanabilir"
olmasinin anlami bu: notlandirma modele degil, calistirilabilir kurallara
dayaniyor.

---

## Mimari (`src/verifiable_dataset/terminal/`, ~3.6k satir)

| Modul | Isi |
|---|---|
| `seeds.py` | Gorev tohumlari: arac demeti, konu, hedef, girdi/cikti bicimi, olcek/bukulme eksenleri |
| `prompts.py` | Uretim promptlari (spec → prompt → referans cozum → check) |
| `generate.py` | Tohumdan aday uretimi; adaylar es zamanli uretilir |
| `derive.py` | Referans cozumu sandbox'ta kosturup **check'leri gozlemden turetir**; `task.yaml`'a yazar |
| `gates.py` | Kalite kapilari (asagida) |
| `checks.py` | 12 deklaratif check op'u: `file_content_eq`, `file_lines_eq`, `file_matches`, `dir_entries_eq`, `json_eq`, `json_field_eq`, `run_stdout_eq`, ... |
| `equiv.py` | Esdegerlik: arsiv / JSON / cok satirli metin **byte** yerine semantik karsilastirilir |
| `task.py` | `task.yaml` yukleme + host tarafinda `grade()`; betikler `set -o pipefail` ile kosar |
| `sweep.py` | Korpusu kosturup gorevleri **banda** ayirir |
| `pipeline.py` | Uctan uca metascript (asagida) |

### Kalite kapilari (`gates.py`)
spec tamligi, ASCII veri, Turkce prompt, prompt↔cikti ortusmesi, arac uyumu,
**determinizm**, **ayirt edicilik**, **ucuz hile** (bos/kopya cozumun
gecmemesi), esdegerlik.

### Odul ve bant tanimi
- `reward`: **ikili** — tum check'ler gectiyse 1.0, aksi halde 0.0
  (`checks.py:373`).
- `partial`: gecen check orani; **yalnizca teshis**, odul degil.
- Bant (`sweep.py:band_of`): pass-rate'e gore `BANT` (ogretici, 1..G-1) /
  `OLU-kolay` (G/G) / `OLU-zor` (0/G) / `BOZUK`.

Bant **egitim tarafinda mufredat** olarak kullaniliyor; olcumun hangi
kosullarda yapilmasi gerektigi `EGITIM-ORTAMI.md`'de (acik kusur).

---

## `pipeline.py` — uctan uca metascript

Asamalar ayri komutlardi ve her biri uc/anahtar/model'i yeniden cozumluyordu;
biri farkli bir uca giderse korpus sessizce karisiyordu. Dordu **tek istemci
ve tek uc** uzerinden kosuyor:

| Asama | Is | API |
|---|---|---|
| `uret` | seed → spec (setup, referans cozum, turetilen check'ler) + spec-only kapilar | evet |
| `prompt` | Turkce gorev metni (referans cozum gosterilmeden) | evet |
| `kapi` | tam gauntlet (prompt kapilari dahil); dusenler karantinaya | hayir |
| `bant` | gorev basina N rollout → BANT / OLU-kolay / OLU-zor | evet |

- **Yeniden calistirilabilir**: her asama isini bitirmis task'lari atlar.
- **Karantina** (`envs_gen/_red`): kapilarda dusen silinmez, kenara alinir.
- **Token sayaci**: asama basina istek/giris/cikis, `--fiyat-*` ile dolar tahmini.

`uret` ve `kapi` asamalari yerel (Docker'siz) modda **reddediliyor** — kapilar
konteyner aciyor.

---

## Saglayici: DeepSeek

Eski uc (`t3ai.baykartech.net/Jailbreak/v1`) HTTP 401 vermeye basladi.
Endpoint on-kontrolu bunu kosu baslamadan yakaladi, para harcanmadi.

Yeni uc **DeepSeek**, model **`deepseek-v4-flash`**. `llm.py` uca gore anahtar
secen bir tabloya (`SAGLAYICILAR`) cevrildi; `.env`'de eski t3ai degerleri
yorum satirinda korundu.

### Kritik bulgu: `deepseek-v4-flash` bir reasoning modeli
Prompt asamasinda 9 task'in 3'u yazilamadi. Belirti yaniltiyordu (denetleyici
"prompt cok kisa, Turkce karakter yok" diyordu). Gercek sebep: model
`max_tokens`'in **tamamini dusunmeye** harciyor, `content` bos donuyordu
(`finish_reason=length`, `reasoning_tokens=1024/1024`). `max_tokens`'i 4096
yapmak cozmedi — onu da tamamen dusunmeye harcadi.

Cozum **`reasoning_effort="none"`**, `llm.py`'da **tek yerde**: `_Completions`
her istege ortak alan enjekte ediyor, ucun **tanimadigi alan sessizce
dusurulup** istek tekrarlaniyor (yerel vLLM bu alani bilmeyebilir).

> Not: `reasoning_effort="none"` DeepSeek'te dusunmeyi kapatiyor ama
> **Qwen3.5'te ciktiyi bozuyor** ("12+17" → 55, 3 token). Qwen tarafinda
> dogru anahtar `chat_template_kwargs={"enable_thinking": False}`.
> Ayrinti `EGITIM-ORTAMI.md`'de.

### Dusunme kapali/acik olcumu (n=10, `uret` asamasi)

| | dusunme acik | dusunme kapali |
|---|---|---|
| Kabul | 9/10 | 9/10 |
| Ortalama deneme | 2.2 | **1.9** |
| 10 aday maliyeti | $0.42 | **$0.016** |

Karar: **uretimde dusunme kapali**. Kabul orani dusmuyor, onarim turu
azaliyor. (n=10 kucuk orneklem; gerekce "kotu degil + cok ucuz".)

`bant` asamasi bu uctan **hic kosulmuyor** — bant egitilecek modelle
olculmeli, DeepSeek'in cozebildigi gorev kumesi Qwen3.5-4B'ninkiyle ayni degil.

### Butce
Onceki "150 aday $6.3" tahmini **tamamen dusunme token'iydi**, gecersiz.
Olculen: 10 aday = 20 istek, 27.6k giris, 7.9k cikis → 400 aday ~$0.65,
prompt asamasi ~$0.24, uctan uca **~$1**.

---

## Kod ailesi (SWE-bench'e yaklasma)

Korpus yalnizca Turkce kabuk gorevlerindendi; olcmek istedigimiz ajanik
kodlamaya uzakti. `seeds.py`'a ikinci seed ailesi eklendi (`aile: "kod"`,
varsayilan `"kabuk"` oldugu icin eski `task.yaml`'lar aynen yukleniyor):

| Hedef | Pay | Ajanin isi |
|---|---|---|
| `hata-bul` | %40 | 2-3 modulun birinde SESSIZ hatayi bul ve duzelt (hep cok dosyali) |
| `veri-yapisi` | %30 | Iskeleti (her metot `raise NotImplementedError`) gerceklestir |
| `yarisma` | %30 | Hedef .py'yi sifirdan yaz (SETUP onu hic olusturmaz) |

Cesitlilik eksenleri: 11 hata tipi, 10 algoritmik kalip, 10 veri yapisi,
zorluk (kolay/orta/zor). Talimat "bu yonde, ayrintiyi sen sec" diyor —
cesitliligin bir kismi bilincli olarak uretici modele birakildi.

**Dogrulama**: `kind: program` + `run:`. Test **task.yaml icinde satir ici**
tasiniyor, calisma dizininden okunmuyor — ajan dosya sistemini degistirerek
odulunu yukseltemiyor.

**Kapi degisikligi**: `arac uygunlugu` kod ailesinde atlaniyor. Referans cozum
`python3`'u komut konumunda cagirmiyor (heredoc ile dosya yaziyor); kural
oldugu gibi birakilsa her aday haksiz yere duserdi.

**`derive.py`**: `run` hicbir sey basmiyorsa net uyari. Gercek bir acikti —
bos stdout her dunyada esit oldugu icin check bozuk dunya ile dogruyu ayirt
edemiyordu.

### Kabul orani
1/6 → 2/6 → 3/8 → **81/200 (%40)**. Yukselten sey soyut ilke degil **mekanik
tarif** oldu (ornegin "SETUP hedef dosyayi HIC olusturmaz").
Hedef bazinda: `veri-yapisi` %57, `hata-bul` %40, `yarisma` %31.
Kabul edilen gorev basina ~$0.009 (kabuk ailesinde ~$0.002).

### Karantinadan cikan iki kusur (kod okunarak bulundu, API'ye para vermeden)
1. **Mutlak `/workspace` yolu** — Docker'da calisir, LocalSandbox'ta calismaz.
   **18 gorev** goreli yola cevrildi.
2. **Test verisi calisma dizinindeki bir `.py`'dan okunuyor**
   (`from test_data import CART`) — ajan o dosyayi degistirip odulu
   kandirabilir. **1 gorev** (`gen-k-0139`) karantinaya alindi.

Ikisi icin de talimata kural eklendi (kural 12 ve 13).

---

## Korpus durumu

| Kume | Sayi |
|---|---|
| `envs/` (elle yazilmis smoke/seed) | 9 |
| `envs_gen/` (kabuk) | 172 |
| `envs_gen_code/` (kod) | 80 |
| **Uretilmis toplam** | **255** |

Kabuk hedefleri: `ozetle, donustur, ayikla, denetle, bol, birlestir,
karsilastir, onar, grupla-say, yeniden-duzenle, arsivle, dogrula, sirala-sec,
toplu-yeniden-adlandir`.

`max_turns` gorev basina degisiyor: **8/10/12/14/16**. 255 gorevin
**199'unda** 12 degil — egitim tarafi duz 12 kullaniyor, bu bir uyusmazlik
(`EGITIM-ORTAMI.md`).

### Referans sagligi (Docker, modelsiz)
- kabuk: **171/175** (dusenler: `gen-s-0001-donustur`, `gen-s-0059-arsivle`,
  `gen-s-0063-bol`, `gen-s-0085-donustur`)
- kod: **80/80**

Yerel modda (Windows/Git Bash) 154/175 — aradaki 17 fark Git Bash'in GNU arac
farklari. **Yerel/Docker esdegerligi Linux'ta yeniden olculmeli**;
Windows'taki karsilastirma yaniltici.

---

## Dondurulmus split — `data/split.json`

| | Gorev |
|---|---|
| train | **173** (kabuk 115, kod 58) |
| held-out | **78** (kabuk 53, kod 25) |
| dislanan | 4 (referansi gecmiyor) |

`(aile, hedef)` tabakalamasi: her tabakada held-out payi %27-40, toplam %31.
Duz rastgele bolme held-out'a orantisiz sayida `yarisma` dusurebilirdi.

Dosya **`tohum=1234`** ile muhurlu, `--force` olmadan uzerine yazilmiyor —
held-out ancak ayni kume kaldigi surece epoch'lar arasi anlam tasir.

**Bu dosyaya dokunmak butun epoch karsilastirmalarini gecersiz kilar.**

---

## Korpusun olculen zorlugu

Uretim tarafinin bilmesi gereken tek sey: uretilen gorevlerin egitilecek
modele gore nerede durdugu. Qwen3.5-4B ile, G=8, held-out 78 gorev:

| Aile | pass@8 | En az bir kez cozulen |
|---|---|---|
| Kod (24) | %32.3 | 17/24 |
| Kabuk (54) | %18.3 | 25/54 |
| **Toplam (78)** | **%22.6** | 42/78 |

Train split'inde (173 gorev, G=8): bantta **103**, olu-zor **65**,
olu-kolay **5**.

> Bu sayilar `--protocol text` ile olculdu. Egitim native tool-calling
> kullaniyor ve olcum o yuzden egitimin dunyasini temsil etmiyor —
> `EGITIM-ORTAMI.md`'deki acik kusura bakin. Sweep native'e cevrildi
> (commit `f93ccf2`) ama **henuz kosulmadi**.

Onceki sweep (154 gorev, kabuk-agirlikli, `data/sweep2.jsonl`): cozulen
36/154, bant 44/154 (%29), ortalama tur 5.1. Bant olcumu **kodda %59,
kabukta %29**.

---

## Bilinen acik konular (uretim tarafi)

- `data/*.log` dosyalari UTF-16 yazilmis; Turkce karakterler bozuk gorunuyor.
- Bazi `reference_solution` alanlari LLM'in "dusunerek" yazdigi coklu deneme
  metinleri iceriyor (orn. `gen-s-0043-dogrula`) — calisiyor ama kirli.
- Kabuk ailesinin bandi kod ailesinin yarisi kadar (%29 vs %59); kabuk
  gorevleri bu model icin belirgin sekilde zor.
- Prompt ve talimatlar **aksansiz** Turkce yazilmis ("gorev", "calistir").
  Model anliyor ve Turkce cevap veriyor, ama bu modelin ciktisini da aksansiz
  Turkce'ye itiyor olabilir. Degistirmek butun baseline'lari gecersiz kilar.
- Uretim **Colab'e tasinmiyor** — Docker'li lokal makinede kaliyor.

---

## Ek (2026-09-10): saglayici bir nesil atladi

`deepseek-v4-flash` uc listesinden dustu, endpoint on-kontrolu kosuyu
baslatmadan reddetti (yine para harcanmadi). Arastirinca sebep cikti:
**ayni gun DeepSeek-V4.1-Flash yayinlandi** ve resmi id ciplak
**`deepseek-flash`** oldu. Eski ad "gecici olarak yonlendiriliyor" deniyor
ama `/models` listesinde artik yok.

> **Kusur (on-kontrol):** `preflight` modeli **listede** ariyor. Yonlendirilen
> ama listelenmeyen bir id, calisacakken reddediliyor. Kontrol listeye degil
> kucuk bir gercek istege bakmali.

`deepseek-v4-pro` pilotlanmadi ve pilotlanmayacak: **14 Eylul 12:00 Pekin
saatinden sonra o da V4.1 Flash'a yonlendirilip Flash fiyatindan
faturalanacak.**

### Fiyat da ayni gun degisti (04:00 UTC)

| | eski V4-Flash | V4.1 Flash |
|---|---|---|
| Giris (cache miss) | $0.22/M | **$0.15/M** |
| Cikis | $0.66/M | **$0.60/M** |
| Giris (cache hit) | $0.007/M | **$0.003/M** |

Hafta ici 01:00-04:00 ve 06:00-10:00 UTC arasi **iki kati**.

### Pilot: V4.1 Flash, n=10, aile=karisik (%60 kod), `uret,prompt,kapi`

| | kayitli v4-flash temeli (kabuk) | V4.1 Flash (karisik) |
|---|---|---|
| Kabul | 9/10 | 6/10 |
| Ortalama deneme | 1.9 | 1.8 |
| Istek (10 aday) | 20 | 29 |
| Giris / cikis token | 27.6k / 7.9k | 67.7k / 16.8k |
| Maliyet | $0.016 | $0.020 (yogun saatte $0.041) |

Iki satir dogrudan kiyaslanmiyor: temel saf kabuk, pilot %60 kod ve kod
ailesinin kayitli kabulu zaten ~%40. Token'in ~2.4x artmasinin bir kismi kod
ailesinin uzun promptlari, bir kismi ekstra onarim turu.

Onemli olan: **bu artik ayni model degil.** Bu tarihten onceki kabul orani,
deneme sayisi ve butce rakamlari V4-Flash'in; asagisi V4.1 Flash'in.

`reasoning_effort="none"` V4.1 Flash'ta da onurlandiriliyor — istek basina
~674 cikis token'i, eski "butun `max_tokens`'i dusunmeye harcama" belirtisi
yok.

### Kosu: 100 aday, karisik (%60 kod), ID 310+
`--asamalar uret,prompt,kapi` (bant kasitli disarida: bant egitilecek modelle
olculmeli). Pilot gorevleri ID 300-309'da, korpusta kaldi.

---

## Ek (2026-09-10): karmasiklik ekseni ve zincirleme gorevler

Korpus tek eksende zordu: bir hedef, bir hata, 2-5 adim, tek `run` testi.
SWE-bench'te bir issue tipik olarak SIRALI kesif gerektiriyor; olcmek
istedigimiz buydu ve korpusta yoktu.

### Zorluk ve karmasiklik ayri eksenler
`bukulme` (kolay/orta/zor) problemin cetinligini soyluyor. Yeni
`KOD_KARMASIKLIK` gorevin KAC ADIMA yayildigini soyluyor. Tek satirlik ama
cetin bir algoritma ile kolay ama uc module yayilmis bir onarim farkli
seyler olcuyor.

| | zincir | regresyon | bagimli | kesif zor | adim | max_turns |
|---|---|---|---|---|---|---|
| `duz` | 0 | - | - | - | 2-5 | 10-16 |
| `orta` | 2 | var | var | - | 5-8 | 16-22 |
| `derin` | 3 | var | var | var | 8-11 | 22-28 |

Dort ozellik bagimsiz bayrak degil seviyeye bagli: serbest birakmak 16
kombinasyon uretir ve cogunun anlami yok (tek dosyada bagimli modul
grafigi gibi). Tek dugme oldugu icin bant ayarlanabiliyor:
`pipeline --karmasiklik`.

Tur butcesi `adim`'dan turedigi icin karmasik gorev otomatik olarak genis
butce aliyor -- bu ancak asagidaki max_turns duzeltmesiyle anlam kazandi.

### Kapatilan kusur: egitim gorev basina max_turns'u okumuyordu
`train_grpo.py`'da tur butcesi YAPICIDA sabitleniyordu; `reset()` gorevi
yukluyor ama `max_turns`'e hic bakmiyordu. 255 gorevin 199'unda deger 12
degildi, yani uzun gorevler bitiremeden kesiliyordu: **gorev zor degildi,
butcesi yanlisti.** Artik her gorev kendi butcesini aliyor;
`--tur-butcesi duz` eski davranisi yeniden uretilebilir tutuyor.

### Yeni kapi: ara durum
Uretici modelden "birbirine bagli 2-3 hata" istemek YETMIYOR. Model pekala
yan yana duran BAGIMSIZ hatalar yazip ayni sozu tutmus gorunur, ve
`task.yaml`'a bakarak bu anlasilmaz. Fark ajan icin buyuk: bagimsiz hatalar
paralel, zincirli hatalar sirali kesif gerektirir.

Olcum spec'in yeni `ARA_ADIM` blogunu kullaniyor (yalnizca ILK hatayi
gideren betik). O dunyada:
- check'lerin HEPSI geciyorsa zincir yok, tek hata var;
- SETUP dunyasindan FAZLA check gecmiyorsa ara adim bir sey ilerletmemis.

Ikisi de degilse ilerleme var ama is bitmemis -- zincir budur.

### Kapatilan kusur: `derive.py` ayni dosyaya iki test yazmayi yasakliyordu
`overrides = {o["path"]: o for o in task.outputs}` bildirimleri **yola gore**
sozlukluyordu; ayni yola ikinci bir `run` bildirimi SESSIZCE siliniyordu.
Regresyon korumasi tam olarak bunu gerektiriyor (bir modulun hem duzeltilen
hem bozulmamasi gereken davranisi), yani ozellik yapisal olarak imkansizdi
ve hata da vermiyordu. `overrides` artik yol -> bildirim listesi.

> Bu kusur cascade'den bagimsiz: veri modeli bir dosyaya iki test yazmayi
> yasakliyordu ve bunu sessizce yapiyordu.

### Pilotlar (n=30, `derin` -> `orta`)

| Pilot | Kabul | Baskin arıza |
|---|---|---|
| derin #1 | 3/30 | prompt celiskisi: temel talimat "dort bolum yaz, baska hicbir sey yazma" diyordu, ARA_ADIM besinciydi |
| derin #2 | 2/30 | tek check turetiliyor (yukaridaki derive kusuru) |
| derin #3 | 1/30 | 18 adayda "ilk duzeltme tek basina hepsini geciriyor" |
| **orta** | **7/30** | ayni, ama gecenler saglam |

**Karar: `derin` (3 halka) uretici modelin kapasitesinin ustunde.** Baskin
ariza modelin uc bagimsiz hata yazip zincir diye sunmasi. `orta` (2 halka)
ayni sirali kesif ozelligini tasiyor ve uretilebiliyor.

Kabul edilen 7 gorevin 7'si de gercek zincir:
`SETUP 0/2 -> ara 1/2 -> referans 2/2`.

Kabul basina $0.015 (duz kod ailesinde ~$0.009) -- ayni mertebe.

### Acik kalan
- `orta` kabul orani %23; kalan dusmelerin buyugu hala "iki bagimsiz hata".
- `envs_gen_code3` bandi **olculmedi**. Karmasikligi artirmak gorevleri
  OLU-zor tarafina itebilir ve cozulmeyen gorev GRPO'da gradyan uretmez.
  Bu korpus egitime girmeden once Qwen3.5-4B ile bant olcumu sart.
- `split.py` varsayilan kaynaklari hala `("envs_gen", "envs_gen_code")`;
  `envs_gen2`, `envs_gen_code2`, `envs_gen_code3` disarida.

### Sonuc: `envs_gen_code3` (2026-09-10)

| | |
|---|---|
| Kabul (100'luk parti) | 19/100, kapilardan 19/19 |
| Klasordeki toplam | **27** (7 pilot + 19 parti + 1 `derin`) |
| Karantina | 128 |
| Zincir dogrulamasi | **27/27** gercek zincir |
| max_turns | 16-26, ortalama 19.1 (eski korpus: 8-16) |
| Maliyet | $0.349 |

Kabul oraninin dusuk olmasi bu ailede beklenen: ara durum kapisi adaylarin
cogunu "iki bagimsiz hata yazmissin" diye eliyor. Kapi olmasa 128'i de
korpusa girer ve karmasik gorundukleri halde sirali kesif gerektirmezlerdi.
Bu partinin kazanci geçenler kadar **elenenlerde**.

### Korpus durumu (gun sonu)

| Klasor | Gorev |
|---|---|
| `envs_gen` (eski kabuk) | 175 |
| `envs_gen_code` (eski kod) | 80 |
| `envs_gen2` (yeni kabuk) | 73 |
| `envs_gen_code2` (yeni kod) | 76 |
| `envs_gen_code3` (zincirleme) | 27 |
| **Toplam** | **431** |

---

## Ek (2026-09-10): yapisal cesitlilik

### Olculen sorun
Kod ailesinde uretilen kodun neredeyse tamami ayni iskeletteydi: metni
ayristir, hesapla, liste/sayi dondur. Sinif durumu, generator, context
manager, ozyineleme, istisna yolu yoktu. LLM'e "gorev uydur" dendiginde
gittigi yer burasi.

### Once olcum aletini duzelt
`fingerprint` kod ailesinde **kordu**: eksenlerinin cogu bu ailede sabit
(`tools` hep python3, op hep `run_stdout_eq`, `cikti` hep program), toplam
19 kova. 183 kod gorevi 33 tekil parmak izine dusuyordu ve en kalabalik
kovada 38 gorev vardi. Dedup calisiyormus gibi gorunup hicbir sey ayirt
etmiyordu.

> Cesitliligi artirdigimizi iddia etmeden once onu OLCEBILIYOR olmak
> gerekiyordu. Parmak izi artik kod ailesinde yapisal eksenleri de goruyor.

### Uc yeni eksen (binlerce ornegi saklamadan)
Elle binlerce seed script toplamak yerine dik eksenlerin carpimi aliniyor.
Dosyada ~50 satir duruyor, ulasilabilir uzay **~18 milyon** kombinasyon.

| Eksen | Deger |
|---|---|
| `PY_KALIPLARI` | 12: duz-fonksiyon, durumlu-sinif, generator, ozyineleme, context-manager, dekorator, dataclass-dogrulama, iterator-protokolu, closure-fabrika, istisna-hiyerarsisi, abc-protokol, operator-asiri-yukleme |
| `VERI_AKISLARI` | 6: bellek-ici, dosya-okuma, cok-dosya-birlestirme, parcali-akis, stdin-boru, ic-ice-yapi |
| `TOPOLOJILER` | 6: tek-modul, iki/uc-modul-zincir, yildiz, elmas, paket-alt-paket |
| `KOD_KONULARI` | 10 -> 30 |

`kod_uyumlu()` anlamsiz eslesmeleri eliyor (%14): bellek-ici veride context
manager'in yonetecegi kaynak yok, tek modulde "hatayi bul" okumaya degil
goze indirgenir.

Eksenler prompt'ta **baglayici** yazildi (`ayrinti` gibi serbest degil).

### Yeni kapi: yapisal uyum
Talimat vermek yetmedi -- pilotta 19 gorevin 4'unde model kalibi yoksayip
duz fonksiyon yazdi. Kapi seed'in istedigi kalibin kodda gercekten
oldugunu kontrol ediyor. Guvenilir belirteci olmayan kaliplar
(`closure-fabrika`, `ozyineleme`) bilerek ATLANIYOR: kirilgan bir desenle
dogru adayi elemek, birkac uyumsuzu gecirmekten kotu.

> Kapi once duzenli ifadeyle yazildi ve desenler bozuldu (`\b` gercek
> backspace karakterine dondu); kapi 7 gorevi yanlislikla eledi. Duz alt
> dize kontrolune cevrildi. **Ders: kapi sessizce yanlis olcerse dogru
> adaylari eler ve bu kabul oranina bakarak fark edilmez.**

### Sonuclar

| | Kabul | Yapisal uyum redleri | Maliyet |
|---|---|---|---|
| Pilot (n=30) | 19/30 (%63) | - | $0.060 |
| `envs_gen_code4` (n=50) | 30/50 (%60) | 10 | $0.111 |
| `envs_gen_code5` (n=50) | 31/50 (%62) | 14 | $0.109 |

Taban (yapisal eksen yokken, saf kod) %55 ve ortalama 1.8 denemeydi.
**Baglayici eksenler kabul oranini dusurmedi, yukseltti** (deneme 1.8 ->
1.4); daha spesifik talimat modelin daha az bocalamasina yol aciyor.

### Cesitlilik kaniti

| | Gorev | Tekil parmak izi | En kalabalik kova |
|---|---|---|---|
| Eski kod korpusu | 183 | 33 (%18) | 38 |
| `code4` + `code5` | 81 | 79 (**%98**) | 2 |

Uc eksen de dengeli doldu. Model hala her dort-bes adayin birinde kalibi
yoksayiyor; kapi olmasa bu rakam gercegi yansitmazdi.

### Korpus (gun sonu, 2026-09-10)

| Klasor | Gorev |
|---|---|
| `envs_gen` / `envs_gen_code` (eski) | 175 / 80 |
| `envs_gen2` / `envs_gen_code2` | 73 / 76 |
| `envs_gen_code3` (zincirleme) | 27 |
| `envs_gen_code4` / `envs_gen_code5` (yapisal) | 50 / 31 |
| **Toplam** | **512** |

Gun basi 255. Toplam API harcamasi ~$1.8.

### Acik kalan (oncelik sirasiyla)
1. **512 gorevin HICBIRININ bandi olculmedi.** Cozulmeyen gorev GRPO'da
   gradyan uretmiyor; daha fazla uretmeden once mevcudun ne kadarinin
   egitilebilir bantta oldugu olculmeli. Bu adim API parasi harcamiyor.
2. Zincirleme gorevlerde odul hala ikili: `ara 1/2` durumuna ulasmak
   hicbir sey yapmamakla ayni odulu aliyor. Gorev bilerek "ilerleme
   kaydedilebilir" diye tasarlandi, sonra ilerleme odulde gorunmez
   birakildi. Bant olcumunden SONRA karar verilmeli.
3. `split.py` varsayilan kaynaklari hala `("envs_gen", "envs_gen_code")`;
   dort yeni klasor disarida.
