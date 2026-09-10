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
