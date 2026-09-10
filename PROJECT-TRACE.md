# PROJECT-TRACE

Bu dosya projenin karar ve calisma gunlugudur. Yeni bolumler **append** edilir,
gecmis bolumler yeniden yazilmaz. Her bolum tarih + baslik ile acilir.

---

## Bolum 0 — Su ana kadar yapilanlar (ozet, 2026-09-09 itibariyla)

### Amac
Turkce, **dogrulanabilir** (verifiable) terminal/arac-kullanimi gorevlerinden olusan
bir RL veri seti uretmek. Her gorev, bir Linux sandbox'inda coklu tur arac cagrisiyla
cozulur ve sonuc host tarafinda deterministik check'lerle notlandirilir.

### Mimari (`src/verifiable_dataset/terminal/`, ~3.6k satir)

| Modul | Isi |
|---|---|
| `seeds.py` | Gorev tohumlari: arac demeti, konu, hedef, girdi/cikti bicimi, olcek/bukulme eksenleri |
| `prompts.py` | LLM'e verilen uretim promptlari (spec → prompt → referans cozum → check) |
| `generate.py` | Tohumdan aday gorev uretimi; adaylar es zamanli uretilir, key endpoint'e gore secilir |
| `derive.py` | Referans cozumu sandbox'ta calistirip **check'leri gozlemden turetme**; turetilen check'ler `task.yaml`'a yazilir |
| `gates.py` | Kalite kapilari: spec tamligi, ASCII veri, Turkce prompt, prompt↔cikti ortusmesi, arac uyumu, **determinizm**, **ayirt edicilik**, **ucuz hile** (bos/kopya cozumun gecmemesi), esdegerlik |
| `checks.py` | 12 deklaratif check op'u: `file_content_eq`, `file_lines_eq`, `file_matches`, `dir_entries_eq`, `json_eq`, `json_field_eq`, `run_stdout_eq`, ... |
| `equiv.py` | Esdegerlik: arsiv / JSON / cok satirli metin **byte** yerine semantik karsilastirilir |
| `sandbox.py` | Docker sandbox — episode basina bir konteyner, turler arasi FS durumu korunur, `docker exec` ile calisir, `--network none`, 1 cpu / 512m / pids 256 |
| `task.py` | `task.yaml` yukleme + host tarafinda notlandirma (`grade`); betikler `set -o pipefail` ile kosar |
| `runner.py` | Episode dongusu: native tool-calling **veya** `<komut>` metin protokolu (Gemma gibi tool'suz modeller icin); dusunce zinciri (`reasoning_content`) ve komut ciktisi ekrana basilir |
| `sweep.py` | Tum korpusu kosturup gorevleri **banda** ayirir |

### Odul ve bantlama
- `reward`: **ikili** — tum check'ler gectiyse 1.0, aksi halde 0.0.
- `partial`: gecen check orani; **yalnizca teshis icin** tutuluyor, odul degil.
- Bant (`sweep.py:band_of`): pass-rate'e gore `BANT` (ogretici) / `OLU-kolay` (hep cozuluyor)
  / `OLU-zor` (hic cozulemiyor) / `BOZUK`.

### Korpus durumu
- `envs/` — 9 elle yazilmis smoke/seed gorevi.
- `envs_gen/` — **154** uretilmis gorev (`gen-s-XXXX-<hedef>`), her biri tek dizin + `task.yaml`.
- Gorev alanlari: `ozetle, donustur, ayikla, denetle, bol, birlestir, karsilastir, onar,
  grupla-say, yeniden-duzenle, arsivle, dogrula, sirala-sec, toplu-yeniden-adlandir`.

### Son sweep sonucu (`data/sweep2.log`, `data/sweep2.jsonl`, 154 gorev)

| Metrik | Deger |
|---|---|
| Cozulen | 36 / 154 |
| Ortalama reward (ikili) | 0.393 |
| Ortalama kismi (teshis) | 0.408 |
| Ortalama tur | 5.1 |
| **BANT (egitimde ise yarar)** | **44 / 154 (%29)** |
| OLU-kolay | 36 |
| OLU-zor | 74 |

Ana darbogaz: 74 gorev hic cozulemiyor (OLU-zor). Egitim sinyali tasiyan korpus
su an **44 prompt**.

### Bilinen acik konular
- `data/*.log` dosyalari UTF-16 yazilmis; Turkce karakterler bozuk gorunuyor.
- Bazi `reference_solution` alanlari LLM'in "dusunerek" yazdigi coklu deneme
  metinleri iceriyor (orn. `gen-s-0043-dogrula`) — calisiyor ama kirli.
- `.mcp.json` takip edilmiyor; `data/sweep2.log` commit edilmemis.

### Sonraki hedef
Bu veri setinin **ise yarayip yaramadigini** olcmek: Colab uzerinde GRPO ile
4-5 epoch egitim, her epoch sonunda sabit 30 soruluk SWE-Bench alt kumesiyle
dogrulama. Plan Bolum 1'de.

---

## Bolum 1 — GRPO egitim plani (2026-09-09, mutabik kalindi)

### Amac
Veri setinin **ise yarayip yaramadigini** olcmek. Hipotez: bu Turkce dogrulanabilir
terminal korpusunda GRPO ile egitim, modelin ajanik arac-kullanimi becerisini
olcelebilir sekilde artirir.

### Mutabik kalinan kararlar

| Konu | Karar |
|---|---|
| Platform | **Colab** (A100 80GB). Lokal makine devrede degil. |
| Sandbox | Colab'de Docker yok → **`LocalSandbox`** (tempdir + subprocess). `docker exec` semantigi birebir korunur. |
| Taban model | **Qwen3.5-4B** (tam repo id Faz 0'da teyit edilecek) |
| Egitim | **Full fine-tune** + `adamw_bnb_8bit` + gradient checkpointing (~40GB, vLLM'e ~35GB kalir) |
| Thinking modu | **Kapali** — cok turlu dongude baglam ve sure patlamasin, maskeleme temiz kalsin |
| Cerceve | **TRL GRPOTrainer** + kendi cok-turlu rollout kopru kodumuz |
| Uretim | **vLLM**, egitimle ayni kartta (colocate) |
| Odul | **Ikili** (solved=1, aksi 0). `partial` yalnizca teshis. Sekillendirme/format odulu **yok**. |
| Grup | **G = 8**, egitim rollout'larinda yuksek sicaklik; eval'de `temperature=0`, tek rollout |
| Epoch | **4-5**, ayrica **epoch 0 baseline** |
| Kontrol grubu | Yok — epoch'lar kendi aralarinda karsilastirilacak |
| Basari esigi | Held-out cozum oraninda epoch 0'a gore **+%15 mutlak**, son iki epoch'ta korunuyor |
| Repo | **public**, Colab'e `git clone` ile |
| Kayit | Drive'a checkpoint + `data/train/epochNN.jsonl`; **W&B**. Tum log yazimi **UTF-8**. |

### Faz 0 — Dogrulama (kod yazmadan once)
- **0.1** Colab'de A100 80GB tahsisini `nvidia-smi` ile teyit
- **0.2** **sb-cli** (swebench.com barindirilan degerlendirme) erisimini test et.
  Calisiyorsa Colab'de SWE-bench icin **hic Docker gerekmez**: ajan `git clone` +
  `git checkout <base_commit>` yapip **patch uretir**, patch'ler uzaga gonderilir.
  Kapaliysa **yedek plan**: Colab patch uretir, degerlendirme Docker'li lokal makinede kosar.
- **0.3** TRL surumunde custom-rollout API'sinin (`rollout_func` vb.) varligini teyit
- **0.4** `Qwen3.5-4B` icin dogru HF repo id, context uzunlugu, tool-calling sablonu

Ucu de yesil degilse plan revize edilir.

### Faz 1 — `LocalSandbox`
- `sandbox.py`'a yeni sinif: episode basina `mkdtemp()`, `subprocess.run(..., cwd=root, shell=True)`
- **`docker exec` semantigi bilerek korunur**: her tur taze shell, `cd` ve env turlar arasi tasinmaz.
  Yoksa mevcut bant olcumleri gecersizlesir.
- `snapshot()` / timeout / cikti kirpma arayuzu birebir ayni → `task.py` ve `grade()` degismez
- `--sandbox {docker,local}` bayragi
- **Guvenlik (izolasyon olmadigi icin):** tehlikeli komut deseni reddi, calisma dizini
  disina yazma engeli, per-komut timeout
- **Kabul testi:** `envs/` + `envs_gen/` uzerinde `--reference` sweep'i Docker'li kosuyla
  **ayni bant dagilimini** vermeli

### Faz 2 — Bant genisletme
- Jeneratoru tekrar kostur: **+150 gorev**, **concurrency ≤ 8**, mevcut `.env` endpoint'i
- Hedef: BANT ~200 gorev
- **Held-out ayrimi simdi dondurulur**: genisletilmis korpus `domain` bazinda tabakali
  **%70 train / %30 held-out**, dosyaya yazilip sabitlenir. Held-out egitime **hic** girmez.
- **Dinamik mufredat:** ayri sweep kosmaya gerek yok — egitimde zaten her prompt'un G=8
  rollout'u var, pass-rate oradan gelir. Her epoch sonunda `0/8` ve `8/8` prompt'lar bir
  sonraki epoch'tan dusurulur; OLU-zor havuzu ara sira yoklanip BANT'a girenler geri alinir.

### Faz 3 — Cok turlu GRPO koprusu (en agir parca)
TRL'in GRPOTrainer'i tek turluk uretir; ortam cok turlu. Cozum **tur-senkron toplu rollout**:

```
her adim:
  B prompt x G=8 = N episode baslat (N adet LocalSandbox)
  tur t = 1..max_turns:
      aktif episode'larin mesaj gecmislerini tek batch'te vLLM'e ver
      yanitlardan komutu ayikla (native tool-call veya <komut>)
      komutlari thread pool ile paralel calistir (16 worker)
      gozlemleri gecmise ekle; komut yoksa episode kapanir
  her episode grade edilir -> ikili odul
  GRPO: grup ici avantaj = (r - grup_ort) / grup_std
  loss YALNIZCA assistant token'lari uzerinde (gozlem token'lari maskeli)
```

Kritik detaylar:
- **Gozlem token'lari loss'tan maskelenmeli** — aksi halde model kendi uretmedigi metni ogrenir
- Bir gruptaki G rollout ayni prompt'tan gelmeli
- Grup std = 0 ise (hepsi 0 veya hepsi 1) o prompt o adimda avantaj uretmez, atlanir

**Hiz hedefi:** ~200 prompt x 8 = 1600 episode, ~5 tur → 5 toplu uretim dalgasi.
Dalga ~3 dk + ortam ~2 dk + egitim ~20 dk ≈ **epoch basina 45-60 dk**.
5 epoch + eval tek Colab oturumuna sigar.

### Faz 4 — Degerlendirme
Her epoch sonunda **ve epoch 0'da**:
1. **Held-out cozum orani** — ANA METRIK. `temperature=0`, tek rollout, ~60 gorev.
2. **SWE-bench Verified 30** — zorluk etiketine gore **10 kolay / 10 orta / 10 zor**,
   sabit seed, her epoch **ayni liste**. Scaffold: **kendi `runner.py`'imiz** (egitilen
   protokolun aynisi olsun ki transfer olcumu temiz kalsin). Patch uret → sb-cli → resolved orani.
3. Yardimci: ortalama tur sayisi, gecersiz arac cagrisi orani, bant dagilimi kaymasi.

### Faz 5 — Operasyon
- `notebooks/train_grpo.ipynb` — Colab'de, Colab MCP uzerinden yonetilir
- Drive mount → her epoch sonu checkpoint + optimizer state + episode kayitlari; kopma sonrasi resume
- W&B: reward, cozum orani, KL, grad norm, bant dagilimi, held-out, SWE-bench

### Bilinen riskler
| Risk | Etki | Azaltma |
|---|---|---|
| sb-cli kapali | SWE-bench olcumu Colab'de kosmaz | Yedek: patch Colab'de, degerlendirme lokal Docker'da |
| LocalSandbox'ta izolasyon yok | Model komutu VM'e zarar verebilir | Desen reddi + dizin siniri + timeout; VM zaten atilabilir |
| LocalSandbox bant dagilimini kaydirir | Tum onceki olcumler gecersizlesir | Faz 1 kabul testi bunu yakalar |
| 44 → 200 prompt hala kucuk | Zayif sinyal, asiri uyum | Dinamik mufredat + held-out ile izleme |
| Transfer yok (Turkce shell → Python repo) | SWE-bench oynamaz | Ana metrik held-out; SWE-bench ikincil metrik |
| Colab oturumu duser | Egitim kaybi | Drive checkpoint + resume |

---

## Bolum 2 — Uretim hattinin toparlanmasi (2026-09-09)

### Saglayici degisikligi: t3ai -> DeepSeek
Eski uc (`t3ai.baykartech.net/Jailbreak/v1`) HTTP 401 vermeye basladi
("invalid api key for app Jailbreak"). Endpoint on-kontrolu bunu kosu
baslamadan yakaladi, para harcanmadi.

Yeni uc: **DeepSeek**, model **`deepseek-v4-flash`**.
- `llm.py` yalnizca `OPENAI_API_KEY` ve `MISTRAL_API_KEY` taniyordu; uca gore
  anahtar secen yapiya **DeepSeek eklendi** (`DEEPSEEK_API_KEY`, `DEEPSEEK_API`).
  Saglayicilar artik tek bir tabloda (`SAGLAYICILAR`), mevcut davranis bozulmadi.
- `.env`'de eski t3ai degerleri **yorum satirinda korundu**, yedegi `.env.bak`.

### `pipeline.py` — uctan uca metascript
Asamalar (`generate`, `prompts`, `gates`, `sweep`) ayri komutlardi ve her biri
uc/anahtar/model'i yeniden cozumluyordu; biri farkli bir uca giderse korpus
sessizce karisiyordu. Yeni modul dordunu **tek istemci ve tek uc** uzerinden
kosuyor:

| Asama | Is | API |
|---|---|---|
| `uret` | seed -> spec (setup, referans cozum, turetilen check'ler) + spec-only kapilar | evet |
| `prompt` | Turkce gorev metni (referans cozum gosterilmeden) | evet |
| `kapi` | **tam gauntlet** (prompt kapilari dahil); dusenler karantinaya | hayir |
| `bant` | gorev basina N rollout -> BANT / OLU-kolay / OLU-zor | evet |

- **Yeniden calistirilabilir**: her asama isini bitirmis task'lari atlar.
- **Karantina** (`envs_gen/_red`): kapilarda dusen silinmez, kenara alinir.
- **Token sayaci**: asama basina istek/giris/cikis; `--fiyat-*` ile dolar tahmini.
  Butce gercek bir kisit ve panel geriye donuk kirilim vermiyordu.

### Kritik bulgu: `deepseek-v4-flash` bir reasoning modeli
Prompt asamasinda 9 task'in 3'u yazilamadi. Belirti yaniltiyordu: denetleyici
"prompt cok kisa + hic Turkce karakter yok + cikti dosyasi anilmiyor" diyordu.
Gercek sebep: model `max_tokens`'in **tamamini dusunmeye** harciyor, `content`
bos donuyordu (`finish_reason=length`, `reasoning_tokens=1024/1024`).

`max_tokens` buyutmek cozmedi — 4096 verildiginde onu da tamamen dusunmeye
harcadi. Cozum **`reasoning_effort="none"`**.

Duzeltme `llm.py`'da **tek yerde**, cunku sorun her asamayi etkiliyordu
(`runner.py` da `max_tokens=1024` kullaniyor, yani bant asamasi da cokecekti):
- `_Completions` her istege ortak alan enjekte ediyor
- Ucun **tanimadigi alan sessizce dusurulup** istek tekrarlaniyor (yerel vLLM
  `reasoning_effort` bilmeyebilir; kosu comesin diye)
- `pipeline.py`'da `--reasoning-effort`, varsayilan `none`

### Dusunme kapali/acik olcumu (n=10, `uret` asamasi)

| | dusunme acik | dusunme kapali |
|---|---|---|
| Kabul | 9/10 | 9/10 |
| Ortalama deneme | 2.2 | **1.9** |
| 10 aday maliyeti | $0.42 (panelden) | **$0.016** (tahmin) |

Karar: **dusunme kapali**. Kabul orani dusmuyor, onarim turu azaliyor.
(n=10 kucuk bir orneklem; karar "kotu degil + cok ucuz" gerekcesiyle alindi.)

Asama bazinda: `prompt` kapali, `uret` kapali, `bant` **bu uctan hic
kosulmayacak** -- bant, egitilecek modelle olculmeli, DeepSeek'in cozebildigi
gorev kumesi Qwen3.5-4B'ninkiyle ayni degil. Bant Colab'de olculecek.

### Butce -- onceki tahmin gecersiz
Onceki "150 aday $6.3, BANT 200 hedefi ~$30" tahmini **tamamen dusunme
token'iydi**. Olculen: 10 aday = 20 istek, 27.6k giris, 7.9k cikis.
Buradan 400 aday ~$0.65, prompt asamasi ~$0.24, uctan uca **~$1**.
Yatirilan $2 yeterli.

### Korpus durumu
**163 task**, prompt'u bos olan yok, yeni 9'un dokuzu da tam gauntlet'ten gecti.
En cok tetiklenen kapi: `arac uygunlugu` (model seed'in verdigi araci
kullanmadan cozuyor, onarim turunda duzeliyor).

---

## Bolum 3 — Kod ailesi, Docker'siz sandbox, dondurulmus split (2026-09-09)

### Kod ailesi -- SWE-bench'e yaklasma
Korpus yalnizca Turkce kabuk gorevlerinden olusuyordu; olcmek istedigimiz
ajanik kodlamaya uzakti (Bolum 1'deki B1 riski). `seeds.py`'a ikinci bir
seed ailesi eklendi (`aile: "kod"`, varsayilan `"kabuk"` oldugu icin eski
task.yaml'lar aynen yukleniyor):

| Hedef | Pay | Ajanin isi |
|---|---|---|
| `hata-bul` | %40 | 2-3 modulun birinde SESSIZ hatayi bul ve duzelt (hep cok dosyali) |
| `veri-yapisi` | %30 | Iskeleti (her metot `raise NotImplementedError`) gerceklestir |
| `yarisma` | %30 | Hedef .py'yi sifirdan yaz (SETUP onu hic olusturmaz) |

Cesitlilik eksenleri: **11 hata tipi**, 10 algoritmik kalip, 10 veri yapisi,
zorluk (kolay/orta/zor). Talimat "bu yonde, ayrintiyi sen sec" diyor --
cesitliligin bir kismi bilincli olarak uretici modele birakildi.
`hata-bul` her zaman, digerleri %70 cok dosyali.

**Dogrulama**: `kind: program` + `run:`. Test **task.yaml icinde satir ici**
tasiniyor, calisma dizininden okunmuyor -- ajan dosya sistemini degistirerek
odulunu yukseltemiyor.

**Kapi degisikligi**: `arac uygunlugu` kod ailesinde atlaniyor. Referans
cozum `python3`'u komut konumunda cagirmiyor (heredoc ile dosya yaziyor),
kural oldugu gibi birakilsa her aday haksiz yere duserdi.

**derive.py**: `run` hicbir sey basmiyorsa net uyari. Bu gercek bir acikti --
bos stdout her dunyada esit oldugu icin check bozuk dunya ile dogruyu ayirt
edemiyordu.

### Kabul orani ve talimat turlari
1/6 -> 2/6 -> 3/8 -> **81/200 (%40)**. Yukselten sey soyut ilke degil
mekanik tarif oldu (ornegin "SETUP hedef dosyayi HIC olusturmaz").
Hedef bazinda: `veri-yapisi` %57, `hata-bul` %40, `yarisma` %31.
Kabul edilen gorev basina ~$0.009 (kabuk ailesinde ~$0.002).

### Karantinadan cikan iki kusur (kod okunarak bulundu, API'ye para vermeden)
1. **Mutlak `/workspace` yolu** -- Docker'da calisir, LocalSandbox'ta calismaz.
   Korpusta **18 gorev** (8 elle yazilmis dahil) goreli yola cevrildi; 19
   gorev iki modda da ayni sonucu veriyor.
2. **Test verisi calisma dizinindeki bir `.py`'dan okunuyor**
   (`from test_data import CART`) -- ajan o dosyayi degistirip odulu
   kandirabilir. Korpus tarandi: **1 gorev** (`gen-k-0139`) karantinaya alindi.

Ikisi icin de talimata kural eklendi (kural 12 ve 13).

### `LocalSandbox` -- Colab'de Docker yok
`sandbox.py`'a DockerSandbox ile **birebir ayni arayuze** sahip yerel
sandbox. Dogrulanan semantik: `cd` ve `export` turlar arasi tasinmiyor
(docker exec ile ayni), zaman asimi 124, tehlikeli komut reddi 126,
`snapshot()` calisiyor.

**Izolasyon yok** -- bu kodun basina acikca yazildi. Ucuz korumalar:
yikici komut deseni reddi, per-komut timeout, HOME/TMPDIR kok dizinde.

`checks.py`: `run_stdout_eq` imaj adi bosken host'ta kosuyor. Bu sartti --
kod ailesinin TUM dogrulamasi bu check'ten geciyor.

`task.py`: `make_sandbox()` moda gore seciyor; butun cagri noktalari
oradan gectigi icin mod tek yerden ayarlaniyor (`sandbox.yerel_kullan`).
`--sandbox {docker,yerel}` bayragi sweep, gates ve pipeline'a eklendi.
`uret`/`kapi` asamalari yerel modda reddediliyor (kapilar konteyner aciyor).

`pipeline.py` bant asamasi **paralellestirildi**: 80 gorev x 8 rollout x
~5 tur = 3200 ardisik istek sunucuyu bos birakiyordu.

### Referans sagligi (Docker, modelsiz)
- kabuk: **171/175** (dusenler: gen-s-0001-donustur, gen-s-0059-arsivle,
  gen-s-0063-bol, gen-s-0085-donustur)
- kod: **80/80**

Yerel modda (Windows/Git Bash) 154/175 -- aradaki 17 fark Git Bash'in GNU
arac farklari. **Yerel/Docker esdegerligi Linux'ta yeniden olculmeli**;
Windows'taki karsilastirma yaniltici.

### Dondurulmus split -- `data/split.json`
| | Gorev |
|---|---|
| train | **173** (kabuk 115, kod 58) |
| held-out | **78** (kabuk 53, kod 25) |
| dislanan | 4 (referansi gecmiyor) |

`(aile, hedef)` tabakalamasi: her tabakada held-out payi %27-40, toplam %31.
Duz rastgele bolme held-out'a orantisiz sayida `yarisma` dusurebilirdi.
Dosya `tohum=1234` ile muhurlu, `--force` olmadan uzerine yazilmiyor --
held-out ancak ayni kume kaldigi surece epoch'lar arasi anlam tasir.

### Colab durumu (Faz 0)
| | |
|---|---|
| GPU | **A100-SXM4-80GB** teyit edildi |
| Makine | 12 CPU, 167 GB RAM, 189 GB disk, Linux 6.6 |
| Docker | yok (beklendigi gibi) |
| Paketler | vllm 0.29.0, trl 1.12.0, torch 2.13.0+cu130 |
| Model | **`Qwen/Qwen3.5-4B` gercekten var** (9.3 GB, `-Base` ayri repo) |
| LocalSandbox | Colab'de calisiyor |

**Uretim Colab'e tasinmiyor** -- Docker'li lokal makinede kaliyor. Colab
yalnizca egitim, bant olcumu ve degerlendirme icin.

---

## Bolum 4 — Colab'de ilk olcumler ve GRPO kurulumu (2026-09-09)

### Ortam kurma sirasinda cikan gercek engeller

| Engel | Teshis | Cozum |
|---|---|---|
| vLLM hic ayaga kalkmadi | `pip install vllm` torch'u cu130'a cikardi, Colab'in torchaudio'su cu128 kaldi | torchaudio kaldirildi (vLLM metin modelleri icin ona ihtiyac duymuyor) |
| Arac cagrisi ayristirilamadi | `--tool-call-parser hermes` Qwen3.5'in `<function=...><parameter=...>` bicimini tanimiyor | `--protocol text` (`<komut>` etiketleri) -- ayristiriciya hic ihtiyac yok |
| `reasoning_effort="none"` | DeepSeek'te dusunmeyi kapatiyordu; **Qwen3.5'te ciktiyi bozuyor** ("12+17" -> 55, 3 token) | Bu modelde kullanilmiyor; dogru anahtar `chat_template_kwargs={"enable_thinking": False}` |

### Dusunme acik/kapali karari
Kullanici karari: **dusunme ACIK**. Gerekce: kod gorevlerinde muhakeme
gercekten yardimci oluyor ve en buyuk sorunumuz OLU-zor havuzuydu.
Maliyeti olculdu: istek basina ~530 cikis token, epoch suresi yaklasik
iki katina cikiyor.

Dusunce `reasoning_content`'e degil dogrudan `content`'e akiyor.

### Baglam siniri -- olcumu bozan hata
Ilk bant kosusunda `--max-model-len 8192` ile **640 episode'un 139'u (%22)**
baglam tasmasindan hata aldi ve cozulmemis sayildi; olcum OLU-zor'a dogru
yanliydi. 32k'ya cikarilinca:

| | 8k (yanli) | **32k** |
|---|---|---|
| BANT | 41 | **47 / 80 (%59)** |
| OLU-kolay | 7 | 6 |
| OLU-zor | 32 | **27** |
| Hatali gorev | 64 | **1** |
| Ortalama pass_rate | 0.333 | 0.392 |
| Ortalama tur | 5.99 | 6.50 |
| Sure | 16.7 dk | 21.5 dk |

### KOD KORPUSU BANT OLCUMU (Qwen3.5-4B, dusunme acik, G=8)

| Hedef | BANT | OLU-kolay | OLU-zor | BANT% |
|---|---|---|---|---|
| `hata-bul` | 24 | 2 | 4 | **%80** |
| `veri-yapisi` | 17 | 2 | 12 | %55 |
| `yarisma` | 6 | 2 | 11 | %32 |
| **Toplam** | **47** | **6** | **27** | **%59** |

Kabuk korpusunda (DeepSeek ile olculmustu) bu oran %29'du. **Kod ailesi
egitim sinyali acisindan belirgin sekilde daha iyi** -- ve en iyi alt grup
(`hata-bul`, %80) ayni zamanda SWE-bench'e en yakin olani.

Throughput: 640 episode / 21.5 dk, GPU %85. Epoch tahmini artik tahmin degil.

### TRL kurtarma operasyonu
TRL 1.12 + vLLM 0.29 **import bile edilemiyor** (`NCCLTrainerSendWeightsArgs`).
vLLM 0.27.1'e dusuruldu; iki kontrol de gecti: TRL import ediyor ve
`Qwen3_5ForConditionalGeneration` destekleniyor.

**Buyuk kazanc**: TRL 1.12'de `environment_factory` var. Cok turlu arac
dongusunu, **arac ciktisi token'larinin loss'tan maskelenmesini**
(`tool_mask`, Faz 3'te sart kostugumuz sey) ve vLLM colocate agirlik
senkronunu TRL yurutuyor. Planlanan koprü kodu uc metoda indi:
`reset` / `run_command` / `get_reward`.

Ortam dogrulandi: hicbir sey yapmadan odul 0.0, referans cozumden sonra 1.0.

### Bellek: 4B full FT + colocate vLLM 80 GB'a zar zor sigiyor

| Deneme | Sonuc |
|---|---|
| `vllm_gpu_memory_utilization=0.30` | vLLM'e KV cache icin yer kalmadi |
| `=0.85` | baslangicta o kadar bos bellek yok (61.84 < 67.36 GiB) |
| mikro-batch 8 | `lm_head` OOM: 8 episode x 2048 token x 152k kelime = 7.58 GiB logit tensoru |
| **uyku modu + mikro-batch 2 + 0.45** | **calisti** (GPU %100, 52 GB) |

Belirleyici olan `vllm_enable_sleep_mode`: vLLM uretim disinda bellegi
biraktigi icin egiticinin (~40 GB) ve vLLM'in zirveleri ust uste binmiyor.

### Colab runtime kaybi -- alinan ders
Egitim ilk adimlarindayken **runtime yeniden atandi**: repo, 9.3 GB model
cache, kurulumlar ve bant kayitlari silindi. `data/*` gitignore'da oldugu
icin bant olcumu hic push edilmemisti.

Alinan onlemler:
- `data/bant-*.jsonl` gitignore'dan muaf tutuldu (olcum pahali, deneyin kaydi)
- Bir sonraki kurulumda **once Drive baglanacak**, cikti ve checkpoint oraya yazilacak

Colab oturumu kalici degil; bunu varsayan her sey (kurulum, cache, cikti)
yeniden uretilebilir ya da Drive'da olmali.

---

## Bolum 5 — GRPO egitimi basladi (2026-09-09)

### Kurulum (runtime kaybindan sonra yeniden)
Colab runtime'i yeniden atandi ve her sey silindi. Ikinci kurulumda
**once Drive baglandi**; egitim ciktisi ve log'u
`MyDrive/turkish-agentic-rl/` altina yaziliyor.

Calisan bilesim: `vllm 0.27.1` + `trl 1.12.0` + `torch 2.13.0+cu130` +
`transformers 5.16.1`, `torchaudio` kaldirilmis.

### Egitim yapilandirmasi
| | |
|---|---|
| Model | Qwen/Qwen3.5-4B, full fine-tune, `adamw_bnb_8bit` |
| Korpus | **173 gorev** (train split: 115 kabuk + 58 kod) |
| G (rollout/grup) | 8 |
| Iterasyon basina prompt | 4 → 32 episode |
| Mikro batch | 2 (logit tensoru yuzunden), biriktirme 16 |
| Odul | ikili |
| Dusunme | acik |
| KL (`beta`) | 0 |
| lr | 1e-6 |
| vLLM | colocate + **uyku modu**, bellek 0.45, baglam 8192 |

Kabuk gorevlerinin bandi olculmemis olmasina ragmen egitime dahil edildi:
GRPO kendini duzeltiyor -- bir grubun butun rollout'lari ayni sonucu
verirse avantaj sifir olur ve o prompt gradyan uretmez.

### Ilk 5 adim

| Adim | reward | zero_std | clipped | mean_len | call_freq | entropy | sure |
|---|---|---|---|---|---|---|---|
| 1 | 0.344 | 0.25 | 0.59 | 1379 | 7.13 | 0.569 | 205 |
| 2 | 0.094 | 0.75 | 0.25 | 1152 | 7.31 | 0.621 | 302 |
| 3 | 0.125 | 0.50 | 0.41 | 1238 | 5.88 | 0.706 | 221 |
| 4 | 0.719 | 0.75 | 0.19 | 1054 | 5.63 | 0.530 | 227 |
| 5 | 0.063 | 0.75 | 0.44 | 1234 | 5.75 | 0.732 | 219 |

**Sistem calisiyor**: odul sifir degil, grup ici varyans var, arac cagrisi
sikligi ~6-7 (gercekten cok turlu), arac hata orani %1.8.

**Odul egrisinden ogrenme okunamaz.** Adim basina yalnizca 4 prompt
cekiliyor (173 icinden), yani odul buyuk olcude "hangi dortlu geldi"
sorusunun cevabi. 0.34 / 0.09 / 0.13 / 0.72 / 0.06 salinimi bunun sonucu.
Anlamli bir egim icin en az 8-10 adim gerekiyor; asil olcut held-out.

Uc adimlik entropi tirmanisi (0.569 → 0.706) 4. adimda 0.530'a dondu --
erken yorumdu, `beta` onerisi geri cekildi.

Gercek trend gosteren tek metrik **kesilme orani**: 0.59 → 0.25 → 0.41 →
0.19 ve ortalama uzunluk 1379 → 1054. Model daha derli toplu cozumler
uretiyor olabilir.

### Sure sorunu (cozulmedi)
Adim ~235 sn, epoch 43 adim → **~2.9 saat/epoch**. 5 epoch 14.5 saat,
Colab'in ~12 saatlik oturumuna sigmiyor. Secenekler: `--max-komut` 12→8,
`--rollouts` 8→6, epoch sayisini dusurmek, ya da dusunmeyi kapatmak.
Karar ertelendi; once trendi gormek gerekiyor.

### Checkpoint riski
`save_steps=20`, yani ilk checkpoint 20. adimda (~80 dk). O ana kadar
kopma olursa is bastan baslar. Adim basina kaydetmek pratik degil:
4B checkpoint ~8 GB ve Drive'a yazmak 8-13 dk suruyor.

---

## Bolum 6 — Epoch 1 sonucu: veri seti ogretiyor, ama neyi ogrettigi korpusa bagli (2026-09-10)

### Held-out olcumu (78 gorev, sicaklik 0, tek rollout, ayni protokol)

| Held-out | epoch 0 | epoch 1 | Delta |
|---|---|---|---|
| **Kod** (24) | 9 (%37.5) | 4 (**%16.7**) | **-20.8** |
| **Kabuk** (54) | 5 (%9.3) | 11 (**%20.4**) | **+11.1** |
| **Toplam** (78) | 14 (%17.9) | 15 (%19.2) | +1.3 |

Ayrinti:

| Kume | Cozulen | Ort. tur | Ort. kismi |
|---|---|---|---|
| epoch0 kabuk | 5/54 | 6.37 | 0.099 |
| epoch1 kabuk | 11/54 | 6.11 | **0.237** |
| epoch0 kod | 9/24 | 6.71 | 0.375 |
| epoch1 kod | 4/24 | **5.67** | **0.167** |

Gorev bazinda: kod'da 8 kayip / 3 kazanc, kabuk'ta 8 kazanc / 2 kayip.

### Okuma
- **Kabuk'taki iyilesme gercek**: ikili odul iki katindan fazla artti ve
  kismi puan da 0.099 -> 0.237. Kismi puan daha az kesikli oldugu icin bu
  daha guclu kanit -- model onceden hicbir check'i gecemedigi gorevlerde
  artik bir kismini geciyor.
- **Kod'daki dusus de gercek**: ikili ve kismi ayni yonde, tur sayisi
  6.71 -> 5.67 (%16 dusus). Kabukta tur neredeyse degismedi.
- Iki aciklama var, ayirmak zor: (a) model daha kestirme davranmayi
  ogrendi -- kabukta ise yariyor, kod'da "pes etmek" demek; (b) egitim
  korpusunun **ucte ikisi kabuk**ti, politika o tarafa kaydi.

### Plandaki esik
"+%15 mutlak" hedefi konmustu. Toplamda **+%1.3** (14 -> 15 gorev).
Esik yakalanmadi; tek epoch icin zaten beklenmiyordu. Asil bilgi
toplamda degil **ayrismada**.

### Alinan ders: olcmeden egitime katmak
"173 gorevin hepsiyle egitelim, kabuk bandini olcmedik ama GRPO kendini
duzeltir" karari **yanlis cikti**. GRPO gercekten kendini duzeltti
(kabuk gorevleri egitimi bozmadi) ama cogunluk olduklari icin politikayi
kendi tarzlarina cektiler ve onemsedigimiz kod tarafina zarar verdiler.

Baseline ayrica kabuk korpusunun bu model icin ne kadar zor oldugunu
gosterdi: **%9.3**. 54 gorevin 49'u cozulemiyor, yani cogunda 8 rollout'un
8'i de basarisiz -> grup ici varyans sifir -> gradyan yok. Yuksek
`frac_reward_zero_std`'nin kaynagi buydu.

### KESILME: ortuk uzunluk cezasi (olculdu)
`max_completion_length = 2048` ile episode'larin **%40'i kesiliyordu**.
42 adimlik veriden:

```
korelasyon(odul, kesilme_orani)    = -0.507
korelasyon(odul, ortalama_uzunluk) = -0.523
korelasyon(kesilme, uzunluk)       = +0.753
```

Uzun -> kesiliyor -> odul 0 -> negatif avantaj. Sinir bir butce degil,
**ortuk bir uzunluk cezasi** haline gelmis. RLVR'de bilinen bir patoloji
(DAPO kesilen rollout'lari ayri ele aliyor).

Duzeltme: `--max-completion` varsayilani 2048 -> **6144**,
`--vllm-baglam` 8192 -> **16384**, ve baglam completion'a gore darsa
uyari basiliyor. Bir episode'un TAMAMI (butun turlar + arac ciktilari)
completion butcesine sayiliyor.

Not: "egitim boyunca uzunluk 1379 -> 1100 dustu" gozlemi **yanlisti** --
uc noktalar secilmisti. Gercek: ilk 10 adim 1221, son 10 adim 1201.
Egitim sirasinda sistematik kisalma yok; kisalma degerlendirmede goruldu.

### Epoch 2 icin oneri
Sadece **kod** korpusu ile egitmek -- bu sabahki karara gore tam ters,
ama artik olcum var. Kabuk gorevlerinin %91'i cozulemiyor, yani compute'un
ucte ikisi gradyan uretmeyen gruplara gidiyor ve uretebildigi kadari da
politikayi istemedigimiz yone cekiyor.

---

## Bolum 7 — Dogru baseline, ve uzunluk duzeltmesinin bedeli: OOM (2026-09-10)

### Dongu adim 1'de oldu

`dongu.log` 23:30'da durdu. `train-epoch1.log` sonu:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.68 GiB.
79.25 GiB kapasitede 1.29 GiB bos
4%|▍ | 1/25 [13:56<5:34:34]
```

Ilk optimizer adiminda, `_compute_loss` -> `selective_log_softmax` icinde.
Sebep dogrudan Bolum 6'daki duzeltme: `max_completion` 2048 -> 6144 ve
`vllm-baglam` 8192 -> 16384. Logits tensoru `batch x 6144 x vocab` oldu;
4B full FT + colocate vLLM zaten 80 GB'a zar zor siğiyordu (Bolum 4).
**Ortuk uzunluk cezasini kaldirdik, yerine bellek tavani geldi.**
Checkpoint yok; diskte sadece `mufredat/epoch1.txt` ve sweep olcumleri.

### Ilk GUVENILIR baseline (held-out 78, G=8 pass-rate)

Sicaklik ve split hatalari duzeldikten sonraki ilk olcum:

| Aile | pass@8 | En az bir kez cozulen |
|---|---|---|
| Kod (24) | **%32.3** | 17/24 |
| Kabuk (54) | **%18.3** | 25/54 |
| **Toplam (78)** | **%22.6** | 42/78 |

**Bolum 6'nin "sadece kod ile egit" onerisi zayifladi.** O oneri kabuk
icin "%9.3, 54 gorevin 49'u cozulemiyor" sayisina dayaniyordu. O sayi tek
rollout'luk ve sicakligi bozuk olcumdendi. G=8 ile kabugun 54'unun **25'i**
en az bir kez cozuluyor -- yani kabuk korpusu olu degil, sadece varyansi
yuksek. Kod ile kabuk arasindaki gercek fark 32.3 vs 18.3, ucuruma benzemiyor.

### Epoch 1 mufredati (173 train gorevi, G=8)

- Bantta (1..7): **103**
- Olu-zor (0/8): **65**
- Olu-kolay (8/8): **5**

173'un %60'i egitici. Bolum 6'daki "compute'un ucte ikisi gradyan
uretmeyen gruplara gidiyor" endisesi de bu yuzden abartiliydi: bant
olcumu zaten olu gorevleri disariya atiyor.

### Duzeltme: yalnizca (a), tek kol

Karar: once en ucuz mudahale, olmazsa yeniden degerlendir.

**Bulgu:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` o kosuda
**zaten aciktı** (`dongu.py:136` -> `ortam` -> `kos(env=ortam)`). OOM
mesajindaki "104.55 GiB allocated" 80 GB'lik kartta zaten expandable
segments muhasebesi. Yani (a)'nin o yarisi harcanmis durumda.

Geriye tek gercek kol kaldi: `--vllm-bellek`, `dongu.py`'de **0.45 olarak
sabit yazilmisti**. Bayrak yapildi, varsayilan **0.30** (~12 GB serbest).

### Kural uygulandi: koşmadan once test

Uc olcum hatasi da "eklendi ama calismiyor"du. Bu sefer once test:
`kos` ve `vllm_baslat` stub'lanip egitim komutu yakalandi.

```
vllm-bellek    -> --vllm-bellek 0.3   (komut satirinda)
max-completion -> 6144
vllm-baglam    -> 16384
ALLOC_CONF alt-surece gecti -> expandable_segments:True
```

`train_grpo.py:266` `args.vllm_bellek`'i dogrudan `GRPOConfig`'e veriyor --
sicaklik hatasindaki gibi import-aninda yakalama yok.

### Olmazsa siradakiler (henuz yapilmadi)

- (c) `max-completion` 6144 -> 4096; ama once 4096'da kac episode'un
  kesildigi olculmeli, korten once.
- (b) `prompt-batch` 4 -> 1. IsoCompute'a gore batch'teki problem sayisi
  kararliligi belirliyor, yani en son basvurulacak kol.

---

## Bolum 8 — OOM avi: sebep fp32 cikti; ve bant olcumu egitimle ayni olcekte degil (2026-09-10)

Bolum 7'deki duzeltme yetmedi; iki kosu daha yandi. Her adimda neyin
OLCULDUGUNE bakmak gerekiyor, cunku iki kez yanlis kola asildi.

### Kosu 2: yanlis kol oldugu olculdu

`--vllm-bellek` 0.45 -> 0.30 yapildi. Kosu 2 adim 1'i gecti, **adim 2'de
ayni yerde ayni sayiyla** oldu: `selective_log_softmax` -> `logsumexp`,
**5.68 GiB**. PyTorch tahsisi 104.55 -> 92.66 GiB dusmustu; 0.30 gercekten
~12 GB acti ama tepe daha fazlasini istiyordu.

8 gorevluk kisa bir kosuda bellek 2 sn'de bir orneklendi. Faz gecisi:

```
06:37:30   40.0 GB            uretim (vLLM uyanik)
06:37:33   26.3 GB  (-13.7)   sleep -> KV cache birakildi
06:37:35   22.0 GB  ( -4.3)   sleep level 2 -> agirliklar da birakildi
06:37:37   49.5 GB  (+27.4)   loss fazi
06:38:41   75.7 GB            <- TEPE, duvar 79.25
```

- **OOM egitimde, uretimde degil.** Uretim 40 GB'de dumduz.
- **Uyku modu calisiyor** (`vllm_generation.py`, kosulsuz: init'te sleep(2),
  sync_weights'te wake_up(weights), generate oncesi wake_up(kv_cache),
  sonrasi sleep(2)). vLLM loss tepesinde **18 GB birakiyor** -- yani onu
  kismakla kazanilacak sey yoktu. Kismamizin sebebi kanit degildi:
  "sadece (a)" denince (a)'nin icinde kalan tek kol oydu.

### Gercek sebep: TRL modeli float32 yukluyor

`grpo_trainer.py` docstring'i: *"If dtype is not specified in
args.model_init_kwargs, it defaults to float32. This differs from
from_pretrained..."*. Biz `model_init_kwargs` hic gecmiyorduk.
`bf16=True` yalnizca autocast, yani hesabi bf16 yapiyor, agirligi degil.

Uc ayaktan dogrulandi:
1. TRL dokumani
2. `train_grpo.py` sadece `bf16=True` geciyordu
3. **Olculen ilk plato 17.3 GB = 4.66e9 x 4 bayt** (fp32 agirlik 17.4 GiB)

Kelime dagarcigi **248320** (Bolum 4'te "152k" yaziyor -- yanlis; oradaki
7.58 GiB aritmetigi yine de dogru degeri kullanmis). Bununla:

```
mb=2, L=6144, fp32 satiri: 6144 x 248320 x 4 bayt = 5.68 GiB
```

**Her OOM'daki 5.68 GiB tam olarak bu.** fp32 oldugu icin
`selective_log_softmax` fp32 dalina dusuyor ve `logsumexp` satir basina
bu boyutta gecici aciyor. Sayi mikro-batch'ten BAGIMSIZ -- yani
"mikro-batch'i dusurelim" de tam isabet olmayacakti.

### Kosu 3: bf16 + liger + vLLM 0.50

| | kosu 2 (fp32) | kosu 3 (bf16) |
|---|---|---|
| agirlik platosu | 17.3 GB | **9.0 GB** |
| uretim platosu | 40.0 GB | 48-50 GB (vLLM'e pay geri verildi) |
| **loss tepesi** | **75.8 GB** | **64.5 GB** |
| mapping-failed uyarisi | 11+ | **0** |

Duvara pay 3.5 GB -> **14.75 GB**.

### Liger: devrede, ama sandigimiz kismi degil

`apply_liger_kernel_to_qwen3_5` varsayilanlari: `rms_norm` ve `swiglu`
devrede (ornek uzerinde yamaniyor; sinif adi ayni kaldigi icin sinif-adi
probu goremiyor, CPU'da Triton hatasi kanitladi). `rope` qwen3_5'te
desteklenmiyor. `fused_linear_cross_entropy` ise **hic tetiklenmiyor**:

`liger_kernel/transformers/model/qwen3_5.py:88`
`skip_logits = self.training and (labels is not None or shift_labels is not None)`

GRPO `model(...)`'i labels vermeden cagirip `outputs.logits`'i kendi
okuyor, yani logits her zaman materyalize ediliyor. Liger bize aktivasyon
ve hiz kazandiriyor, logits tensorunu ORTADAN KALDIRMIYOR. Kaldirmak icin
GRPO'nun kendi `selective_log_softmax`'ini degistirmek gerekir; TRL'de
GRPO icin `use_liger_loss` yok (sadece experimental/sdft ve iw_opd'de var).

### vLLM payi: sinir uretim fazinda

Kisit uyku fazinda degil. vLLM uyanikken egiticinin sabit kismi (bf16'da
~17 GB) da kartta duruyor: `vllm_util x 80 + 17 <= 79`. Ustelik daha buyuk
KV bir yerden sonra bos duruyor -- bir adimda 32 dizi x 10240 baglam
~ **327k token** yetiyor. 0.50 secildi; 0.70-0.80 uretim fazinda egiticiyi
sikistirir, karsiliginda bir sey vermez.

### Uretim neden bu kadar uzun suruyor (olculdu, cozulmedi)

Adim suresi 544 / 566 sn. 60 sn boyunca 1 sn'lik ornekleme:
**GPU >=%80'de zamanin %81'i, <%20'de sadece %4** -- sandbox beklemesi
degil, gercek uretim hesabi.

Ama aritmetik tutmuyor: `completions/mean_length 2160` x 32 episode
= ~69k cikis token / 555 sn = **~124 token/sn**. A100'de 4B model 32 yollu
batch'te bunun kat kat ustu beklenir. Sebep cok turlu yapi: turlar SERI
(`tools/call_frequency` ~9), her tur ayri `generate` cagrisi ve butun batch
en yavas diziyi bekliyor. Ayrica TRL colocate LLM'i
`max_num_batched_tokens=4096` ile kuruyor ve `enable_prefix_caching`'i hic
gecmiyor (vLLM varsayilanina birakiyor). Bunlar incelenmedi.

`max_num_seqs = micro_batch x tp x steps_per_generation = 2 x 1 x 16 = 32`.
Mikro-batch'i 4 yapmak birikimi 8'e dusurdugu icin bu carpimi degistirmez,
yani uretim eszamanliligini ARTIRMAZ; kazanci yalnizca loss fazinda.

### ACIK KUSUR: bant olcumu egitimle ayni parametrelerde kosmuyor

Egitimden gelen iki anormallik:

```
frac_reward_zero_std = 0.50   (sweep'ten beklenen 0.14)
reward               = 0.66 - 0.75
```

Beklenen deger banttaki her gorev icin `p^8 + (1-p)^8` ile hesaplandi.
Bantta en kalabalik grup `p=1/8` (64 kabuk gorevinin 20'si) ve onlarin yeni
bir 8 rollout'ta 0/8 gelme ihtimali %34. Yani bant "olu"yu eliyor,
"oluye yakin"i eleyemiyor -- uyelik tek bir 8-orneklik tahmin. Ayrica
metrik adim basina 4 problemden hesaplandigi icin yalnizca
0/0.25/0.5/0.75/1 olabiliyor; kosu2'nin 0.25'i ile kosu3'un 0.5'i
arasindaki fark **4 problemden biri**, kiyaslanabilir degil.

Bugun bu sinifta uc hata ciktigi icin tek tek dogrulandi:
- Egitim gercekten **sadece bantta**: `veri_kumesi` `--gorevler` verilince
  split'e hic bakmadan erken donuyor; log "egitim kumesi: 103 gorev";
  eslesen 64 kabuk gorevinin solved araligi 1..6, bant disi sifir.
- Odul gercekten **ikili**: `checks.py:373` `1.0 if solved else 0.0`.
  `partial` ayri tutuluyor, egitimde kullanilmiyor.

Geriye kalan tutarsizlik -- ikili odulde `reward=0.75`, yani 32 rollout'un
24'u tam cozulmus, oysa gorevler 8'de 1-6 bandinda -- su farka isaret ediyor:

| | Sweep (bant olcumu) | Egitim |
|---|---|---|
| yanit basina token | **2048** (`runner.py:58`, `VDS_MAX_TOKENS`) | 6144'e kadar |
| tur sayisi | **`task.max_turns`** (gorev basina 8/10/12/14/16) | **duz 12** (`--max-komut`) |

255 gorevin **199'unda** `max_turns` 12 degil. Yani bant, modelin daha dar
bir butceyle olculdugu bir dunyada belirlendi; egitimde ayni gorevler
kolaylasiyor, gruplar 8/8'e oturuyor ve sifir varyans uretiyor. Tek sebep
iki anormalligi birden acikliyor.

**KARAR: sweep bundan sonra egitimle AYNI inference parametreleriyle
kosacak.** Kosu 3'e dokunulmadi.

### Kosu 3 durumu
`--mufredati-kullan` ile sweep atlandi (model degismemisti, ~40 dk kazanildi).
2/25 adim, ~558 sn/adim, tahmini 3.5 sa. Bellek tepesi 64.5 GB, uyari yok.

### Uyusmazligin asil buyugu: sweep ile egitim FARKLI PROTOKOL kullaniyor

`train-epoch1.log`'daki completion tablosunda system prompt'un Ingilizce
gorunmesi uzerine bakildi. O metin bizim degil, **Qwen3.5'in kendi chat
template'i**; tokenizer enjekte ediyor:

```
# Tools
You have access to the following functions:
<tools>{"type":"function","function":{"name":"run_command",
  "description":"Sandbox terminalinde tek bir shell komutu calistirir.",...}}</tools>
If you choose to call a function ONLY reply in the following format...
<IMPORTANT> Reminder: - Function calls MUST follow the specified format
```

Bizim Turkce arac aciklamamiz onun icinde duruyor ve model Turkce cevap
veriyor -- yani cikti dilini bozmuyor. Template'i ezmek riskli: Bolum 4'te
`--tool-call-parser hermes` Qwen3.5'in `<function=...>` bicimini tanimadigi
icin zaten basimiz yanmisti. **Dokunulmadi.**

Asil bulgu bunu ararken cikti:

| | Egitim | Sweep / eval |
|---|---|---|
| protokol | Qwen native tool-calling (`<tool_call><function=`) | `--protocol text` (`<komut>` etiketleri) |
| system prompt | chat template + Ingilizce iskele | bizim Turkce `TEXT_PROTOCOL_PROMPT` |
| yanit basina token | 6144'e kadar | 2048 (`runner.py:58`) |
| tur sayisi | duz 12 (`--max-komut`) | `task.max_turns` (8/10/12/14/16) |

Yani bant olcumu modelden **bambaska bir arayuzle** cevap istiyor. Gorev
ayni, ama modelin gordugu dunya farkli. Native tool-calling modelin
uzerinde egitildigi bicim; metin protokolu bizim Gemma icin yazdigimiz
yedek yol (`runner.py`'deki yoruma bakilirsa native tool calling'i olmayan
modeller icin eklenmisti).

Bu, `reward = 0.72` (32 episode'un 22-23'u cozuluyor) ile bant olcumunun
"8'de 1-6" demesi arasindaki tutarsizligin en guclu aciklamasi -- token
butcesi farkindan daha buyuk bir etki.

**KARAR (guncellendi): sweep egitimle ayni PROTOKOLU ve ayni inference
parametrelerini kullanacak.** Sadece sicaklik/rollout degil: protokol,
yanit token butcesi ve tur sayisi da hizalanacak. Aksi halde mufredat,
egitimin icinde bulundugu dunyanin zorlugunu olcmuyor.

Hizalama yonu tercih gerektiriyor:
- protokol: egitim native kullaniyor ve model onunla egitilmis -> sweep de
  native'e gecmeli (`--protocol text` yerine arac cagrisi).
- tur sayisi: `task.max_turns` gorev tasariminin parcasi, duz 12 degil.
  Ama TRL'in `environment_factory`'si goreve gore parametre almiyor;
  `TerminalOrtami`'nin `task_dir`'den kendi okumasi gerekir.

### Tur limiti bagliyici degil (olculdu)

78 held-out gorevinde `mean_turns / task.max_turns` dagilimi:

```
0.2 |1   0.3 |7   0.4 |16   0.5 |20   0.6 |14   0.7 |13   0.8 |5   0.9 |2
```

Hicbir gorevin ortalamasi tavana dayanmiyor; tepe 0.5'te. Yani episode'lar
tur limitinden elenmiyor, model tur bitmeden ya cozuyor ya birakiyor.
`--max-komut 12`'yi buyutmek muhtemelen bir sey kazandirmaz. (Uyari: bu
ortalama uzerinden, 8 rollout'un biri tavana dayanmis olabilir; ayrica
sweep'in butcesiyle olculdu, egitimde `tools/call_frequency` 8.8-10.1 ile
12'ye daha yakin.)

### Uyku yamasinin sonucu (kosu 4)

Tur basina `sleep(level=2)` kaldirildi, adim basina bir kez uyunuyor.
Bellek izinde dogrulandi: kurulus sonrasi sicramalar

```
kosu 3 (tur uykusu): 154 sicrama / 93 dk = 1.7/dk, 72 saniyede 6 tam dongu
kosu 4 (tek uyku)  :   9 sicrama / 19 dk = 0.5/dk, sicramalar ~5-6 dk arayla
```

Kosu 4'teki sicramalar adim sinirlari (uyu -> uyan -> KV rampasi), tur
degil; aralarda 5-6 dakika dumduz.

| | kosu 3 | kosu 4 |
|---|---|---|
| step_time | 544.3 / 565.6 | **495.6 / 508.2** |
| ortalama | 555 sn | **502 sn** |
| bellek tepesi | 64.5 GB | 64.5 GB (degismedi, beklendigi gibi) |
| clipped_ratio | 0.094 / 0.156 | 0.063 / 0.156 |

**%9.5 kazanc.** Loss fazi adimin ~%62'si ve yama ona dokunmuyor, yani
kazancin tamami uretimden geliyor: uretimde ~%23.

Geri donus: `--tur-uykusu` bayragi eski davranisi geri getiriyor.

### Loss fazinda 30 katlik acik (olculdu, aciklanmadi)

Loss fazi ~314 sn. Teorik: govde fwd+bwd ~1400 TFLOP, A100'de gercekci
150 TFLOPS ile ~10 sn. Aradaki fark 30 kat.

`lm_head` + log-softmax bu isin **%6'si** (88 TFLOP), yani fuzyonlu
cross-entropy tarzi cozumler burada hizdan cok **bellek** kazandirir --
ve bellek su an sikismiyor (tepe 64.5, duvar 79.25). Ayrica GRPO
cross-entropy kullanmiyor: ornekklenmis token'in log-olasiligini isteyip
politika orani hesapliyor, yani hazir CE modulu takilamaz; liger'in
GRPO'ya ozel chunked loss'u gerekir ve o TRL'in `_compute_loss`'unu
devre disi birakmayi gerektirir.

`selective_log_softmax`'daki Python dongusu batch satiri uzerinde donuyor
(mikro-batch 2 -> iki iterasyon), token uzerinde degil; TRL'in yorumu
"loop to reduce peak mem consumption" diyor, yani kasitli bir takas.
Vektorlestirmek tepe bellegi artirir.

Attention n^2 degil: Qwen3.5 hibrit (`layer_types` = linear_attention +
full_attention), `flash-attn` kurulu degil, transformers `sdpa`'ya dusuyor
ve o da matrisi materyalize etmiyor. Bu acigin sebebi attention degil.

Kalan supheliler: gradient checkpointing'in yeniden hesabi, mikro-batch
2'de kartin dolmamasi, ve faz atfinin (bellek izinden cikarim) yanlis
olma ihtimali. **Siradaki dogru adim tahmin degil profiler.**

---

## Bolum 9 — Iki oturum ayni calisma agacinda (2026-09-10)

Bugun proje iki ayri ayakta paralel calisti: **uretim** (bu bolumun konusu,
ayrinti `VERI-URETIMI.md`'nin 2026-09-10 ekleri) ve **egitim profili**
(Bolum 8'in devami). Ikisi ayni klasoru, ayni git calisma agacini kullandi.

### Bulgu: commit'ler birbirine karisti

Uretim tarafinin calisma agacindaki duzenlemeleri, egitim tarafinin
commit'lerine **suprulmus** durumda. `git commit -a` / `git add -A`
yarim kalmis baska bir isin dosyalarini da alir; burada olan bu.

| Degisiklik | Girdigi commit |
|---|---|
| `PY_KALIPLARI` (yapisal eksenler, `seeds.py`) | `eb6d0a6` "Profile the loss phase instead of the whole step" |
| `gate_ara_durum` (`gates.py`) | `f43c14f` "Stop the profiler from eating the host out of memory" |
| `gate_yapisal_uyum` (`gates.py`) | `f76461a` "Add a torch.compile flag..." |

Kod kayip degil, calisiyor ve test edildi. Sorun **gecmisin yalan
soylemesi**: uc commit mesaji kendi diff'ini anlatmiyor. Bu projede
commit mesajlari ayrintili yazildigi icin ("neden" tasiyorlar), yanlis
atif ilerideki arkeolojiyi dogrudan yaniltir -- `git log -S` ile bir
kusurun ne zaman girdigini ararken profil commit'ine cikilacak.

Gecmis **yeniden yazilmadi**: `main` paylasilmis olabilir ve rebase
yikici. Karar sahibinin secmesi icin duruyor.

### Onlem
Paralel is icin **ayri git worktree**. Ayni agacta iki oturum varken
`git add -A` / `git commit -a` kullanmamak; degisiklikleri yol vererek
eklemek (`git add <yol>`). Bugun uretim tarafinin son commit'i (`69bbe13`)
boyle atildi ve yalnizca kendi dosyalarini icerir.

### Uretim tarafinin gun ozeti
Korpus 255 -> 512. Dort gercek kusur bulundu ve kapatildi (egitimin gorev
basina `max_turns` okumamasi, `derive.py`'nin ayni dosyaya iki test
yazmayi sessizce yasaklamasi, `fingerprint`'in kod ailesinde hicbir sey
ayirt etmemesi, endpoint on-kontrolunun yonlendirilen model id'sini
reddetmesi). Ayrinti ve olcumler `VERI-URETIMI.md`'de.

**Siradaki dogru adim uretim degil olcum:** 512 gorevin hicbirinin bandi
olculmedi ve bu adim API parasi harcamiyor.
