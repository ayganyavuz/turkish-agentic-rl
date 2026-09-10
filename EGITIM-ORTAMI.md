# EGITIM-ORTAMI

Sandbox, GRPO egitimi, olcum ve Colab operasyonu. Bu ayak **Colab'de
(A100 80GB)** kosuyor.

Kardes dosya: `VERI-URETIMI.md` (korpus, kapilar, split).
Tarihce: `PROJECT-TRACE.md` (Bolum 0-8, append-only gunluk; bu dosya oradan
turetildi, orasi silinmedi).

> Yeni bulgular bu dosyaya **append** edilir; gecmis yeniden yazilmaz.

---

## ALTIN KURAL

**Yeni bir bayrak/ayar ekledigin zaman, kosuyu baslatmadan ONCE gercekten
etki ettigini test et.** Bu projede bugune kadar bu sinifta **bes** hata
cikti; her biri saatler yakti:

1. `--split/--bolum` eklendi ama `asama_bant` onu cagirmiyordu → "held-out"
   diye 80 gorevin tamami (egitim dahil) puanlandi.
2. `--sicaklik 0` hic uygulanmadi: runner `TEMPERATURE`'i import aninda
   okuyordu, pipeline runner'i `main()`'den ONCE import ediyordu. Butun
   evaller 0.7'de kostu, deterministik saniildi. Ayni model ayni 54 kabuk
   gorevinde 5/54 ve 13/54 verdi.
3. `max_completion 2048` ortuk uzunluk cezasiydi (asagida).
4. `--vllm-bellek` `dongu.py`'de 0.45 olarak **sabit yazilmisti**, bayrak degildi.
5. `vllm_durdur` yalnizca `pkill -f 'vllm serve'` yapiyordu; KV cache'i tutan
   `VLLM::EngineCore` alt-sureci baska adla hayatta kalip kartin 68 GB'ini
   elinde tutuyordu.

> **Bu liste bes maddede kalmadi.** 10 Eylul'de bu siniftan **dort** hata
> daha cikti (`--sicaklik` hicbir yere bagli degildi, `VDS_MAX_TOKENS`
> modul seviyesinde okunuyordu, `--profil`/`--tur-uykusu` `dongu.py`'den
> iletilmiyordu, `TerminalOrtami` gorev basina `max_turns` okumuyordu) ve
> profiler'in kendisi olcumu iki kez oldurdu. Tam liste: dosyanin
> sonundaki **BUG KAYDI — 10 Eylul**.

Test sekli: `kos` / `vllm_baslat` stub'lanip komut yakalanir, ya da sahte
`GRPOConfig` ile alanlar yakalanir. Ornekler repoda calisir durumda.

---

## Ortam

| | |
|---|---|
| GPU | A100-SXM4-80GB (79.25 GiB kullanilabilir) |
| Makine | 12 CPU, 167 GB RAM, 189 GB disk, Linux 6.6 |
| Docker | **yok** → `LocalSandbox` |
| Calisan bilesim | vllm 0.27.1, trl 1.12.0, transformers 5.16.1, torch 2.13.0+cu130, liger_kernel 0.8.2 |
| Model | `Qwen/Qwen3.5-4B` — **4.66e9 parametre**, checkpoint 8.68 GiB, **vocab 248320**, hidden 2560, 32 katman |

### Kurulum tuzaklari
- `pip install vllm` torch'u cu130'a cikariyor, Colab'in `torchaudio`'su
  cu128 kaliyor → vLLM hic ayaga kalkmiyor. **`torchaudio` kaldirilmali**, ve
  vLLM kurulumu onu geri getirdigi icin kurulumdan SONRA tekrar kaldirilmali.
- `--tool-call-parser hermes` Qwen3.5'in `<function=...><parameter=...>`
  bicimini tanimiyor.
- `reasoning_effort="none"` Qwen3.5'te **ciktiyi bozuyor** ("12+17" → 55,
  3 token). Dogru anahtar `chat_template_kwargs={"enable_thinking": False}`.
- Colab runtime yeniden atanabiliyor: repo, model cache, kurulumlar silinir.
  **Once Drive baglanir**, cikti ve checkpoint oraya yazilir.
- vLLM ilk ayaga kalkisinda `torch.compile` cache'i bossa **500 sn** suruyor
  (cache sicakken 160 sn). Takilma sanma.

---

## `LocalSandbox`

`sandbox.py`'da DockerSandbox ile **birebir ayni arayuz**. Dogrulanan semantik:
`cd` ve `export` turlar arasi tasinmiyor (`docker exec` ile ayni), zaman asimi
124, tehlikeli komut reddi 126, `snapshot()` calisiyor.

**Izolasyon yok** — kodun basina acikca yazili. Ucuz korumalar: yikici komut
deseni reddi, per-komut timeout, HOME/TMPDIR kok dizinde.

`checks.py`: `run_stdout_eq` imaj adi bosken host'ta kosuyor — kod ailesinin
TUM dogrulamasi bu check'ten geciyor, yani sart.

`task.py:make_sandbox()` moda gore seciyor; mod tek yerden ayarlaniyor
(`sandbox.yerel_kullan`). `--sandbox {docker,yerel}` bayragi sweep, gates ve
pipeline'da.

---

## Dusunme modu: ACIK

Kullanici karari. Gerekce: kod gorevlerinde muhakeme gercekten yardimci
oluyor ve en buyuk sorunumuz OLU-zor havuzuydu. Maliyeti olculdu: istek
basina ~530 cikis token, epoch suresi yaklasik iki katina cikiyor.

(Bolum 1'deki ilk plan "kapali" diyordu; olcumden sonra degisti.)

---

## Egitim yapilandirmasi

| | |
|---|---|
| Model | Qwen3.5-4B, full fine-tune, `adamw_bnb_8bit` |
| dtype | **bfloat16** (`model_init_kwargs={"dtype": ...}`) — asagidaki fp32 tuzagina bak |
| Korpus | o epoch'un mufredati (`mufredat/epochN.txt`) |
| G (rollout/grup) | 8 |
| prompt-batch | 4 → adim basina 32 episode |
| micro-batch | 2 → `gradient_accumulation_steps` 16 |
| `max_completion_length` | **6144** |
| vLLM | colocate + uyku modu, `--vllm-bellek 0.50`, baglam 10240 |
| Odul | ikili |
| KL (`beta`) | 0 |
| lr | 1e-6 |
| liger | acik |

### Boyutlar nereden geliyor
```
generation_batch_size = micro_batch x num_processes x steps_per_generation
                      = 2 x 1 x 16 = 32          (grpo_config.py:1083-1085)
max_num_seqs          = micro_batch x tp x steps_per_generation = 32
```
Yani `micro_batch x steps_per_generation` carpimi sabit: **mikro-batch'i
buyutmek uretim eszamanliligini ARTIRMAZ**, yalnizca loss fazini etkiler.

Bir adim = 32 episode uretilir → vLLM uyur → 16 mikro-geciste gradyan
biriktirilir → **tek** optimizer adimi → agirliklar vLLM'e senkronlanir.

---

## Bellek: ne nereyi dolduruyor (olculdu)

Kart 79.25 GiB. bf16 ile:

| Ne | Boyut | Ne zaman |
|---|---|---|
| agirliklar (bf16) | 8.68 GiB | hep |
| optimizer (8-bit x 2) | 8.68 GiB | hep |
| → **egitici dinlenme** | **~17-22 GB** | olculdu |
| vLLM (agirlik kopyasi + KV) | ~40 GB (0.50) | yalnizca uretimde |
| → **uretim platosu** | **48-50 GB** | olculdu |
| gradyanlar + logits + aktivasyon | ~+42 GB | loss fazinda |
| → **loss tepesi** | **64.5 GB** | olculdu |

### TUZAK: TRL modeli float32 yukluyor
`grpo_trainer.py` docstring'i: *"If dtype is not specified in
args.model_init_kwargs, it defaults to float32. This differs from
from_pretrained..."*. `bf16=True` **yalnizca autocast** — hesabi bf16 yapar,
agirligi degil.

Uc ayaktan dogrulandi: (1) TRL dokumani, (2) kodumuz sadece `bf16=True`
geciyordu, (3) **olculen ilk plato 17.3 GB = 4.66e9 x 4 bayt**.

fp32'de logits de fp32 uretiliyor ve `selective_log_softmax` fp32 dalina
dusuyor; `logsumexp` **satir basina** su boyutta gecici aciyor:
```
6144 x 248320 x 4 bayt = 5.68 GiB
```
**Her OOM'daki "Tried to allocate 5.68 GiB" tam olarak bu.** Sayi
mikro-batch'ten BAGIMSIZ.

bf16'ya gecince: agirlik 17.4 → 8.7 GiB, gradyan ayni sekilde yariya,
loss tepesi **75.8 → 64.5 GB**.

### Uyku modu gercekten calisiyor
`vllm_generation.py`'de kosulsuz: init'te `sleep(2)`, `sync_weights`'te
`wake_up(["weights"])`, generate oncesi `wake_up(["kv_cache"])`, sonrasi
`sleep(2)`. Olculdu: **loss fazinda vLLM 18 GB birakiyor** (40 → 22 GB).

Sonuc: **`--vllm-bellek`'i kismak loss tepesine yaramaz.** Kisit uretim
fazinda: `vllm_util x 80 + ~17 <= 79` → `util <= 0.77`. Ustelik daha buyuk KV
bos duruyor — bir adimda 32 dizi x 10240 baglam ≈ **327k token** yetiyor.
0.50 secildi.

### ORTUK UZUNLUK CEZASI (cozuldu)
`max_completion_length = 2048` ile episode'larin **%40'i kesiliyordu**:
```
korelasyon(odul, kesilme_orani)    = -0.507
korelasyon(odul, ortalama_uzunluk) = -0.523
korelasyon(kesilme, uzunluk)       = +0.753
```
Uzun → kesiliyor → odul 0 → negatif avantaj. RLVR'de bilinen patoloji
(DAPO kesilen rollout'lari ayri ele aliyor). **6144'e cikarildi**;
`clipped_ratio` %40 → **%6-16**.

Not: "egitim boyunca uzunluk 1379 → 1100 dustu" gozlemi **yanlisti** — uc
noktalar secilmisti. Gercek: ilk 10 adim 1221, son 10 adim 1201.

---

## Liger: devrede, ama sandigimiz kismi degil

`apply_liger_kernel_to_qwen3_5` varsayilanlari:

| Parca | Durum |
|---|---|
| `rms_norm` | devrede, **ornek uzerinde** yamaniyor (sinif adi ayni kalir; sinif-adi probu goremez, CPU'da Triton hatasi kanitlar) |
| `swiglu` | devrede (`LigerQwen3MoeSwiGLUMLP`) |
| `fused_linear_cross_entropy` | **hic tetiklenmiyor** |
| `rope` | qwen3_5'te desteklenmiyor (hibrit attention) |

Tetiklenmeme sebebi `liger_kernel/transformers/model/qwen3_5.py:88`:
```python
skip_logits = self.training and (labels is not None or shift_labels is not None)
```
GRPO `model(...)`'i **labels vermeden** cagirip `outputs.logits`'i kendi
okuyor → logits her zaman materyalize ediliyor. Liger aktivasyon ve hiz
kazandiriyor, **logits tensorunu ortadan kaldirmiyor**.

Kaldirmak icin GRPO'nun kendi `selective_log_softmax`'ini degistirmek gerekir;
TRL'de GRPO icin `use_liger_loss` **yok** (sadece `experimental/sdft` ve
`iw_opd`'de). liger'in `chunked_loss` altinda GRPO'ya ozel bir loss var ama
TRL'in `_compute_loss`'unu devre disi birakmayi gerektirir.

---

## vLLM'i tur basina uyutma (duzeltildi)

TRL'in cok turlu dongusu:
```
_tool_call_loop:  while idxs_with_tool and iteration_num < max_tool_calling_iterations
  -> _generate_single_turn -> vllm_generation.generate()
       basta: llm.wake_up(["kv_cache"])
       sonda: llm.sleep(level=2)      <- KV CACHE'I ATIYOR
```
Yani prefix cache **turlar arasinda yasayamiyordu**: ~9 turun her biri butun
konusmayi sifirdan prefill ediyordu, maliyet tur sayisinin karesiyle
buyuyordu. Bellek izinde adim basina 10-14 uyku/uyanma salinimi
(21.7 ↔ 59.9 GB).

Duzeltme (`train_grpo.py: episode_boyu_uyanik_tut`): `_generate` sarmalaniyor,
tur dongusu boyunca uyku kapali, dongu bitince **bir kez** `sleep(level=2)`.
Sarmalamadan once agirliklar uyandiriliyor (level 2 onlari da atiyor;
yoksa `sync_weights` serbest birakilmis bellege yazar).

Olculen:
```
kosu 3 (tur uykusu): 154 sicrama / 93 dk = 1.7/dk, 72 sn'de 6 tam dongu
kosu 4 (tek uyku)  :   9 sicrama / 19 dk = 0.5/dk, sicramalar ~5-6 dk arayla
step_time          : 555 sn -> 502 sn  (%9.5)
bellek tepesi      : 64.5 GB -> 64.5 GB (degismedi, beklendigi gibi)
```
Geri donus: `--tur-uykusu` bayragi.

---

## Hiz: nereye gidiyor (olculdu, tam cozulmedi)

Adim ~502 sn. 60 sn boyunca 1 sn'lik ornekleme: **GPU >=%80'de zamanin %81'i,
<%20'de %4** → sandbox beklemesi degil, gercek hesap.

Faz ayrimi (bellek izinden, **cikarim**): uretim ~%38, loss ~%62.

Aritmetik tutmuyor:
- Uretim: 32 episode x ~2160 token = ~69k cikis token / ~190 sn ≈ **124 tok/sn**.
  A100'de tek bir dizinin decode hizi bile (~215 tok/sn) bunun ustunde.
- Loss: govde fwd+bwd ~1400 TFLOP, 150 TFLOPS ile ~10 sn beklenir, **~314 sn**
  olculuyor. **30 kat acik.**

Elenen supheliler:
- **attention n^2 degil**: Qwen3.5 hibrit (`layer_types` = `linear_attention`
  + `full_attention`), `flash-attn` kurulu degil, transformers `sdpa`'ya
  dusuyor ve o da matrisi materyalize etmiyor.
- **`lm_head` + log-softmax bu isin %6'si** (88 TFLOP). Fuzyonlu cozumler
  burada hizdan cok **bellek** kazandirir, ve bellek su an sikismiyor.
- **`selective_log_softmax`'daki Python dongusu** batch satiri uzerinde donuyor
  (mikro-batch 2 → iki iterasyon), token uzerinde degil. TRL'in yorumu
  "loop to reduce peak mem consumption" — kasitli takas; vektorlestirmek tepe
  bellegi artirir.

Kalan supheliler: gradient checkpointing'in yeniden hesabi, mikro-batch 2'de
kartin dolmamasi, ve faz atfinin yanlis olma ihtimali.

**Siradaki dogru adim tahmin degil profiler.** `--profil N` bayragi eklendi:
ilk adimi isinma sayar, sonraki N adimi `torch.profiler` ile olcer, tabloyu
loga basar ve chrome trace'i `--cikti` altina yazar.

### Mikro-batch 4 degerlendirmesi
Kazanc kucuk: 16 gecis → 8 gecis ama **toplam hesap ayni**. Loss fazinda
%10-25, adimda ~%6-15. Maliyeti: logits 5.68 → 11.37 GiB, tepe ~72 GB,
duvara pay 14.75 → **~7 GB**. Su an **onerilmiyor**.

### Donanim
- Tek H100: loss hesap-bagimli (~2.2x), uretim bant-bagimli (~1.6x) ama
  turlar seri oldugu icin gerceklesecek kazanc daha az → **~1.5-1.8x**.
- 1 node = 4x H100 (MareNostrum 5 ACC): epoch ~5:00 → **~1:20**.
  **Kart bellegini teyit et** — MN5'in H100'leri 64 GB olabilir, tepemiz
  64.5 GB. Oyleyse FSDP sart (model 4'e bolununce kart basina ~7 GB).
  Node basina 80 CPU cekirdegi var, sandbox darbogazi kalkar.
- **Ayrik generation/training kartlari tek basina kazandirmaz** — GRPO senkron,
  fazlar seri. Kazanc **boru hattindan** gelir (batch k+1 uretilirken k ile
  egit): adim `uretim + loss` yerine `max(uretim, loss)`. Bedeli off-policy'ye
  gecmek; TRL'de `vllm_importance_sampling_correction` ve
  `experimental/async_grpo` var.

---

## Dongu (`dongu.py`)

```
her epoch:
  1) SWEEP    -- o anki modelle train split'ine G=8 rollout
  2) MUFREDAT -- 1..7 cozen gorevler secilir (0/8 ve 8/8 disarida)
  3) EGITIM   -- yalnizca o listeyle 1 epoch
  4) EVAL     -- held-out, sweep ile ayni sekilde: N rollout, pass-rate
  5) checkpoint bir sonraki epoch'un modeli olur
```

**Mufredat neden her epoch yeniden olculuyor**: bir grubun butun rollout'lari
ayni sonucu verirse avantaj sifir olur; model degistikce hangi gorevin hangi
bantta oldugu da degisiyor.

**Held-out neden sweep gibi olculuyor**: tek rollout'la "cozdu mu" cok
gurultuluydu — ayni taban model ayni 54 kabuk gorevinde bir kosuda 5,
digerinde 13 cozdu. Bu fark, epoch'lar arasinda aradigimiz farktan buyuk.

**Sweep, egitim ve eval ayri sureclerde** kosuyor: uc asama da GPU'nun
tamamini istiyor.

Faydali bayraklar:
- `--mufredati-kullan` — kosu egitim tamamlanmadan koptuysa model degismemistir,
  diskteki `mufredat/epochN.txt` hala gecerli. Bir kez kullanilir (~40 dk kazanc).
- `--atla-baseline` — epoch 0 held-out olcumunu atla.
- `--protokol {native,text,auto}` — varsayilan **native**.
- `--dtype`, `--vllm-bellek`, `--vllm-baglam`, `--tur-uykusu`, `--profil`.

### Sure
| Asama | Sure |
|---|---|
| sweep (173 x 8) | ~55 dk |
| egitim (25 adim x ~502 sn) | ~3:30 |
| checkpoint → Drive (~9 GB) | ~10 dk |
| held-out eval (78 x 8) | ~25 dk |
| **epoch toplami** | **~5:00** |

**5 epoch ≈ 21 saat — tek Colab oturumuna sigmiyor.** Cozulmedi. Her epoch'u
ayri oturumda kosmak gerekiyor: checkpoint Drive'da, sonraki oturum
`--taban-model <ciktilar/epochN>` ile devam eder ve `--mufredati-kullan`
**kaldirilmalidir** (model artik degismis).

---

## Olcum sonuclari

### Baseline (held-out 78, G=8 pass-rate, taban model)
| Aile | pass@8 | En az bir kez cozulen |
|---|---|---|
| Kod (24) | %32.3 | 17/24 |
| Kabuk (54) | %18.3 | 25/54 |
| **Toplam (78)** | **%22.6** | 42/78 |

### Epoch 1 mufredati (173 train gorevi, G=8)
bantta **103** (64 kabuk + 39 kod), olu-zor **65**, olu-kolay **5**.

### GECERSIZ olcum — Bolum 6'nin epoch 1 sonucu
"kod -20.8, kabuk +11.1" sonucu **sicaklik hatasi** yuzunden gecersiz
(butun evaller 0.7'de kostu, 0 saniliyordu). Kirli olcumler
`kirli-olcumler/` altinda, o kosunun modeli `kosu1-kabuk-agirlikli/` altinda.

Ayni bolumun **"sadece kod ile egit" onerisi de zayifladi**: gerekcesi
kabuk icin "%9.3, 54'un 49'u cozulemiyor" idi; o sayi tek rollout'luk ve
sicakligi bozuk olcumdendi. G=8'de kabugun 54'unun 25'i en az bir kez
cozuluyor.

### Egitim metrikleri (kosu 4, adim 1-2)
```
step_time 495.6 / 508.2      reward 0.6875 (22/32) / 0.71875 (23/32)
clipped_ratio 0.063 / 0.156  tools/call_frequency 10.09 / 8.84
frac_reward_zero_std 0 / 0
```
`reward` degerleri 1/32'nin kati — ikili odul varsayimi dogrulaniyor.

---

## ACIK KUSUR: bant olcumu egitimle ayni dunyada degil

Iki anormallik:
```
frac_reward_zero_std = 0.50   (sweep'ten beklenen 0.14)
reward               = 0.66 - 0.75   (32 episode'un 22-23'u cozuluyor)
```
oysa gorevler "8'de 1-6" bandinda secildi.

Beklenen deger banttaki her gorev icin `p^8 + (1-p)^8` ile hesaplandi.
Bantta en kalabalik grup `p=1/8` (64 kabuk gorevinin 20'si), yeni bir
8 rollout'ta 0/8 gelme ihtimali **%34** — bant "olu"yu eliyor, "oluye
yakin"i eleyemiyor. Ayrica metrik adim basina 4 problemden hesaplandigi icin
yalnizca 0/0.25/0.5/0.75/1 olabiliyor: **kosu2'nin 0.25'i ile kosu3'un 0.5'i
arasindaki fark 4 problemden biri, kiyaslanabilir degil.**

Tek tek dogrulandi (bu sinifta bes hata ciktigi icin):
- Egitim gercekten **sadece bantta**: `veri_kumesi` `--gorevler` verilince
  split'e bakmadan erken donuyor; log "egitim kumesi: 103 gorev"; eslesen
  64 kabuk gorevinin `solved` araligi 1..6, bant disi sifir.
- Odul gercekten **ikili**: `checks.py:373`.

Geriye kalan aciklama **parametre/protokol uyusmazligi**:

| | Egitim | Sweep / eval (eski) |
|---|---|---|
| protokol | Qwen native tool-calling | `--protocol text` (`<komut>`) |
| system prompt | chat template + Ingilizce iskele | Turkce `TEXT_PROTOCOL_PROMPT` |
| yanit basina token | 6144'e kadar | **2048** (`runner.py:58`, `VDS_MAX_TOKENS`) |
| tur sayisi | duz 12 (`--max-komut`) | `task.max_turns` (8/10/12/14/16) |

Metin protokolu native tool calling'i olmayan modeller (Gemma) icin yazilmis
yedek yoldu; Qwen3.5 native bicimle egitilmis. 255 gorevin 199'unda
`max_turns` 12 degil.

**KARAR: sweep egitimle AYNI PROTOKOLU ve AYNI inference parametrelerini
kullanacak.**

Durum:
- protokol: `dongu.py --protokol` eklendi, varsayilan **native** (`f93ccf2`).
  **Henuz kosulmadi** — bant yeniden olculmeli.
- yanit token butcesi: `VDS_MAX_TOKENS` hala 2048. **Yapilmadi.**
- tur sayisi: **YAPILDI** (2026-09-10). `environment_factory` goreve gore
  parametre almiyor ama gerekmiyordu: butce `reset()` icinde,
  `Task.load(task_dir)`'dan sonra ayarlaniyor. `--tur-butcesi duz` eski
  davranisi yeniden uretilebilir tutuyor. Dogrulandi: `max_turns: 16` olan
  gorev 16, duz modda 12 aliyor.

### Yan bulgu: Ingilizce system prompt bizim degil
`train-epoch1.log`'daki Ingilizce metin **Qwen3.5'in kendi chat template'i**
(`# Tools / You have access to the following functions / <IMPORTANT> Reminder`).
Bizim Turkce arac aciklamamiz onun icinde; model Turkce cevap veriyor.
Template'i ezmek tool-call ayristirmasini bozma riski tasiyor. **Dokunulmadi.**

### Tur limiti bagliyici degil (olculdu)
78 held-out gorevinde `mean_turns / task.max_turns`:
```
0.2|1  0.3|7  0.4|16  0.5|20  0.6|14  0.7|13  0.8|5  0.9|2
```
Hicbiri tavana dayanmiyor, tepe 0.5'te. `--max-komut 12`'yi buyutmek
muhtemelen bir sey kazandirmaz. (Uyari: ortalama uzerinden; ayrica sweep'in
butcesiyle olculdu, egitimde `tools/call_frequency` 8.8-10.1 ile 12'ye yakin.)

---

## Yapilmamis isler

- **SWE-Bench hic yapilmadi.** Plandaki Faz 0.2 (sb-cli erisim testi) ve
  Faz 4 (30 gorev: 10 kolay / 10 orta / 10 zor, sabit liste, kendi
  `runner.py`'imiz scaffold olarak). Ana metrik held-out; SWE-bench ikincil.
- **Basari esigi**: held-out cozum oraninda epoch 0'a gore +%15 mutlak, son
  iki epoch'ta korunuyor. Henuz gecerli bir epoch sonucu yok.
- **Yerel/Docker esdegerligi Linux'ta olculmedi** (Windows karsilastirmasi
  yaniltici).
- **W&B** bagli degil (`--wandb` bayragi var, bos).
- 5 epoch'un tek oturuma sigmamasi.

---

## Kosu arsivi (Drive: `/content/drive/MyDrive/turkish-agentic-rl/`)

| Dizin | Icerik |
|---|---|
| `kirli-olcumler/` | sicaklik/split hatali olcumler + vLLM oldurulurken yazilmis bozuk sweep |
| `kosu1-kabuk-agirlikli/` | ilk kosunun modeli ve olcumleri |
| `sweep1-yedek/` | epoch 1 mufredatini ureten sweep (kabuk) + `epoch1.txt` |
| `kosu3-turuykusu/` | bf16 + liger, tur uykusu ile (8 adim) |
| `kosu4-tekuyku/` | bf16 + liger + tek uyku (8 adim) |
| `olcumler/`, `mufredat/`, `ciktilar/` | canli kosunun ciktilari |

---

## 10 Eylul: kosu runtime ile birlikte gitti

Devir notu "kosu durduruldu, 8/25 adimda arsivlendi" diyordu. **Durdurulmamis.**
MCP baglandiginda epoch 1 egitimi 08:04'ten beri kosuyordu ve **11/25**
adimdaydi (`epoch 0.44`). Iki dakika sonra **Colab runtime yeniden atandi**:
`nvidia-smi: command not found`, `/content` bos, `torch.cuda.is_available()`
False, `uptime` 3 dakika.

Kayip: ~1.6 saat hesap. Checkpoint yazilmamisti (`ciktilar/epoch1` altinda
yalnizca `completions_*.parquet`, model yok).

Sureci `ps` ile yakalarken ogrenilenler:
- PPID 1 idi, yani `dongu.py` coktan olmustu; `train_grpo` yetim kosuyordu.
  Dongu'nun eval/checkpoint adimlari zaten yapilmayacakti.
- Log dosyasi `train-epoch1.log` Drive'da yoktu ama sureç fd'sini tutuyordu:
  `/proc/<pid>/fd/1` uzerinden okundu. Hedef `kosu4-tekuyku/train-epoch1.log`
  idi -- arsiv dizini sureç canliyken tasinmis, fd tasinan dosyayi takip
  etmisti. **Arsivleme yapilmis, sureç oldurulmemis.**

Olculen adim sureleri epoch 0.36-0.44 araliginda: `513.8`, `479.1`, **`892.7`**.
Sonuncusu 502 ortalamasinin cok ustunde; tek gozlem, aciklanmadi.

---

## ALTIN KURAL, ALTINCI ORNEK: `--sicaklik` hic baglanmamisti

Hata #2 "duzeltildi" saniliyordu. Yalnizca **yarisi** duzeltilmisti.

`runner.sicaklik()` cagri aninda `VDS_TEMPERATURE` okuyacak sekilde
yazilmis (dogru). Ama bayragi o env'e yazan halka **hic kurulmamis**:

- `pipeline.py`'de `--sicaklik` yalnizca `add_argument`'ta geciyordu;
  `args.sicaklik` hicbir yerde okunmuyordu (dosyada `temperature` kelimesi
  bile yok).
- `dongu.py`'nin alt sureclere verdigi `ortam` sozlugunde `VDS_TEMPERATURE`
  yoktu.

Yani `dongu --sicaklik 1.0` varsayilani **etkisizdi**: butun sweep ve
eval'ler -- epoch 0 baseline ve sweep1 dahil -- `runner`'in varsayilani
**0.7**'de kostu.

Bu, "bant egitimle ayni dunyada degil" kusurunun **ikinci** bilesenidir.
Protokolun yaninda sicaklik da ayrisiyordu: GRPO rollout'lari 1.0'da
ornekleniyor, bant 0.7'de olculuyordu. Dusuk sicaklik daha az cesitlilik
demek, yani bant gorevleri gercekte olduklarindan daha kararli gorundu.

### Ayni sinifta: `VDS_MAX_TOKENS`
`MAX_TOKENS` modul seviyesinde okunuyordu -- hata #2'nin birebir yapisi.
`sicaklik()` fonksiyona cevrilirken bu atlanmis.

### Ve: `--profil` / `--tur-uykusu` dongu'ya bagli degildi
Ikisi de yalnizca `train_grpo.py`'de vardi. Bu dosyanin "Faydali bayraklar"
listesi ikisini de `dongu.py` bayragi sayiyordu -- **yanlisti**. `dongu`
ile kosulan bir egitim profil alamazdi. (Ayrica `--profil`, devir notunun
"commit'lenmedi" dediginin aksine `6d6f553`'e dokumanlarla birlikte girmis.)

### Duzeltme
- `runner.py`: `MAX_TOKENS` → `max_tokens()`, cagri aninda okunur.
- `pipeline.py`: `--sicaklik` ve yeni `--yanit-token`, `main()` icinde
  `os.environ`'a **yaziliyor**. runner ikisini de cagri aninda okudugu icin
  modulun onceden import edilmis olmasi sorun degil.
- `dongu.py`: `--yanit-token` (varsayilan `--max-completion`, yani 6144),
  `--profil` ve `--tur-uykusu` iletiliyor.

### Test (kosudan ONCE, altin kural)
Uc ayri test, hepsi yerelde GPU'suz:
1. `kos` stub'lanip egitim komutu yakalandi → `--profil 2 --tur-uykusu` var.
2. Ayni sekilde sweep komutu → `--sicaklik 1.0 --yanit-token 6144
   --protocol native` var.
3. `runner` **onceden import edildikten sonra** `pipeline.main()` cagrildi:
   ```
   once : 0.7  2048      <- hatanin kaniti
   sonra: 1.0  6144
   ```

Boylece "bant egitimle ayni dunyada degil" kusurunun **uc bileseni de
kapandi**: protokol native, token butcesi 6144, sicaklik 1.0, ve tur
sayisi artik gorev basina (`--tur-butcesi task`, 2026-09-10; uretim
ayaginda bulundu, ayrinti `VERI-URETIMI.md`).

**10. `TerminalOrtami` gorev basina `max_turns` okumuyordu.** Altin kural
sinifinin onuncusu ve en pahalisi: deger `task.yaml`'da vardi, 255 gorevin
**199'unda** 12 degildi, ve hicbir yerde okunmuyordu. Uzun gorevler
bitiremeden kesiliyordu -- **gorev zor degildi, butcesi yanlisti.** Bant
olcumu bu yuzden zorlugu sistematik olarak abartiyordu.

> Uc bilesenin ucu de ayni sekli tasiyordu: bant, egitimin kullandigi
> parametreden farkli bir parametreyle olculuyordu. Uc bagimsiz kusur
> degil, tek bir kor nokta.

Bant olcumunun onunde artik engel yok.

> Onceki tum sweep/eval sayilari 0.7'de ve 2048 ile olculdu. Bant yeniden
> olculdugunde bunlarla kiyaslanamaz.

---

## FA2 bayragi (kullanici karari; henuz olculmedi)

`--attn {sdpa,flash_attention_2,eager}`, varsayilan `sdpa`. `train_grpo` ve
`dongu`'da. `model_init_kwargs`'a `attn_implementation` olarak giriyor.

Kapsam -- **yalnizca egitim (loss) fazi**. Uretim vLLM'de kosuyor ve vLLM
attention backend'ini kendi seciyor; bu bayrak ona dokunmaz. Yani adimin
~%38'lik uretim kismi degismez.

Beklenti olcusunu bastan kucuk tutmak gerekiyor, iki sebeple:
1. Qwen3.5 **hibrit**: `layer_types` bir kismi `linear_attention`. FA2
   yalnizca `full_attention` katmanlarinda devreye girer.
2. `sdpa` zaten attention matrisini materyalize etmiyor. Hiz farki
   cekirdek kalitesinden gelir, karmasiklik sinifindan degil.

Test (kosudan once): argparse blogu izole calistirildi, varsayilan `sdpa`
ve `--attn flash_attention_2` verildiginde `model_init_kwargs`'a giden
dict dogrulandi; gecersiz deger `choices` ile reddediliyor. `dongu`'nun
urettigi egitim komutunda `--attn` iletiliyor (kos stub'lanarak).

**Kurulumda dogrulanmasi gereken** (yerelde torch yok, Colab'de bakilacak):
- `flash-attn` tekeri cu130 icin var mi; yoksa kaynaktan derleme 30+ dk.
- Qwen3.5 sinifi FA2'yi kabul ediyor mu (`_supports_flash_attn`).
- Model yuklendikten sonra `model.config._attn_implementation` gercekten
  `flash_attention_2` mi -- transformers desteklemeyen bir modelde sessizce
  sdpa'ya dusebilir. **Bu, bayragin etki ettigini gosteren tek kanit.**

---

## Colab kurulumu: iki yeni tuzak (10 Eylul, olculdu)

### 1. Kurulum hucresi torch'u IMPORT ETMEMELI
Kurulum hucresi GPU kontrolu icin basta `import torch` yapiyordu. Colab'in
on-yuklu torch'u **2.11.0+cu128**; pip onu **2.13.0+cu130**'a yukseltiyor.
Bir kez import edilince kernel eskisini bellekte tutuyor ve vLLM'in cu130
kutuphaneleriyle eslesmiyor:
```
ImportError: libcudart.so.13: cannot open shared object file
```
Teshis `pip list` ile netlesti: **diskte 2.13.0, kernel'de 2.11.0+cu128**.
Yani kurulum bozuk degildi, kernel bayatti.

Duzeltme: GPU kontrolu `nvidia-smi` ile; kurulum dogrulamasi **temiz bir alt
surecte**. Dogru soru zaten buydu -- egitim `nohup python` ile ayri surecte
kosuyor, onemli olan O surecin gordugu ortam.

Alt surecte dogrulanan (kurulum saglam):
```
torch 2.13.0+cu130 | cuda 13.0 | gorunuyor True
vllm 0.27.1 | trl 1.12.0 | transformers 5.16.1 | liger 0.8.2
liger qwen3_5 yamasi: VAR
uyku yamasi tutar mi: True
```

### 2. `find /` mount'lu Drive'i tariyor
Teshis icin yazilan `find / -name 'libcudart.so*'` Drive'a girip kernel'i
kilitledi. Colab'de genis `find` kullanma.

### 3. flash-attn cu130 icin teker sunmuyor
`pip install flash-attn --no-build-isolation` kaynaktan derlemeye girdi ve
**16 saniyede** `bdist_wheel` hatasiyla dustu. Model tarafinda engel yok:
```
Qwen3_5ForCausalLM._supports_flash_attn = True
```
Yani engel modelde degil, paketin CUDA 13 ile derlenememesinde. `--attn`
bayragi yerinde duruyor; FA2'ye deger mi sorusu **profil tablosuna** birakildi
(attention'in gercek payi olculdukten sonra derleme zahmetine girilir).

---

## Profil denemesi 1: profiler'in KENDISI OOM ettirdi (10 Eylul)

Kosu `--profil 2` ile basladi, dogru basladi:
```
vLLM uyku yamasi: AKTIF
attn_impl: istenen=sdpa gercek=sdpa
profil acik: 1. adim isinma, sonraki 2 adim olculecek
adim 0 (isinma):     step_time 497.6      <- 502 referansiyla tutarli
adim 1 (profilli):   step_time 845.4      <- profiler yuku
```
Sonra adim 2'de oldu. **Kart degil, host RAM**:
```
oom-kill: constraint=CONSTRAINT_MEMCG, task=python3, pid=46097
Memory cgroup out of memory: Killed process 46097
anon-rss: 171184872 kB  (~171 GB; makinede 167 GB var)
```

Sebep `ProfilCallback`'in ayarlari: `record_shapes=True` ve
`profile_memory=True`, ustelik **iki tam adim** boyunca. Bir adimda ~9 tur
uretim + 16 mikro-gecis var, yani olay sayisi zaten cok; her olaya sekil ve
bellek kaydi eklenince trace host RAM'de birikti. Tabloyu uretmeye sira bile
gelmedi.

Duzeltme: ikisi de varsayilan **kapali**, `--profil-ayrinti` ile aciliyor.
Test: sahte `torch.profiler.profile` ile cagri argumanlari yakalandi
(varsayilan `False/False`, bayrakla `True/True`).

### Bu kosudan yine de kalan olcumler
| | |
|---|---|
| `attn_impl` | istenen=sdpa **gercek=sdpa** (yeni log satiri calisiyor) |
| uyku yamasi | **AKTIF** (sessizce eski davranisa dusmedi) |
| uretim platosu | **48.7 GB** — dokumandaki 48-50 GB ile birebir |
| loss tepesi (profilli) | 68.4 GB — profiler yuku dahil, 64.5 referansinin ustunde |
| adim 0 | `step_time 497.6`, reward 0.594, clipped 0.031 |
| adim 1 (profilli) | `step_time 845.4`, reward 0.656, clipped 0.125 |

`step_time` 845.4 profilli oldugu icin hiz tartismasinda **kullanilamaz**.

### flash-attn: cu130 tekeri yok
`pip install flash-attn --no-build-isolation` kaynaktan derlemeye girip
**16 saniyede** `bdist_wheel` hatasiyla dustu. Model tarafinda engel yok
(`Qwen3_5ForCausalLM._supports_flash_attn = True`), engel paketin CUDA 13
ile derlenememesi. `--attn` bayragi yerinde; karar profil tablosuna bagli.

---

## Faz olcumu artik cikarim degil (10 Eylul)

"Uretim ~%38 / loss ~%62" ayrimi bugune kadar **bellek izinden cikarimdi**.
Hangi hizlandirmanin degdigi tamamen bu ayrima bagli oldugu icin bu, uzerine
oneri kurulacak bir sayi degildi.

`_generate` uyku yamasi icin zaten sarmalaniyordu; disina bir kronometre
kondu (`uretim_suresini_olc` + `FazCallback`). Uretim **dogrudan** olculuyor,
adim suresinin kalani loss. Profiler gerekmiyor, maliyeti yok, her adimda
basiliyor:
```
faz[adim N]: toplam X sn = uretim Y (A%) + loss Z (B%)
```
Kosulsuz acik -- yama tutmazsa sessizce devre disi kaliyor.

Test: sahte `trl.GRPOTrainer` ile 0.30 sn "uretim" + 0.20 sn "loss" →
`60% / 40%` basildi; ikinci adimda sayacin sifirlandigi ve yamanin iki kez
uygulanmadigi dogrulandi.

### Hizlandirma kaldiraclari (kanit seviyeleriyle)
| Kaldirac | Beklenen | Kanit | Bedel |
|---|---|---|---|
| dusunme modunu kapatmak | ~2x | **olculdu**: istek basina ~530 ek cikis token | kalite; olcumden sonra bilincli acilmisti |
| `--max-komut` dusurmek | uretim dogrusal kisalir | egitimde `call_frequency` **9.5-9.9 / 12** (tavana yakin) | cozulemeyen gorev → odul duser |
| boru hatti (async GRPO) | adim = `max(uretim, loss)` | TRL `experimental/async_grpo` | off-policy'ye gecmek |
| mikro-batch 4 | ~%6-15 | olculdu | duvara pay 14.75 → ~7 GB |
| H100 / 4xH100 | ~1.5-1.8x / ~3.7x | tahmin | donanim |

**Onemli:** egitimde tur sayisi 12'nin 9.5-9.9'unu tuketiyor. Turlar seri
kostugu icin uretim maliyeti dogrudan tur sayisiyla artiyor. Held-out
olcumunde tepe 0.5'teydi ama o **sweep'in butcesiyle** olculmustu (2048
token, metin protokolu); egitimde durum farkli.

---

## Profil denemesi 2 ve kapsamin daraltilmasi (10 Eylul)

Hafif profille (`record_shapes`/`profile_memory` kapali) tek adim olculdu.
Adim tamamlandi, profiler kapandi, ama **tablo uretilirken RAM buyumeye
devam etti**:
```
profiler_stop 11:36:18   (profilli adim 775 sn)
+16 dk   RSS 79.7 GB   bos 87 GB
+20 dk   RSS 97.1 GB   bos 71 GB     -> ~4.3 GB/dk, yine OOM'a gidiyordu
```
Kesildi. Tablo alinamadi.

**Kok sorun ayarlar degil KAPSAM.** Bir adimda ~9 tur uretim + 16
mikro-gecis var; uretimin milyonlarca kernel cagrisi trace'e giriyor ve
sonra atiliyor, cunku sorumuz loss fazinin **icinde** zamanin nereye
gittigi.

Yeni yaklasim (`loss_fazini_profille`): `ProfilCallback` kaldirildi, yerine
`compute_loss` sarmalaniyor. Ilk mikro-gecis isinma, sonraki N mikro-gecis
olculuyor, tablo basiliyor, profiler bir daha acilmiyor. **Uretim trace'e
hic girmiyor.** `--profil N` artik adim degil mikro-gecis sayisi.

Test: sahte `torch.profiler` ve sahte `trl.GRPOTrainer` ile 6 mikro-gecis
kosuldu -- 1. isinma, 2-3. olculdu, tablo basildi, 4-6. profilsiz gecti;
`compute_loss` altisinda da gercek fonksiyona ulasti; yama iki kez
uygulanmiyor.

### Bu iki denemeden kalan temiz olcumler
| | |
|---|---|
| `attn_impl` | istenen=sdpa **gercek=sdpa** |
| uyku yamasi | **AKTIF** |
| uretim platosu | **48.7 GB** (dokumandaki 48-50 ile birebir) |
| isinma adimi (profilsiz) | `step_time` **497.6** ve **476.4** |
| adim basina token | `num_tokens` 7.2e4 - 7.9e4 |
| tur sayisi | `tools/call_frequency` **9.5 - 9.9** / 12 |
| odul | 0.594 / 0.656 |

Teorik karsilastirma (adim basina ~72-79k token): govde fwd+bwd
`6 x 79k x 4.66e9 ~ 2200 TFLOP`, gradient checkpointing ile ~2900;
A100'de gercekci 150-200 TFLOPS ile **15-20 sn** eder. "Loss ~%62" dogruysa
olculen ~300 sn, yani **~20 kat acik**. Ama o yuzde hala cikarim --
`FazCallback` bir sonraki kosuda kesinlestirecek.

---

## PROFIL SONUCU: acik attention'da degil, kernel launch'ta (10 Eylul)

### Faz ayrimi artik OLCULDU
```
faz[adim 1]: toplam 879.0 sn = uretim 155.1 (18%) + loss 724.0 (82%)
```
Bu adim profilli, yani loss siskin (profilsiz adimlar 476-497 sn). Ama
**uretim 155.1 sn kesin** -- profiler yalnizca `compute_loss`'a bagli.
Profilsiz adima gore duzeltince: **uretim ~%32, loss ~%68.**

**Onemli duzeltme:** dokumandaki "uretim 124 tok/sn" hesabi yanlis faz
atfindan geliyordu. Gercek: 32 episode x ~2100 token ~ 67k token / 155 sn
= **~433 tok/sn**. Uretim fazinda ciddi bir kayip YOK. Kayip loss fazinda.

### Loss fazinin ic dagilimi (2 mikro-gecis)
```
Self CPU time total : 32.143 s      <-- CUDA'dan BUYUK
Self CUDA time total: 20.531 s
```
| Op | CUDA | Cagri |
|---|---|---|
| `aten::mm` | 5.29 s (26%) | 4 168 |
| `vectorized_elementwise_kernel` | 5.08 s (25%) | **63 777** |
| `aten::add_` | 5.05 s (25%) | **40 775** |
| `aten::copy_` | 2.42 s (12%) | **72 845** |
| `ampere_sgemm_*` (**fp32**) | 4.95 s (24%) | ~54 000 |
| `aten::bmm` | 1.88 s (9%) | **63 266** (self CPU 4.4 s) |
| `_efficient_attention_fwd+bwd` | 1.48 s (7%) | **32** |
| `ampere_bf16_s16816gemm_*` | 1.82 s (9%) | 2 328 |

### Iki teshis
1. **Kernel launch seli / CPU-bound.** Iki mikro-geciste 63k `bmm`, 72k
   `copy_`, 63k elementwise. Self CPU > Self CUDA, yani GPU bekliyor, CPU
   kernel firlatiyor. Bir yerde Python dongusu on binlerce ufak kernel
   uretiyor. En guclu aday Qwen3.5'in **`linear_attention` katmanlari**:
   gercek attention (`_efficient_attention`) yalnizca **32 cagri** ve
   zamanin %7'si, yani maliyet orada DEGIL.
2. **GEMM'lerin dortte biri fp32.** `ampere_sgemm_*` fp32; bf16 olanlar
   ayri gorunuyor (`ampere_bf16_s16816gemm_*`, %9). `--dtype bfloat16`
   verilmesine ragmen hesabin bir kismi fp32'de -- linear attention
   state'inin kararlilik icin fp32 tutulmasi bunu aciklar.

Onceki dokumandaki "govde fwd+bwd 30 kat acik" ifadesinin adresi budur:
attention karmasikligi degil, **launch overhead + fp32 yol**.

### Elenen supheliler (bu tabloyla)
- attention n^2 degil ve pahali degil: 32 cagri, %7.
- `lm_head` + log-softmax tek basina baskin degil (`aten::mm` %26'nin
  tamami degil; 4168 cagri govdeyi de kapsiyor).
- gradient checkpointing'in yeniden hesabi tek basina aciklamiyor --
  aciklasa GEMM'ler baskin olurdu, oysa elementwise/copy/bmm seli baskin.

### Siradaki olculebilir hipotezler
| Hipotez | Nasil test edilir |
|---|---|
| launch overhead → `torch.compile` / CUDA graphs kernel'leri birlestirir | `torch_compile=True` ile bir adim, `faz[adim]` karsilastir |
| linear attention referans implementasyonu yavas | `flash-linear-attention` (`fla`) kurulu mu, transformers onu seciyor mu |
| fp32 yol gereksiz | fp32 sgemm'leri hangi modulun urettigini `--profil-ayrinti` ile sekillerden bul |

### DUZELTME: tablodaki yuzdeler toplanamaz
Yukaridaki tablo hem `aten::` op satirlarini hem de onlarin altindaki CUDA
kernel satirlarini birlikte listeliyor (`aten::mm` ve `ampere_*gemm_*` ayni
isin iki seviyesi). Yuzdeler bu yuzden **cift sayiliyor**; toplandiginda
%100'u asiyor. "fp32 GEMM zamanin %24'u" ifadesi bu hatayi tasiyor.

Guvenle soylenebilecek olan: `Self CUDA time total 20.531 s` (2 mikro-gecis)
ve **fp32 kernel'lerin varligi**.

### bf16 notu hala gecerli, bulgu baska bir katmanda
Agirliklar gercekten bf16:
- olculen agirlik platosu **8.68 GiB** = 4.66e9 x 2 bayt (fp32 olsaydi 17.4;
  tuzak yakalanmadan once tam da 17.3 olculmustu),
- tabloda bf16 GEMM'ler acikca var: `ampere_bf16_s16816gemm_bf16_*`.

Cozulen tuzak "**butun model** fp32 yukleniyordu" idi. Yeni bulgu farkli:
**bf16 modelin icinde bazi alt-hesaplar fp32'de** --  `ampere_sgemm_*` ve
adi tereddute yer birakmayan `magma_sgemmEx_kernel<float, float, float,...>`.
En olasi yer linear attention'in state hesabi (bu katmanlarda state sayisal
kararlilik icin genelde fp32 tutulur), yani muhtemelen kasitli ama pahali.

**Henuz olculmedi**: fp32 kernel'leri hangi modul uretiyor. `--profil-ayrinti`
ile tensor sekilleri toplanarak bulunabilir; kapsam artik 2 mikro-gecisle
sinirli oldugu icin ilk denemedeki RAM patlamasi riski yok.

### Temiz faz ayrimi (profilsiz adim)
```
faz[adim 1]: toplam 879.0 sn = uretim 155.1 (18%) + loss 724.0 (82%)   <- profilli
faz[adim 2]: toplam 553.6 sn = uretim 181.5 (33%) + loss 372.1 (67%)   <- TEMIZ
```
**Aranan sayi adim 2.** Cikarim %38/%62 idi; olculen **%33/%67**. Yakin, ama
loss biraz daha baskin ve artik bir kronometreden geliyor.

Uretim verimi: 32 episode x `completions/mean_length` 2104 ~ 67k cikis token
/ 181.5 sn = **~370 tok/sn**. Bu A100'de 32 dizi ve dusunme modu aciksa
makul. **Uretim fazinda aranacak bir sey yok.**

Loss: **372 sn**, teorik 15-20 sn. Acik ~20 kat ve profil tablosu adresini
gosteriyor: kernel launch seli (Self CPU > Self CUDA) + fp32 alt-yol.

Kosu adim 2'den sonra durduruldu; profil ve faz ayrimi alinmisti, kalan 23
adim bosuna GPU yakacakti. `loss-trace.json` (1.39 GB) Drive'da
`profil/` altinda.

### Siradaki is icin oncelik notu
Hiz calismasinin sonraki adimi (torch.compile / fla / fp32 kaynagi) ile
**bandin native+1.0+6144 ile yeniden olculmesi** ayri isler. Ikincisi
projenin ilerlemesi icin daha kritik: mevcut `mufredat/epoch1.txt` 0.7
sicaklikta, 2048 token ve metin protokoluyle secildi, yani **gecersiz**.

---

# BUG KAYDI — 10 Eylul

Bugun cikan butun hatalar tek yerde. Ilk ucu ve #10 **altin kural
sinifindan**: bayrak/ayar vardi ama etki etmiyordu. Dosyanin basindaki
"bes hata" listesi artik **on**.

#10 uretim ayaginda bulundu (`VERI-URETIMI.md`) ama buraya ait: kusur
egitim tarafinda. Iki ayak ayri dosyalara yazdigi icin bir sure iki
listede de gorunmedi.

## Altin kural sinifi (bayrak var, etki yok)

**6. `--sicaklik` hicbir yere bagli degildi.**
`pipeline.py`'de yalnizca `add_argument`'ta geciyordu; `args.sicaklik`
hicbir yerde okunmuyordu (dosyada `temperature` kelimesi bile yoktu) ve
`dongu.py` `VDS_TEMPERATURE`'i alt sureclere vermiyordu. Hata #2'nin
**yarisi** duzeltilmisti: `runner.sicaklik()` cagri aninda okuyacak sekilde
yazilmis ama bayragi o env'e yazan halka hic kurulmamisti.
→ Butun sweep/eval'ler (epoch 0 baseline ve sweep1 dahil) 1.0 istenirken
**0.7**'de kostu. Egitim rollout'lari 1.0'da ornekleniyor, yani bant
egitimden farkli bir dunyada olculuyordu.

**7. `VDS_MAX_TOKENS` modul seviyesinde okunuyordu.**
Hata #2'nin birebir yapisi; `sicaklik()` fonksiyona cevrilirken atlanmis.
Bant 2048 ile olculurken egitim 6144 veriyordu.

**8. `--profil` ve `--tur-uykusu` yalnizca `train_grpo.py`'deydi.**
`dongu.py` ikisini de iletmiyordu, yani dongu ile kosulan bir egitim
**profil alamazdi**. Dosyanin "Faydali bayraklar" listesi ikisini de dongu
bayragi sayiyordu -- yanlisti.

**9. Profiler'in kendisi olcumu oldurdu.** (asagida ayrintili)

## Colab ortam tuzaklari

**Kurulum hucresi torch'u IMPORT ETMEMELI.** On-yuklu torch 2.11+cu128;
pip 2.13+cu130'a yukseltiyor ama kernel eskisini bellekte tutuyor →
`ImportError: libcudart.so.13`. Teshis: `pip list` diskte 2.13.0 derken
kernel 2.11.0+cu128 diyordu. GPU kontrolu `nvidia-smi` ile, dogrulama
**alt surecte** yapilmali -- egitim zaten ayri surecte kosuyor.

**Drive ikinci kez mount edilemiyor.** Runtime yeniden atanmadiysa Drive
zaten baglidir ve `drive.mount()` `ValueError: Mountpoint must not already
contain files` ile patlar. `os.path.isdir('/content/drive/MyDrive')` ile
kosullu mount gerekiyor.

**`find /` mount'lu Drive'i tariyor** ve kernel'i kilitliyor. Colab'de genis
`find` kullanma.

**Runtime bir gunde IKI KEZ yeniden atandi** (biri kosan egitimi 11/25
adimda oldurdu, digeri paketleri ve repoyu sildi). Her seferinde ~8 dk
kurulum. Uzun kosular buna gore planlanmali.

**flash-attn'in cu130 tekeri yok.** `--no-build-isolation` ile bile kaynak
derlemesi 16 saniyede `bdist_wheel` hatasiyla dustu. Model tarafinda engel
yok (`_supports_flash_attn = True`).

## Profiler tuzaklari

**Adimin tamamini profillemek imkansiz.** Bir adimda ~9 tur uretim + 16
mikro-gecis var:
- `record_shapes`+`profile_memory` acik, 2 adim → host RAM **171 GB**, OOM
  (makinede 167 GB).
- Ikisi kapali, 1 adim → adim bitti ama **tablo uretilirken** RAM 4 dakikada
  79.7 → 97.1 GB, yine OOM'a gidiyordu; kesildi.
→ Cozum kapsami daraltmak: `compute_loss` sarmalanip yalnizca birkac
mikro-gecis olculuyor. Uretim trace'e hic girmiyor.

**Profil tablosundaki yuzdeler toplanamaz.** Tablo hem `aten::` op'larini
hem altlarindaki CUDA kernel'lerini listeliyor; cift sayim var.

## Kucuk

**Izleme hucresi eski basligi ariyordu.** `PROFIL -- CUDA` grep'leniyordu,
yeni baslik `PROFIL -- loss fazi`; tablo basilmisti ama gorunmuyordu.

## Su anki durum
Colab runtime **yeni** (ikinci yeniden atama sonrasi): paketler ve repo
kurulu DEGIL, A100 bagli, Drive bagli. `--derle` (torch.compile) denemesi
**kosulmadi** -- karsilastirma referansi hazir:
```
derlemesiz: faz[adim 2] = 553.6 sn = uretim 181.5 (33%) + loss 372.1 (67%)
```
