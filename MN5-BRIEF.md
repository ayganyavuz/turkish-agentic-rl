# MareNostrum 5 devir notu

Bu dosya, MN5 üzerinde bu projeyle çalışacak bir sonraki ajanın (ya da
kişinin) hazır bulması gereken her şeyi içeriyor. Bir ajana bağlam olarak
olduğu gibi verilebilir.

Buradaki her madde **10 Eylül 2026'da makinede ölçüldü**. "Doğrulanmadı"
diye işaretlenmemiş hiçbir şey tahmin değil. Bir madde artık tutmuyorsa
düzelt — çıkarım yapıp üstüne bina etme.

---

## 0. Önce oku: kırmızı çizgiler

`/gpfs/projects/etur22` **paylaşımlı bir proje dizini**. Altında başka
kişilerin aylarca sürmüş işleri duruyor (`bilsem`, `eren`, `yusuf`,
`diverse-sft`, model checkpoint'leri...). GPFS'te silinen geri gelmez.

1. **Yalnızca `/gpfs/projects/etur22/turkish-agentic-rl` altında çalış.**
   Kardeş dizinlerin hiçbirine yazma, taşıma, silme.
2. **`rm` kullanma.** Gerekirse önce sor. İzin verildiğinde bile joker
   karakter kullanma, dosyaları tek tek adıyla say.
3. Sıkıştırma açarken orijinali koru: `gunzip -c x.gz > x`, `gunzip x` değil.
4. `/gpfs/projects` **%94 dolu (143G boş)**. Büyük transferlerden önce
   `df -h /gpfs/projects` bak.

---

## 1. Erişim

```
ssh ytu248581@alogin1.bsc.es        # anahtar kayıtlı, parolasız
```

Çalışma alanı ve içindekiler:

```
/gpfs/projects/etur22/turkish-agentic-rl/
├── repo/                     git klonu (0d55fb0 + singularity.patch uygulanmış)
├── base.sif  smoke-001.sif   görev sandbox imajları (42M'er)
├── turkish-agentic-rl.bundle repo/'nun `origin`'i -- SİLME, git remote kırılır
├── singularity.patch         commit'lenmemiş Singularity farkı
├── kos-sweep.sh              envs referans sweep'i (modül+env sarmalayıcı)
├── kos-korpus.sh <korpus>    verilen korpusta referans sweep
├── tek2.sh <korpus> <görev>  tek görevi --ayrintili koştur
└── tmp/                      SINGULARITY_TMPDIR
```

`alogin1` bir **login node, GPU yok**. Sürücü/CUDA sürümü öğrenmek için
`srun` ile bir ACC node'u tutman gerekiyor (henüz yapılmadı).

---

## 2. İnternet YOK

pypi ve github timeout veriyor, proxy değişkeni de yok. `pip install`,
`git clone https://...`, `huggingface-cli download` — hiçbiri çalışmaz.
Compute node'lar login node'dan daha kapalı, daha iyisini bekleme.

Her şey dizüstünden taşınır:

```bash
# repo (dizüstünde)
git bundle create x.bundle HEAD main     # "HEAD main" şart, bkz. §6
scp x.bundle ytu248581@alogin1.bsc.es:/gpfs/projects/etur22/turkish-agentic-rl/
# MN5'te
git clone x.bundle repo

# imaj (dizüstünde)
docker save verifiable-dataset/base:latest -o base.tar
scp base.tar ...
# MN5'te
module load singularity/4.1.5
singularity build base.sif docker-archive://base.tar
```

Repo GitHub'da **public** (`cxrbon16/turkish-agentic-rl`), yani dizüstünden
anonim HTTPS ile çekilebiliyor — ama MN5'ten değil.

---

## 3. Modül sistemi (Lmod)

**`module load` komutunu ASLA pipe'lama.** Pipeline alt kabukta koşar,
`PATH` değişikliği kaybolur ve sonraki komut "command not found" der.
Bu tuzak bu projede iki kez zaman yedi.

```bash
module load singularity/4.1.5 2>&1 | tail -2     # YANLIŞ, PATH kaybolur
module load singularity/4.1.5 2>/dev/null        # doğru
```

### Python

```bash
module load intel impi mkl hdf5 python/3.12.1
```

Bu sırayla; eksik biri Lmod hatası verir (zincir: python→intel, hdf5→impi).
Sonuç: **Python 3.12.1, pyyaml 6.0.1** — referans sweep için yeterli.
`openai` YOK (yalnızca model çağıran koşularda lazım).
Sistemin `/usr/bin/python3`'ü 3.9.21, projeye yetmez.

`python/3.12.1-gcc` varyantı gcc→mkl→hdf5→intel diye dönen bir zincire
giriyor, uğraşma; intel varyantı çalışıyor.

### Singularity

```bash
module load singularity/4.1.5      # 3.11.5 default -- açıkça 4.1.5 seç
```

Modül adı `singularity`, `apptainer` diye bir modül yok. Binary aslında
`/apps/GPP/SINGULARITY/4.1.5/bin` altında (ACC modülü GPP ağacını gösteriyor,
bu normal).

---

## 4. Sandbox: Docker yok, Singularity var

`sandbox.py` üç mod veriyor: `docker`, `singularity`, `yerel`.
**MN5'te `--sandbox singularity` kullan.**

- `docker` — MN5'te imkânsız (root ve daemon ister).
- `yerel` — **MN5'te KULLANMA.** İzolasyon yok, modelin ürettiği komut
  doğrudan GPFS'e yazar. Colab için yazıldı, orada VM atılabilir olduğu için
  kabul edilmişti; paylaşımlı makinede kabul edilemez ve BSC kullanım
  politikasını ihlal eder.

Kurulum:

```bash
export VDS_SIF_DIZIN=/gpfs/projects/etur22/turkish-agentic-rl
export SINGULARITY_TMPDIR=/gpfs/projects/etur22/turkish-agentic-rl/tmp
```

`VDS_SIF_DIZIN` görev dosyalarındaki `image:` alanını `.sif`e çeviriyor:
`verifiable-dataset/base:latest` → `$VDS_SIF_DIZIN/base.sif`. Korpusta
sadece iki imaj var: `base` (568 görev) ve `smoke-001` (1 görev).

### `--cleanenv` şart

`--containall`'ı `instance start`'a vermek sonraki `exec` çağrılarının
ortamını temizlemiyor. Modül yüklü bir kabuktan koşunca host'un
`PYTHONHOME`'u konteynere sızıyor, içerideki python host stdlib'ini arayıp
`ModuleNotFoundError: No module named 'encodings'` ile ölüyor — ve bu
görev bozukmuş gibi FAIL olarak görünüyor. Kod bunu `exec`'e ayrıca
`--cleanenv` vererek çözüyor (`sandbox.py`, `SingularitySandbox.exec`).
**175 görevlik korpusta 4 görevi tek başına düşürüyordu.**

---

## 5. `max_user_namespaces = 0` — kalıcı kısıt

MN5'te user namespace'ler tamamen kapalı. Bunun üç sonucu var ve hiçbiri
aşılamaz:

| Kayıp | Sonuç |
|---|---|
| `--fakeroot` | Konteyner **her zaman senin uid'inle** koşuyor. Docker'daki root paritesi imkânsız. |
| `--net --network none` | Ağ izolasyonu yok. `VDS_AG_IZOLE` varsayılanı bu yüzden kapalı; `=1` yaparsan instance hiç başlamaz. |
| cgroup yönetimi | `--cpus/--memory/--pids-limit` karşılığı yok. Kaynak sınırı Slurm cgroup'una kalıyor. |

### Docker ile ölçülen fark

| | Docker | Singularity (MN5) |
|---|---|---|
| kullanıcı | root | senin uid'in |
| `$HOME` | `/root` | 64M tmpfs, **host home'a sızmıyor** (doğrulandı) |
| `/tmp`, `/workspace` | yazılabilir | yazılabilir |
| `/etc`, `/usr/local` | yazılabilir | salt okunur |
| mod-000 dosya okuma | root okur | okunamaz |

Son satır gerçek bir sapma yaratıyor: **`gen-s-0094-denetle`**. Görev
"okunabilir dosyaları say" diyor ve bilerek mod-000 bir dosya yaratıyor,
ama ground truth Docker-as-root altında üretildiği için o dosyayı
okunabilir sayıp 7 yazmış. Singularity'de doğru cevap 6. Yani görevin
niyetine Singularity daha sadık; Docker'ın root'u testi etkisiz kılıyor.

**`task.yaml` dosyalarını elle DÜZELTME** — bu projenin kuralı: yalnızca
hat kodundaki hatalar düzeltilir, bozuk görevleri karantina mekanizması
ayıklar.

---

## 6. Zaman yiyen tuzaklar (tekrarlama)

1. **`module load`'u pipe'lama** (§3). İki kez yakalandı.
2. **`git bundle create x.bundle main`** yetmiyor — klon
   "remote HEAD refers to nonexistent ref" deyip boş çıkıyor.
   `git bundle create x.bundle HEAD main` yaz.
3. **`docker-archive://` sıkıştırılmış tar kabul etmiyor** —
   `.tar.gz` verirsen `invalid tar header`. Önce `gunzip -c`.
4. **Snapshot ve okunamayan dosyalar**: host tarafındaki `copytree` senin
   uid'inle koşuyor, mod-000 tek bir dosya bütün snapshot'ı `EACCES` ile
   düşürüyordu. `sandbox.py::_snapshot_kopyala` geçici okuma izni verip
   modları hem kaynakta hem hedefte geri koyuyor. Bu hata **yerel modda da
   vardı**, Colab'da root koşulduğu için görünmemişti.
5. SSH'ta "post-quantum key exchange" uyarısı çıkıyor — zararsız, gürültü.

---

## 7. Doğrulanmış koşu sonuçları

```bash
bash -l /gpfs/projects/etur22/turkish-agentic-rl/kos-sweep.sh          # envs
bash -l /gpfs/projects/etur22/turkish-agentic-rl/kos-korpus.sh envs_gen
```

| Korpus | Docker (dizüstü) | Singularity (MN5) | Süre |
|---|---|---|---|
| `envs` (8) | 8/8 | **8/8** | — |
| `envs_gen` (175) | 167/175 | **166/175** | 1m56s |
| `envs_gen_code5` (31) | — | **31/31** | 49s |

~1.6 sn/episode. 173×8'lik gerçek sweep'te konteyner ek yükü ~37 dk;
model üretimi baskın olduğu için GPFS metadata darboğazı gözlenmedi.

Docker'da da düşen 8 görev motordan bağımsız zaten bozuk. Tek gerçek
sapma `gen-s-0094-denetle` (§5).

---

## 8. Bütçe

`bsc_acct` çıktısındaki `khours` **core-hour**. ACC'de GPU saati 20 CPU
saati sayılıyor; tam node = 4 GPU × 20 = 80 = çekirdek sayısı.

**1 node-saat = 80 core-hour = 0.08 khours.**

10 Eylül 2026 itibarıyla: 4324.00 toplam, 2487.85 kullanılmış (%58),
**kalan 1836.15 khours ≈ 22.950 node-saat ≈ 91.800 H100-saat**.
Çıktıdaki %83 kullanıcı kotası değil, grubun harcaması içindeki pay.

Proje 2027-08-23'te bitiyor. Kalanı bitirmek için ~2.8 node'u 7/24
koşturmak gerek — yani **bütçe darboğaz değil, takvim darboğaz**.
EuroHPC erişim politikası çeyreklik az-kullanımı denetliyor ve Gantt'a
uymayan projenin kullanılmayan saatini başkasına dağıtabiliyor.

Referans: `EGITIM-ORTAMI.md:283` — 4×H100'de epoch ~1:20, yani 5 epoch'luk
tam koşu ~6:40, 10 saatlik tek job'a sığıyor (Colab'da 21 saatti).

---

## 9. Henüz çözülmemiş

1. **Eğitim yığını kurulmadı.** `EGITIM-ORTAMI.md:51`'deki hedef bileşim
   `torch 2.13.0+cu130, vllm 0.27.1, trl 1.12.0, transformers 5.16.1,
   liger_kernel 0.8.2` — ama bu Colab için. İnternet yok, `pip` çalışmıyor.
   **Önce MN5'in NVIDIA sürücü sürümü öğrenilmeli** (`srun` ile ACC node);
   cu130 daha yeni sürücü istiyor, eskiyse tüm yığın değişir.
   Sonra dizüstünde `pip download --only-binary=:all: --python-version 3.12`
   ile teker indirip taşımak gerekiyor (~8 GB).
2. **Model ağırlıkları yok.** `Qwen/Qwen3.5-4B` = 8.68 GiB, ne dizüstünde
   ne MN5'te. Proje dizininde `HF_CACHE` ve birkaç başka Qwen var ama o
   değil. Bu da dizüstüne indirilip taşınacak.
3. **`singularity.patch` commit'lenmedi.** MN5'teki `repo/` çalışma
   ağacında duruyor (7 dosya modifiye). Dizüstündeki repoda da
   commit'lenmemiş durumda.
4. **Tahsisat dönemsel mi**, 1836 khours tek havuz mu — `bsc_acct`'in alan
   tanımları yayınlanmamış, `man bsc_acct` bakılmalı. **Doğrulanmadı.**
