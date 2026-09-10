# Eğitim Ortamı ajanı — devir promptu

Aşağıdaki metni yeni agent'a olduğu gibi ver.

---

Türkçe doğrulanabilir RL veri seti projesinin **eğitim ortamı** ayağını
devralıyorsun. Konuşma dili Türkçe.

## ÖNCE OKU

`EGITIM-ORTAMI.md` — sandbox, GRPO, bellek, ölçüm, açık kusurlar. Hepsi
ölçümle gerekçelendirilmiş, tahminle değil.

Yan dosyalar: `VERI-URETIMI.md` (korpus üretimi, ayrı bir agent'ta yürüyor —
sen dokunma), `PROJECT-TRACE.md` (Bölüm 0-8 tarihçe arşivi; yeni bölüm ekleme,
ikiye ayrıldı).

Repo: `C:\Users\PC_2849_YD26\dev\turkish-agentic-rl` · son commit `f93ccf2`
Colab: MCP bağlı, notebook 8 hücreye indirildi (kurulum → başlat → izle).

## ALTIN KURAL

**Yeni bir bayrak/ayar eklediğinde, koşuyu başlatmadan ÖNCE gerçekten etki
ettiğini test et.** Bu projede bu sınıfta beş hata çıktı, her biri saatler
yaktı. Listesi `EGITIM-ORTAMI.md`'nin başında. Test şekli: `kos` /
`vllm_baslat` stub'lanıp komut yakalanır, ya da sahte `GRPOConfig` ile alanlar
yakalanır — repoda çalışır örnekler var.

Aynı disiplin ölçüme de geçerli: bir sayı hakkında konuşmadan önce nereden
geldiğini doğrula. Bugün "loss fazı adımın küçük kısmı" diye yanlış bir atıf
yapıldı ve bir öneri onun üstüne kuruldu; bellek izine bakılınca tersi çıktı.

## DURUM

Koşu **durduruldu** (epoch 1'in 8/25 adımı tamamlanmıştı, `kosu4-tekuyku/`
altında arşivlendi). GPU temizliği yarım kaldı olabilir — `EngineCore`
süreçlerini kontrol et, `pkill -f 'vllm serve'` onları yakalamıyor.

Son iki değişiklik commit'li ama **hiç koşulmadı**:
1. `--protokol` bayrağı, varsayılan `native` (`f93ccf2`) — sweep artık
   eğitimin kullandığı arayüzle ölçecek.
2. `--profil N` bayrağı, `ProfilCallback` — ilk adımı ısınma sayar, sonraki N
   adımı `torch.profiler` ile ölçer. **Bu commit'lenmedi**, çalışma ağacında
   duruyor; testi geçti (adım 0 ısınma, adım 1-2 ölçüldü, bir daha başlamadı).

## ÖNÜNDEKİ İKİ İŞ

**1. Loss fazını profille.** Adım ~502 sn, bunun ~%62'si loss fazı ve teorik
süresinin **30 katı** (gövde fwd+bwd ~1400 TFLOP, A100'de ~10 sn beklenir).
Elenen şüpheliler: attention (Qwen3.5 hibrit, `sdpa` matrisi materyalize
etmiyor), `lm_head`+log-softmax (işin sadece %6'sı), `selective_log_softmax`
içindeki Python döngüsü (batch satırı üzerinde, iki iterasyon). Kalanlar:
gradient checkpointing'in yeniden hesabı, mikro-batch 2'de kartın dolmaması,
ve faz atfının yanlış olma ihtimali. `--profil 2` ile koş, tabloyu oku.

**2. Bandı native protokolle yeniden ölç.** Mevcut müfredat `--protocol text`
ile ölçüldü — Gemma için yazılmış yedek yol. Eğitim native tool-calling
kullanıyor. Bu, `reward=0.72` ile "8'de 1-6 bandı" arasındaki çelişkinin en
güçlü açıklaması. Sweep'i native ile koş, `frac_reward_zero_std`'nin
düşüp düşmediğine bak.

Hizalamanın iki parçası **daha yapılmadı**: `VDS_MAX_TOKENS` hâlâ 2048
(eğitim 6144 veriyor), ve tur sayısı düz 12 (görevlerin 199'unda
`task.max_turns` farklı). İkincisi `TerminalOrtami`'nin `task_dir`'den kendi
okumasını gerektiriyor; TRL'in `environment_factory`'si göreve göre parametre
almıyor.

## KARAR VERİRKEN BİL

- **5 epoch tek Colab oturumuna sığmıyor** (~21 saat). Her epoch ayrı oturum:
  checkpoint Drive'da, sonraki oturum `--taban-model <ciktilar/epochN>` ile
  devam eder ve `--mufredati-kullan` **kaldırılmalıdır**.
- **Mikro-batch 4 önerilmiyor**: geçiş sayısını yarıya indirir ama toplam
  hesabı değiştirmez (~%6-15), buna karşılık duvara payı 14.75 → ~7 GB'a
  düşürür.
- **`--vllm-bellek`'i kısmak loss tepesine yaramaz** — vLLM o sırada uykuda ve
  18 GB bırakıyor (ölçüldü). Kısıt üretim fazında.
- **`frac_reward_zero_std` bu batch boyutunda kıyaslanamaz**: adım başına 4
  problem, yani sadece 0/0.25/0.5/0.75/1 değerlerini alabiliyor.
- **SWE-Bench hiç yapılmadı** (Faz 0.2 sb-cli testi, Faz 4 scaffold). Ana
  metrik held-out; SWE-bench ikincil.

## ÇALIŞMA ŞEKLİ

- `EGITIM-ORTAMI.md`'ye **append** et, geçmişi yeniden yazma.
- Ölçtüğün her sayının nereden geldiğini yaz; tahminse tahmin olduğunu söyle.
- Geçersiz çıkan ölçümleri silme, `kirli-olcumler/` altına taşı ve neden
  geçersiz olduğunu kaydet.
- Uzun koşuları `nohup` ile başlat, MCP'den bağımsız olsun.
- Commit mesajlarını İngilizce yaz, konuşmayı Türkçe sürdür.
