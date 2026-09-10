"""GRPO egitimi: TRL'in environment_factory'si + bizim sandbox'imiz.

    python -m verifiable_dataset.terminal.train_grpo \
        --model Qwen/Qwen3.5-4B --aile kod --rollouts 8

TRL 1.12 cok turlu arac kullanimini kendisi yurutuyor: ortam metotlari
arac olarak sunuluyor, arac ciktisi token'lari loss'tan maskeleniyor
(`tool_mask`), vLLM colocate modunda agirlik senkronu otomatik. Bu yuzden
burada yalnizca UC sey tanimlaniyor:

    reset()       -- gorevin baslangic dunyasini kur
    run_command() -- ajanin tek araci
    get_reward()  -- check'leri kosturup ikili odulu ver

Odul bilerek ikili: check'lerin hepsi gectiyse 1, aksi halde 0. Kismi oran
yalnizca teshis icin loglaniyor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Colab'de Docker yok; sandbox modunu her seyden once sec.
from verifiable_dataset.terminal import sandbox as _sandbox


class TerminalOrtami:
    """Bir episode'luk terminal dunyasi.

    TRL her rollout icin ayri bir ornek yaratiyor, yani iki rollout'un
    dosya sistemi birbirine karismiyor. Sandbox reset'te aciliyor ve
    get_reward'da kapaniyor -- odul hesaplandiktan sonra dunyaya ihtiyac
    kalmiyor.
    """

    def __init__(self, max_komut: int = 12):
        self.max_komut = max_komut
        self._sandbox = None
        self._task = None
        self._komut_sayisi = 0
        self._son_odul = 0.0
        self._son_kismi = 0.0

    # -- TRL sozlesmesi ------------------------------------------------

    def reset(self, task_dir: str = "", **_yoksay) -> None:
        """Gorevin baslangic dunyasini kur. Prompt zaten veri kumesinde."""
        from verifiable_dataset.terminal.task import Task

        self._kapat()
        self._komut_sayisi = 0
        self._son_odul = 0.0
        self._son_kismi = 0.0
        self._task = Task.load(task_dir)
        self._sandbox = self._task.make_sandbox()
        self._sandbox.start()
        try:
            self._task.prepare(self._sandbox)
        except Exception:  # noqa: BLE001 - bozuk setup episode'u dusurur, kosuyu degil
            self._kapat()
        return None

    def run_command(self, command: str) -> str:
        """Sandbox terminalinde tek bir shell komutu calistirir.

        Args:
            command: Calistirilacak shell komutu, ornegin: ls -la
        """
        if self._sandbox is None:
            return "sandbox hazir degil"
        # Tur butcesi ortamda tutuluyor: model sonsuza kadar komut
        # deneyemesin, yoksa tek bir rollout butun batch'i bekletir.
        if self._komut_sayisi >= self.max_komut:
            return f"komut butcesi doldu ({self.max_komut}); gorevi ozetle ve bitir"
        self._komut_sayisi += 1
        try:
            sonuc = self._sandbox.exec(command, timeout=30)
        except Exception as e:  # noqa: BLE001
            return f"komut calistirilamadi: {type(e).__name__}: {e}"
        return sonuc.to_observation()

    def get_reward(self) -> float:
        """Check'leri snapshot uzerinde kostur, ikili odulu dondur."""
        from verifiable_dataset.terminal.task import grade

        if self._sandbox is None or self._task is None:
            return 0.0
        try:
            rapor = grade(self._task, self._sandbox)
            self._son_odul = float(rapor.reward)
            self._son_kismi = float(rapor.partial)
        except Exception:  # noqa: BLE001 - notlandirma cokerse odul 0
            self._son_odul = 0.0
        finally:
            self._kapat()
        return self._son_odul

    # -- ic ------------------------------------------------------------

    def _kapat(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.stop()
            except Exception:  # noqa: BLE001
                pass
            self._sandbox = None


SISTEM = (
    "Sen bir Linux terminalinde calisan bir yardimcisin. Verilen gorevi "
    "tamamlamak icin run_command aracini kullanarak shell komutlari calistir. "
    "Her adimda komutun ciktisini gorursun. Gorev bitince arac cagirmayi "
    "birak ve kisaca ozetle."
)


def veri_kumesi(split_yolu: Path, bolum: str, aile: str, repo_kok: Path,
                gorev_listesi: str = ""):
    """Egitim kumesini kur.

    `gorev_listesi` verilirse split yerine o dosyadaki task dizinleri
    kullanilir -- mufredat her epoch basinda yeniden olculdugu icin
    egitilecek gorevler dosyadan geliyor.

    `task_dir` satirla birlikte reset()'e gecirilecek; TRL satirin tamamini
    forward ediyor.
    """
    from datasets import Dataset
    import yaml

    if gorev_listesi:
        yollar = [l.strip() for l in Path(gorev_listesi).read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")]
        satirlar = []
        for yol in yollar:
            d = Path(yol)
            if not d.is_absolute():
                d = repo_kok / yol
            spec = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8"))
            satirlar.append({
                "prompt": [{"role": "system", "content": SISTEM},
                           {"role": "user", "content": spec["prompt"]}],
                "task_dir": str(d),
                "task_id": spec.get("id", d.name),
            })
        if not satirlar:
            raise SystemExit(f"{gorev_listesi} bos")
        return Dataset.from_list(satirlar)

    split = json.loads(split_yolu.read_text(encoding="utf-8"))
    satirlar = []
    for k in split[bolum]:
        if aile != "hepsi" and k["aile"] != aile:
            continue
        d = repo_kok / k["dir"]
        spec = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8"))
        satirlar.append({
            "prompt": [{"role": "system", "content": SISTEM},
                       {"role": "user", "content": spec["prompt"]}],
            "task_dir": str(d),
            "task_id": k["id"],
        })
    if not satirlar:
        raise SystemExit(f"{bolum}/{aile} icin gorev bulunamadi")
    return Dataset.from_list(satirlar)


class ProfilCallback:
    """Loss fazini ayristirmak icin torch profiler.

    Olculen: loss fazi ~314 sn. Teorik olarak govdenin fwd+bwd'si
    ~1400 TFLOP ve A100'de gercekci 150 TFLOPS ile ~10 sn eder -- yani
    30 kat acik var. `lm_head` + log-softmax bu isin yalnizca %6'si
    (88 TFLOP), attention da sebep degil (Qwen3.5 hibrit: katmanlarin
    cogu linear_attention, full_attention katmanlari da sdpa ile
    matrisi materyalize etmiyor). Kalan supheliler gradient
    checkpointing'in yeniden hesabi ve mikro-batch 2'de kartin
    dolmamasi. Bunlar tahmin; bu callback olcuyor.

    Ilk adim isinma (torch.compile, CUDA graph, allocator) oldugu icin
    atlanir. Cikti hem log'a tablo olarak basilir hem de chrome trace
    olarak yazilir.
    """

    def __init__(self, adim_sayisi: int, cikti_dizini: str):
        self.adim_sayisi = adim_sayisi
        self.cikti = Path(cikti_dizini)
        self.cikti.mkdir(parents=True, exist_ok=True)
        self._prof = None
        self._gorulen = 0
        self._bitti = False

    def _hazir_olunca(self, prof):
        import torch
        print("\n" + "=" * 78, flush=True)
        print("PROFIL -- CUDA suresine gore ilk 25 islem", flush=True)
        print("=" * 78, flush=True)
        try:
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=25), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"profil tablosu basilamadi: {e}", flush=True)
        yol = self.cikti / "profil-trace.json"
        try:
            prof.export_chrome_trace(str(yol))
            print(f"chrome trace -> {yol}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"trace yazilamadi: {e}", flush=True)

    def on_step_begin(self, args, state, control, **kw):
        import torch
        if self._bitti or self._prof is not None or state.global_step < 1:
            return
        # 1. adim isinma; 2. adimdan itibaren olc
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        )
        self._prof.__enter__()
        print(f"profil basladi (adim {state.global_step})", flush=True)

    def on_step_end(self, args, state, control, **kw):
        if self._prof is None:
            return
        self._gorulen += 1
        if self._gorulen >= self.adim_sayisi:
            self._prof.__exit__(None, None, None)
            self._hazir_olunca(self._prof)
            self._prof = None
            self._bitti = True       # tek sefer olc, bir daha baslama
            # Profil kosusu bir olcum kosusudur, egitim kosusu degil:
            # tablo basildiktan sonra kalan adimlar GPU'yu bosuna yakar.
            control.should_training_stop = True
            print("profil bitti -- egitim durduruluyor", flush=True)

    # TrainerCallback arayuzunun kullanmadigimiz kancalari
    def __getattr__(self, ad):
        if ad.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(ad)


def episode_boyu_uyanik_tut() -> bool:
    """vLLM'i tur basina degil, adim basina uyut.

    TRL'in cok turlu dongusu `_tool_call_loop` icinde her tur icin
    `vllm_generation.generate()` cagiriyor ve o fonksiyon sonunda
    `llm.sleep(level=2)` var. Level 2 KV cache'i ATIYOR, yani prefix cache
    turlar arasinda yasayamiyor: 9 turun her biri o ana kadarki butun
    konusmayi sifirdan prefill ediyor ve maliyet tur sayisinin karesiyle
    buyuyor. Bellek izinde adim basina 10-14 uyku/uyanma salinimi olarak
    gorunuyordu (21.7 <-> 59.9 GB).

    Bu yama `_generate`'i sarmaliyor: tur dongusu boyunca uyku kapali,
    dongu bitince bir kez `sleep(level=2)`. Boylece adim sinirlarindaki
    davranis aynen korunuyor -- loss fazinda vLLM yine uykuda, yani
    olculen 64.5 GB'lik tepe degismiyor.

    Sarmalamadan ONCE agirliklar uyandiriliyor: level 2 agirliklari da
    attigi icin, uyku bayragi kapaliyken `sync_weights` serbest birakilmis
    bellege `load_weights` yapardi (TRL'in kendi yorumu bunun bazi
    backend'lerde coktugunu soyluyor).

    Yamanin tuttugunu dondurur; tutmazsa egitim yine kosar, sadece eski
    davranisla.
    """
    try:
        from trl import GRPOTrainer
    except Exception:
        return False
    if not hasattr(GRPOTrainer, "_generate"):
        return False
    if getattr(GRPOTrainer, "_uyku_yamasi", False):
        return True

    _asil_generate = GRPOTrainer._generate

    def _generate_tek_uyku(self, prompts):
        gen = getattr(self, "vllm_generation", None)
        if gen is None or not getattr(gen, "enable_sleep_mode", False):
            return _asil_generate(self, prompts)

        llm = getattr(gen, "llm", None)
        if llm is None:
            return _asil_generate(self, prompts)

        # 1) agirliklari ve KV'yi geri getir (level 2 ikisini de atmisti)
        try:
            if getattr(gen, "_llm_weights_sleeping", False):
                llm.wake_up(tags=["weights"])
                gen._llm_weights_sleeping = False
            llm.wake_up(tags=["kv_cache"])
        except Exception:
            return _asil_generate(self, prompts)

        # 2) tur dongusu boyunca uyutma
        gen.enable_sleep_mode = False
        try:
            return _asil_generate(self, prompts)
        finally:
            gen.enable_sleep_mode = True
            try:
                llm.sleep(level=2)
                gen._llm_weights_sleeping = True
            except Exception:
                pass

    GRPOTrainer._generate = _generate_tek_uyku
    GRPOTrainer._uyku_yamasi = True
    return True


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--aile", default="hepsi", choices=["kod", "kabuk", "hepsi"])
    ap.add_argument("--gorevler", default="",
                    help="egitilecek task dizinlerini iceren dosya (satir basina bir yol). "
                         "Verilirse split yerine bu kullanilir -- mufredat her epoch "
                         "basinda yeniden olculdugu icin.")
    ap.add_argument("--cikti", default="/content/ciktilar/grpo")
    ap.add_argument("--rollouts", type=int, default=8, help="G -- grup basina rollout")
    ap.add_argument("--prompt-batch", type=int, default=4,
                    help="iterasyon basina prompt sayisi (x rollouts = episode)")
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="ayni anda ileri gecen episode sayisi. Logit tensoru "
                         "batch x uzunluk x kelime_dagarcigi kadar yer kapliyor; "
                         "8 episode x 2048 token x 152k kelime A100'u tasiriyor.")
    ap.add_argument("--epoch", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL katsayisi")
    ap.add_argument("--max-komut", type=int, default=12)
    ap.add_argument("--max-completion", type=int, default=6144,
                    help="Bir episode'un TAMAMI (butun turlar + arac ciktilari) "
                         "bu butceye sigmali. 2048 ile episode'larin %40'i "
                         "kesiliyordu ve kesilen episode odul 0 aliyordu: "
                         "korelasyon(odul, kesilme) = -0.51. Yani sinir bir "
                         "butce degil ortuk bir uzunluk cezasi haline geliyordu.")
    ap.add_argument("--sandbox", choices=["docker", "yerel"], default="yerel")
    ap.add_argument("--vllm-bellek", type=float, default=0.50,
                    help="Colocate'te vLLM'e ayrilan kart orani. Kisit uyku "
                         "fazinda degil URETIM fazinda: vLLM uyanikken egiticinin "
                         "sabit kismi (bf16'da ~17 GB) da kartta duruyor. Ustelik "
                         "daha buyuk KV cache bir yerden sonra bos duruyor -- bir "
                         "adimda 32 episode x ~10k token ~ 320k token uretiliyor.")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention cekirdegi. Varsayilan sdpa: flash-attn "
                         "kurulu olmadigi icin transformers zaten buna "
                         "dusuyordu. flash_attention_2 YALNIZCA egitim "
                         "(loss) fazini etkiler -- uretim vLLM'de kosuyor ve "
                         "o kendi backend'ini seciyor. Qwen3.5 hibrit: "
                         "katmanlarin bir kismi linear_attention, FA2 "
                         "yalnizca full_attention katmanlarinda devreye girer.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="model agirliklarinin dtype'i. TRL'nin varsayilani "
                         "float32 ve bu 4B model icin ~20 GB fazladan yer demek.")
    ap.add_argument("--profil", type=int, default=0, metavar="N",
                    help="ilk adimi isinma sayip sonraki N adimi torch "
                         "profiler ile olc; tabloyu log'a basar ve chrome "
                         "trace'i --cikti altina yazar. 0 = kapali.")
    ap.add_argument("--tek-uyku", action="store_true", default=True,
                    help="vLLM'i tur basina degil adim basina uyut; tur "
                         "basina sleep(level=2) KV cache'i atip her turda "
                         "bastan prefill'e zorluyor")
    ap.add_argument("--tur-uykusu", dest="tek_uyku", action="store_false",
                    help="TRL'in varsayilan davranisi (her tur uyu/uyan)")
    ap.add_argument("--liger", action="store_true", default=True,
                    help="liger fuzyonlu cekirdekler (qwen3_5 destekleniyor)")
    ap.add_argument("--liger-kapali", dest="liger", action="store_false")
    ap.add_argument("--vllm-uyku", action="store_true", default=True,
                    help="uretim disinda vLLM bellegi biraksin (colocate'te sart)")
    ap.add_argument("--vllm-baglam", type=int, default=16384,
                    help="vLLM baglam siniri -- KV cache bunun kadar yer kapliyor. "
                         "Prompt + completion buraya sigmali, yani "
                         "--max-completion'dan belirgin buyuk olmali.")
    ap.add_argument("--wandb", default="", help="wandb proje adi (bos = kapali)")
    args = ap.parse_args()

    if args.sandbox == "yerel":
        _sandbox.yerel_kullan(True)
        ok, bilgi = _sandbox.bash_var()
        if not ok:
            print(f"yerel sandbox icin bash gerekli: {bilgi}")
            return 1
    else:
        ok, bilgi = _sandbox.docker_available()
        if not ok:
            print(f"Docker daemon'a ulasilamiyor: {bilgi}")
            return 1
    print(f"sandbox: {args.sandbox} ({bilgi})")

    # Kesilme sessiz bir hata: episode yarim kalir, odul 0 gelir ve gradyan
    # "kisa uret" der. Baglam completion'a esit ya da yakinsa prompt'a yer
    # kalmaz ve bu her episode'da olur.
    if args.vllm_baglam < args.max_completion + 2048:
        print(f"UYARI: --vllm-baglam ({args.vllm_baglam}) "
              f"--max-completion ({args.max_completion}) icin dar; "
              f"prompt'a yer birakmiyor. En az {args.max_completion + 2048} onerilir.")

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb)

    from trl import GRPOConfig, GRPOTrainer

    split_yolu = Path(args.split)
    repo_kok = split_yolu.resolve().parent.parent
    egitim = veri_kumesi(split_yolu, "train", args.aile, repo_kok, args.gorevler)
    kaynak = args.gorevler or f"split/{args.aile}"
    print(f"egitim kumesi: {len(egitim)} gorev  (kaynak={kaynak})")

    cfg = GRPOConfig(
        output_dir=args.cikti,
        num_generations=args.rollouts,
        # Bir iterasyonda B prompt x G rollout episode uretiliyor. Optimizer
        # adimi basina episode sayisi sabit kalsin diye biriktirme mikro
        # batch'e gore hesaplaniyor.
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=max(
            1, (args.rollouts * args.prompt_batch) // args.micro_batch),
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=1.0,
        max_completion_length=args.max_completion,
        bf16=True,
        # TRL, model_init_kwargs'ta dtype verilmezse modeli FLOAT32 yukluyor
        # (from_pretrained'den farkli; grpo_trainer.py docstring'i soyluyor).
        # bf16=True yalnizca autocast, yani hesabi bf16 yapiyor ama agirliklar
        # fp32 kaliyordu. Olculdu: agirlik platosu 17.3 GB = 4.66e9 x 4 bayt.
        # bf16'ya gecince agirlik ve gradyan yariya iniyor, logits de fp32
        # yerine bf16 uretiliyor -- selective_log_softmax'in fp32 dalindaki
        # satir basina 5.68 GiB'lik logsumexp geçicisi de yariya iniyor.
        model_init_kwargs={"dtype": args.dtype,
                           "attn_implementation": args.attn},
        gradient_checkpointing=True,
        # RMSNorm/SwiGLU/RoPE'u fuzyonlu cekirdeklerle degistiriyor; aktivasyon
        # tepesini dusuruyor. Not: GRPO logits'i kendi hesapladigi icin liger'in
        # fuzyonlu linear-cross-entropy'si BU yolda devreye girmiyor -- yani
        # logits tensorunu ortadan kaldirmiyor, sadece aktivasyondan kazandiriyor.
        use_liger_kernel=args.liger,
        optim="adamw_bnb_8bit",       # 4B full FT + vLLM ayni karta ancak boyle sigar
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_bellek,
        # Colocate'te egitici (~40 GB) ile vLLM'in zirveleri ust uste
        # binince 80 GB yetmiyor. Uyku modunda vLLM uretim disinda bellegi
        # birakiyor, yani iki zirve ayni anda olusmuyor.
        vllm_enable_sleep_mode=args.vllm_uyku,
        vllm_max_model_length=args.vllm_baglam,
        logging_steps=1,
        save_steps=20,
        report_to=["wandb"] if args.wandb else [],
        log_completions=True,
    )

    if args.tek_uyku:
        tuttu = episode_boyu_uyanik_tut()
        print(f"vLLM uyku yamasi: {'AKTIF' if tuttu else 'TUTMADI (eski davranis)'}",
              flush=True)

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=egitim,
        environment_factory=lambda: TerminalOrtami(max_komut=args.max_komut),
    )
    if args.profil:
        trainer.add_callback(ProfilCallback(args.profil, args.cikti))
        print(f"profil acik: 1. adim isinma, sonraki {args.profil} adim olculecek",
              flush=True)

    trainer.train()
    trainer.save_model(args.cikti)
    print(f"bitti -> {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
