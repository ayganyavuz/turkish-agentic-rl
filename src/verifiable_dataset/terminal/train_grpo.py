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


def veri_kumesi(split_yolu: Path, bolum: str, aile: str, repo_kok: Path):
    """split.json'dan HF Dataset kur.

    `task_dir` satirla birlikte reset()'e gecirilecek; TRL satirin tamamini
    forward ediyor.
    """
    from datasets import Dataset
    import yaml

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


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--aile", default="kod", choices=["kod", "kabuk", "hepsi"])
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
    ap.add_argument("--vllm-bellek", type=float, default=0.45)
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
    egitim = veri_kumesi(split_yolu, "train", args.aile, repo_kok)
    print(f"egitim kumesi: {len(egitim)} gorev  (aile={args.aile})")

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
        gradient_checkpointing=True,
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

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=egitim,
        environment_factory=lambda: TerminalOrtami(max_komut=args.max_komut),
    )
    trainer.train()
    trainer.save_model(args.cikti)
    print(f"bitti -> {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
