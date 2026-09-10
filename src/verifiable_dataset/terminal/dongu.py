"""Epoch dongusu: her epoch basinda mufredati yeniden olc, egit, degerlendir.

    python -u -m verifiable_dataset.terminal.dongu --epoch 5 \
        --kalici /content/drive/MyDrive/turkish-agentic-rl

Her epoch su sirayla ilerliyor:

    1) SWEEP   -- o anki modelle train split'ine G=8 rollout
    2) MUFREDAT-- 1..7 cozen gorevler secilir (0/8 ve 8/8 disarida)
    3) EGITIM  -- yalnizca o listeyle 1 epoch
    4) EVAL    -- held-out, sweep ile ayni sekilde: N rollout, pass-rate
    5) checkpoint bir sonraki epoch'un modeli olur

Mufredat neden her epoch yeniden olculuyor: bir grubun butun rollout'lari
ayni sonucu verirse GRPO'da avantaj sifir olur, yani 0/8 ve 8/8 gorevler
gradyan uretmez. Model degistikce hangi gorevin hangi bantta oldugu da
degisiyor -- bir kez olcup sabitlemek, birkac epoch sonra compute'un
cogunu olu gorevlere harcamak demek.

Held-out neden sweep gibi olculuyor: tek rollout'la "cozdu mu" sorusu cok
gurultuluydu -- ayni taban model ayni 54 kabuk gorevinde bir kosuda 5,
digerinde 13 cozdu. Bu fark, epoch'lar arasinda aradigimiz farktan buyuk.
N rollout'un pass-rate ortalamasi ayni butceyle cok daha kararli.

Sweep, egitim ve eval ayri sureclerde kosuyor: uc asama da GPU'nun
tamamini istiyor ve ayni surecte sirayla yapmak vLLM ile egiticinin
bellek zirvelerini ust uste bindiriyor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BANT_ALT, BANT_UST = 1, 7      # 8 rollout'ta kac tanesi cozerse egitici sayilir


def kos(cmd: str, cwd: str | None = None, env: dict | None = None) -> int:
    print(f"  $ {cmd[:150]}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, env=env).returncode


def vllm_baslat(model: str, log: str, baglam: int = 16384,
                bekle: int = 1500) -> bool:
    """Modeli 'degerlendirme' adiyla servis et; ad sabit kalsin ki
    komut satirlari epoch'lar arasinda degismesin."""
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(3)
    subprocess.run("pkill -9 -f EngineCore", shell=True)
    time.sleep(10)
    subprocess.run(
        f"nohup vllm serve {model} --served-model-name degerlendirme "
        f"--port 8000 --max-model-len {baglam} --gpu-memory-utilization 0.85 "
        f"--enable-prefix-caching > {log} 2>&1 &", shell=True)
    t0 = time.time()
    while time.time() - t0 < bekle:
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
            print(f"  vllm hazir ({time.time() - t0:.0f} sn)", flush=True)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(10)
    print(f"  vllm ZAMAN ASIMI -- {log}", flush=True)
    return False


def vllm_durdur() -> None:
    # 'vllm serve' yalnizca API sunucusunu yakaliyor; KV cache'i tutan
    # EngineCore alt-sureci ayri bir adla kosuyor ve hayatta kalarak
    # kartin tamamini (olculdu: 68 GB) elinde tutuyor.
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(3)
    subprocess.run("pkill -9 -f EngineCore", shell=True)
    time.sleep(10)


def mufredat_yaz(bant_dosyalari: list[Path], hedef: Path) -> tuple[int, dict]:
    """Bant kayitlarindan 1..7 cozen gorevleri sec ve dosyaya yaz."""
    secilen: list[str] = []
    sayac = {"olu-zor": 0, "olu-kolay": 0, "bant": 0}
    for dosya in bant_dosyalari:
        if not dosya.exists():
            continue
        for satir in dosya.read_text(encoding="utf-8").splitlines():
            if not satir.strip():
                continue
            k = json.loads(satir)
            cozulen = k.get("solved", 0)
            if cozulen <= 0:
                sayac["olu-zor"] += 1
            elif cozulen >= k.get("rollouts", 8):
                sayac["olu-kolay"] += 1
            else:
                sayac["bant"] += 1
                secilen.append(k["task_dir"])
    hedef.parent.mkdir(parents=True, exist_ok=True)
    hedef.write_text("\n".join(sorted(secilen)) + "\n", encoding="utf-8")
    return len(secilen), sayac


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taban-model", default="Qwen/Qwen3.5-4B",
                    help="epoch 1'in baslangic modeli")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--kalici", default="/content/drive/MyDrive/turkish-agentic-rl",
                    help="checkpoint, olcum ve loglarin yazilacagi kalici dizin")
    ap.add_argument("--korpuslar", nargs="*", default=["envs_gen_code", "envs_gen"],
                    help="kod ve kabuk birlikte")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--rollouts", type=int, default=8,
                    help="hem mufredat sweep'inde hem held-out olcumunde")
    ap.add_argument("--sicaklik", type=float, default=1.0,
                    help="sweep ve eval ayni sicaklikta olmali; egitim "
                         "rollout'lariyla da ayni ki bant gercek zorlugu olssun")
    ap.add_argument("--prompt-batch", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--max-komut", type=int, default=12)
    ap.add_argument("--max-completion", type=int, default=6144)
    ap.add_argument("--vllm-baglam", type=int, default=16384)
    ap.add_argument("--vllm-bellek", type=float, default=0.50,
                    help="Colocate'te vLLM'e ayrilan kart orani. 0.45 ile "
                         "max-completion 6144'te ilk optimizer adiminda OOM "
                         "olduk: selective_log_softmax'in logits tensoru "
                         "5.68 GiB istedi, 1.29 GiB bostu. expandable_segments "
                         "o kosuda zaten aciktı, yani tek kalan kol bu.")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--protokol", choices=["native", "text", "auto"],
                    default="native",
                    help="Sweep/eval'in modelle konusma bicimi. EGITIM native "
                         "arac cagrisi kullaniyor (Qwen3.5'in chat template'i); "
                         "metin protokolu native tool calling'i olmayan modeller "
                         "icin yazilmis yedek yoldu. Ikisi farkli olunca bant "
                         "olcumu, egitimin icinde bulundugu dunyanin zorlugunu "
                         "olcmuyor.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--mufredati-kullan", action="store_true",
                    help="ilk epoch'ta sweep'i atla, diskteki "
                         "mufredat/epochN.txt'yi kullan. Yalnizca model o "
                         "olcumden beri degismediyse dogru -- yani kosu "
                         "egitim tamamlanmadan koptuysa.")
    ap.add_argument("--yanit-token", type=int, default=0,
                    help="sweep/eval'de yanit basina token butcesi. 0 = "
                         "--max-completion ile ayni (egitimle hizali). Bant "
                         "2048 ile olculuyordu, egitim 6144 veriyordu.")
    ap.add_argument("--profil", type=int, default=0, metavar="N",
                    help="egitimde ilk adimi isinma say, sonraki N adimi "
                         "torch.profiler ile olc")
    ap.add_argument("--tur-uykusu", action="store_true",
                    help="vLLM'i her turda uyut (eski davranis; geri donus)")
    ap.add_argument("--atla-baseline", action="store_true",
                    help="epoch 0 held-out olcumunu atla")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    K = Path(args.kalici)
    (K / "olcumler").mkdir(parents=True, exist_ok=True)
    (K / "mufredat").mkdir(parents=True, exist_ok=True)
    ortam = dict(os.environ, PYTHONPATH=str(repo / "src"),
                 TRL_EXPERIMENTAL_SILENCE="1",
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    def olc(model_yolu: str, bolum: str, etiket: str, rollouts: int,
            sicaklik: float) -> list[Path]:
        """Bir modeli bir split bolumunde olc; her korpus icin ayri jsonl."""
        ciktilar = []
        if not vllm_baslat(model_yolu, f"{K}/vllm-{etiket}.log", args.vllm_baglam):
            return ciktilar
        for korpus in args.korpuslar:
            cikti = K / "olcumler" / f"{bolum}-{etiket}-{korpus}.jsonl"
            kos(
                "python -u -m verifiable_dataset.terminal.pipeline "
                f"--asamalar bant --sandbox yerel --out {korpus} "
                "--model degerlendirme --base-url http://localhost:8000/v1 "
                f"--split {args.split} --bolum {bolum} "
                f"--rollouts {rollouts} --sicaklik {sicaklik} "
                f"--yanit-token {args.yanit_token or args.max_completion} "
                f"--concurrency {args.concurrency} --protocol {args.protokol} "
                '--reasoning-effort "" '
                f"--bant-out {cikti} "
                f"> {K}/log-{etiket}-{korpus}.txt 2>&1",
                cwd=str(repo), env=ortam)
            ciktilar.append(cikti)
        vllm_durdur()
        return ciktilar

    def ozet(dosyalar: list[Path], baslik: str) -> None:
        """Tek rollout'un 'cozdu mu'su cok gurultuluydu -- ayni model ayni
        gorevlerde 5/54 ve 13/54 verebiliyordu. N rollout'un pass-rate
        ortalamasi ayni butceyle cok daha kararli bir tahmin."""
        t_gorev = 0
        t_oran = 0.0
        for d in dosyalar:
            if not d.exists():
                continue
            kayitlar = [json.loads(s) for s in d.read_text(encoding="utf-8").splitlines()
                        if s.strip()]
            if not kayitlar:
                continue
            oran = sum(k.get("pass_rate", 0.0) for k in kayitlar) / len(kayitlar)
            hic = sum(1 for k in kayitlar if k.get("solved", 0) > 0)
            print(f"    {d.name:46} pass@{kayitlar[0].get('rollouts', '?')}="
                  f"{oran*100:5.1f}%   en az bir kez cozulen: {hic}/{len(kayitlar)}",
                  flush=True)
            t_gorev += len(kayitlar)
            t_oran += oran * len(kayitlar)
        if t_gorev:
            print(f"    {baslik}: ortalama pass-rate = {t_oran/t_gorev*100:.1f}% "
                  f"({t_gorev} gorev)", flush=True)

    model = args.taban_model
    if not args.atla_baseline:
        print("\n" + "=" * 70 + "\nEPOCH 0 -- BASELINE (held-out)\n" + "=" * 70, flush=True)
        ozet(olc(model, "holdout", "epoch0", args.rollouts, args.sicaklik),
             "baseline held-out")

    for epoch in range(1, args.epoch + 1):
        print("\n" + "=" * 70 + f"\nEPOCH {epoch}\n" + "=" * 70, flush=True)

        # 1) mufredat olcumu -- o anki modelle, egitim bolumunde
        liste = K / "mufredat" / f"epoch{epoch}.txt"
        if args.mufredati_kullan and liste.exists():
            # Kosu yarida kaldiginda sweep'i tekrarlamak ~40 dk yakiyor.
            # Model o olcumden beri degismediyse (egitim hic tamamlanmadiysa)
            # olcum hala gecerli. Bir kez kullaniliyor; sonraki epoch'lar
            # yeniden olcuyor cunku model artik degismis oluyor.
            n = len([x for x in liste.read_text(encoding="utf-8").splitlines()
                     if x.strip()])
            print(f"\n[{epoch}] SWEEP ATLANDI -- mevcut mufredat: {liste} "
                  f"({n} gorev)", flush=True)
            args.mufredati_kullan = False
        else:
            print(f"\n[{epoch}] SWEEP (train, G={args.rollouts})", flush=True)
            bantlar = olc(model, "train", f"sweep{epoch}", args.rollouts, args.sicaklik)
            n, sayac = mufredat_yaz(bantlar, liste)
            print(f"\n[{epoch}] MUFREDAT: {n} gorev  "
                  f"(bant={sayac['bant']}, olu-zor={sayac['olu-zor']}, "
                  f"olu-kolay={sayac['olu-kolay']})", flush=True)
        if n == 0:
            print("bantta gorev kalmadi -- dongu duruyor", flush=True)
            return 1

        # 2) egitim
        cikti = K / "ciktilar" / f"epoch{epoch}"
        print(f"\n[{epoch}] EGITIM -> {cikti}", flush=True)
        rc = kos(
            "python -u -m verifiable_dataset.terminal.train_grpo "
            f"--model {model} --gorevler {liste} --sandbox yerel "
            f"--rollouts {args.rollouts} --prompt-batch {args.prompt_batch} "
            f"--micro-batch {args.micro_batch} --epoch 1 --lr {args.lr} "
            f"--max-komut {args.max_komut} --max-completion {args.max_completion} "
            f"--vllm-baglam {args.vllm_baglam} --vllm-bellek {args.vllm_bellek} "
            f"--dtype {args.dtype} "
            + (f"--profil {args.profil} " if args.profil else "")
            + ("--tur-uykusu " if args.tur_uykusu else "")
            + f"--cikti {cikti} > {K}/train-epoch{epoch}.log 2>&1",
            cwd=str(repo), env=ortam)
        if rc != 0 or not (cikti / "config.json").exists():
            print(f"egitim basarisiz (rc={rc}) -- {K}/train-epoch{epoch}.log", flush=True)
            return 1

        # 3) held-out degerlendirmesi
        print(f"\n[{epoch}] EVAL (held-out)", flush=True)
        ozet(olc(str(cikti), "holdout", f"epoch{epoch}", args.rollouts, args.sicaklik),
             f"epoch{epoch} held-out")

        model = str(cikti)   # sonraki epoch buradan devam eder

    print("\nDONGU BITTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
