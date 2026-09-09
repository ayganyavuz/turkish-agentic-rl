"""Korpusu train / held-out olarak bol ve bolmeyi DONDUR.

    # bolmeyi olustur (bir kez)
    python -m verifiable_dataset.terminal.split --yaz

    # mevcut bolmeyi goster / dogrula
    python -m verifiable_dataset.terminal.split

Held-out'un tek isi epoch'lar arasi karsilastirmayi tasimak. Her kosuda
yeniden orneklenirse iki epoch'un sayilari ayni seyi olcmez, o yuzden
bolme dosyaya yazilip donduruluyor: --yaz mevcut dosyanin uzerine
yazmiyor, gercekten degistirmek isteyen --force demek zorunda.

Tabakalama (aile, hedef) ikilisi uzerinden: kod ailesinde 3, kabuk
ailesinde 14 hedef var ve bunlarin zorluklari birbirinden cok farkli.
Duz rastgele bolme, held-out'a orantisiz sayida `yarisma` dusurup
karsilastirmayi gurultuye bogabilirdi.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

VARSAYILAN_KAYNAKLAR = ("envs_gen", "envs_gen_code")
VARSAYILAN_DOSYA = "data/split.json"


def task_dizinleri(kaynaklar) -> list[Path]:
    out: list[Path] = []
    for kaynak in kaynaklar:
        for yol in sorted(Path(kaynak).glob("*/task.yaml")):
            if not yol.parent.name.startswith("_"):
                out.append(yol.parent)
    return out


def kunye(task_dir: Path) -> dict | None:
    """Bolme icin gereken asgari bilgi; bozuk yaml sessizce elenir."""
    try:
        d = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not d:
        return None
    seed = d.get("metadata", {}).get("seed", {}) or {}
    return {
        "dir": task_dir.as_posix(),
        "id": d.get("id", task_dir.name),
        "aile": seed.get("aile", "kabuk"),
        "hedef": seed.get("hedef", "?"),
        "promptlu": bool((d.get("prompt") or "").strip()),
        "checkli": bool(d.get("checks")),
    }


def basarisiz_referanslar(yollar) -> set[str]:
    """Sweep ciktilarindan referansi gecmeyen task id'lerini topla.

    Referansi gecmeyen bir gorev ne ogretir ne olcer: cozulemez oldugu
    icin her rollout 0 alir ve GRPO'da avantaj uretmez.
    """
    kotu: set[str] = set()
    for yol in yollar:
        p = Path(yol)
        if not p.exists():
            continue
        metin = p.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        for satir in metin.splitlines():
            if satir.startswith("Referansi gecmeyen"):
                _, _, liste = satir.partition(":")
                kotu |= {t.strip() for t in liste.split(",") if t.strip()}
    return kotu


def bol(kunyeler: list[dict], oran: float, tohum: int) -> tuple[list[dict], list[dict]]:
    """(aile, hedef) tabakalarinda oranli bolme."""
    rng = random.Random(tohum)
    tabakalar: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for k in kunyeler:
        tabakalar[(k["aile"], k["hedef"])].append(k)

    train: list[dict] = []
    holdout: list[dict] = []
    for _, grup in sorted(tabakalar.items()):
        grup = sorted(grup, key=lambda k: k["id"])
        rng.shuffle(grup)
        # Tek elemanli tabaka train'e gider: held-out'a koymak o hedefi
        # egitimden tamamen silmek olurdu.
        n_hold = int(round(len(grup) * (1 - oran)))
        if len(grup) > 1:
            n_hold = max(1, min(n_hold, len(grup) - 1))
        holdout.extend(grup[:n_hold])
        train.extend(grup[n_hold:])
    return (sorted(train, key=lambda k: k["id"]),
            sorted(holdout, key=lambda k: k["id"]))


def ozet(baslik: str, kayitlar: list[dict]) -> str:
    sayac: dict[str, int] = defaultdict(int)
    for k in kayitlar:
        sayac[k["aile"]] += 1
    dagilim = ", ".join(f"{a}={n}" for a, n in sorted(sayac.items()))
    return f"{baslik:10} {len(kayitlar):4}   ({dagilim})"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaynaklar", nargs="*", default=list(VARSAYILAN_KAYNAKLAR))
    ap.add_argument("--dosya", default=VARSAYILAN_DOSYA)
    ap.add_argument("--oran", type=float, default=0.7, help="train payi")
    ap.add_argument("--tohum", type=int, default=1234)
    ap.add_argument("--sweep", nargs="*",
                    default=["data/sweep-docker.txt", "data/sweep-docker-kod.txt"],
                    help="referansi gecmeyenleri elemek icin sweep ciktilari")
    ap.add_argument("--yaz", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="mevcut bolmenin uzerine yaz (karsilastirmayi gecersiz kilar)")
    args = ap.parse_args()

    dosya = Path(args.dosya)
    if dosya.exists() and not args.yaz:
        mevcut = json.loads(dosya.read_text(encoding="utf-8"))
        print(f"mevcut bolme: {dosya}  (tohum={mevcut.get('tohum')}, "
              f"olusturma={mevcut.get('olusturma')})")
        print(ozet("train", mevcut["train"]))
        print(ozet("holdout", mevcut["holdout"]))
        print(f"dislanan  {len(mevcut.get('dislanan', []))}")
        return 0

    hepsi = [k for k in (kunye(d) for d in task_dizinleri(args.kaynaklar)) if k]
    kotu = basarisiz_referanslar(args.sweep)

    dislanan: list[dict] = []
    uygun: list[dict] = []
    for k in hepsi:
        sebep = None
        if k["id"] in kotu:
            sebep = "referansi gecmiyor"
        elif not k["promptlu"]:
            sebep = "prompt yok"
        elif not k["checkli"]:
            sebep = "check yok"
        if sebep:
            dislanan.append({**k, "sebep": sebep})
        else:
            uygun.append(k)

    train, holdout = bol(uygun, args.oran, args.tohum)

    print(f"bulunan {len(hepsi)} task, uygun {len(uygun)}, dislanan {len(dislanan)}")
    for d in dislanan:
        print(f"  DISLANDI {d['id']:34} {d['sebep']}")
    print()
    print(ozet("train", train))
    print(ozet("holdout", holdout))

    if not args.yaz:
        print(f"\n(yazmak icin --yaz)")
        return 0
    if dosya.exists() and not args.force:
        print(f"\n{dosya} zaten var. Uzerine yazmak epoch'lar arasi "
              f"karsilastirmayi gecersiz kilar; gercekten istiyorsan --force ver.")
        return 1

    import datetime
    dosya.parent.mkdir(parents=True, exist_ok=True)
    dosya.write_text(json.dumps({
        "olusturma": datetime.date.today().isoformat(),
        "tohum": args.tohum,
        "oran": args.oran,
        "kaynaklar": args.kaynaklar,
        "train": train,
        "holdout": holdout,
        "dislanan": dislanan,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nyazildi -> {dosya}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
