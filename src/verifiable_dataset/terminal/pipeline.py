"""Uctan uca uretim hatti: spec -> prompt -> kapilar -> bant.

    # her sey tek uctan, tek kosuda
    python -u -m verifiable_dataset.terminal.pipeline --n 150 --id-basla 110 \
        --rng 2 --concurrency 8 --rollouts 8

    # yalnizca yarim kalmis task'larin prompt'unu yaz ve kapilardan gecir
    python -u -m verifiable_dataset.terminal.pipeline --asamalar prompt,kapi

Asamalar ayri ayri kosturulabiliyordu ama uc, anahtar ve model her
komutta yeniden cozumleniyor; biri digerinden farkli bir uca giderse
korpus sessizce karisiyordu. Burada istemci bir kez kuruluyor ve butun
asamalar ondan geciyor.

Hat yeniden calistirilabilir: her asama kendi isini bitirmis task'lari
atlar, yani kopan bir kosu bastan baslatilabilir.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from verifiable_dataset.terminal.gates import run_gauntlet
from verifiable_dataset.terminal.generate import generate_one
from verifiable_dataset.terminal.llm import make_client, preflight, resolve_model
from verifiable_dataset.terminal.prompts import prompt_yaz, yaz_bir
from verifiable_dataset.terminal.runner import run_model
from verifiable_dataset.terminal.sandbox import sandbox_hazirla
from verifiable_dataset.terminal.seeds import sample, sample_kod
from verifiable_dataset.terminal.sweep import summarize
from verifiable_dataset.terminal.task import Task

ASAMALAR = ("uret", "prompt", "kapi", "bant")


# -- token sayaci ----------------------------------------------------
# Butce bu projede gercek bir kisit ve panel geriye donuk olarak hangi
# asamanin ne kadar yedigini soylemiyor. Sayac asama basina kiriyor.

class _SayanCompletions:
    def __init__(self, inner, sayac):
        self._inner = inner
        self._sayac = sayac

    def create(self, **kwargs):
        yanit = self._inner.create(**kwargs)
        self._sayac.ekle(getattr(yanit, "usage", None))
        return yanit


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _SayanClient:
    """Istemciyi sarar; chat.completions.create disindaki her sey aynen gecer."""

    def __init__(self, inner, sayac):
        self._inner = inner
        self.chat = _Chat(_SayanCompletions(inner.chat.completions, sayac))

    def __getattr__(self, ad):
        return getattr(self._inner, ad)


class TokenSayaci:
    """Asama basina istek ve token toplami."""

    def __init__(self):
        self._kilit = threading.Lock()
        self.asama = "-"
        self.veri: dict[str, dict[str, int]] = {}

    def ekle(self, usage) -> None:
        giris = getattr(usage, "prompt_tokens", 0) or 0
        cikis = getattr(usage, "completion_tokens", 0) or 0
        with self._kilit:
            d = self.veri.setdefault(self.asama, {"istek": 0, "giris": 0, "cikis": 0})
            d["istek"] += 1
            d["giris"] += giris
            d["cikis"] += cikis

    def rapor(self, fiyat_giris: float, fiyat_cikis: float) -> str:
        satirlar = ["", "TOKEN KULLANIMI"]
        t_istek = t_giris = t_cikis = 0
        for ad in ASAMALAR:
            d = self.veri.get(ad)
            if not d:
                continue
            t_istek += d["istek"]
            t_giris += d["giris"]
            t_cikis += d["cikis"]
            satirlar.append(f"  {ad:8} istek={d['istek']:5}  "
                            f"giris={d['giris']:9}  cikis={d['cikis']:9}")
        satirlar.append(f"  {'TOPLAM':8} istek={t_istek:5}  "
                        f"giris={t_giris:9}  cikis={t_cikis:9}")
        if fiyat_giris or fiyat_cikis:
            usd = t_giris / 1e6 * fiyat_giris + t_cikis / 1e6 * fiyat_cikis
            satirlar.append(f"  tahmini maliyet: ${usd:.3f}")
        else:
            satirlar.append("  (maliyet icin --fiyat-giris / --fiyat-cikis ver, $/1M token)")
        return "\n".join(satirlar)


# -- asamalar --------------------------------------------------------

@dataclass
class HatDurumu:
    uretilen: list[Path] = field(default_factory=list)
    promptlu: list[Path] = field(default_factory=list)
    gecen: list[Path] = field(default_factory=list)
    reddedilen: list[tuple[Path, list[str]]] = field(default_factory=list)
    bantlar: list[dict] = field(default_factory=list)


def _task_dizinleri(envs: Path) -> list[Path]:
    return [p.parent for p in sorted(envs.glob("*/task.yaml"))
            if not p.parent.name.startswith("_")]


def _split_suzgeci(hedef: list[Path], split_yolu: str, bolum: str) -> list[Path]:
    """Bant/degerlendirme yalnizca split'in bir bolumunde kossun.

    Held-out'un tek isi epoch'lar arasi karsilastirmayi tasimak; egitimde
    gorulen gorevlerle karistirilirsa o karsilastirma anlamini yitirir.
    """
    if not split_yolu or not bolum:
        return hedef
    split = json.loads(Path(split_yolu).read_text(encoding="utf-8"))
    izinli = {Path(k["dir"]).name for k in split[bolum]}
    secilen = [d for d in hedef if d.name in izinli]
    print(f"  split suzgeci: {bolum} -> {len(secilen)}/{len(hedef)} gorev")
    return secilen


def _karantinaya(task_dir: Path, karantina: str) -> None:
    """Reddedilen bir task dizinini kenara al (silme -- teshis icin lazim)."""
    if not karantina or not task_dir.exists():
        return
    hedef = Path(karantina)
    hedef.mkdir(parents=True, exist_ok=True)
    varis = hedef / task_dir.name
    if varis.exists():
        shutil.rmtree(varis)
    shutil.move(str(task_dir), str(varis))


def _promptsuzlar(envs: Path) -> list[Path]:
    """Prompt'u yazilmamis task dizinleri -- hat yeniden kosturulabilsin."""
    out = []
    for d in _task_dizinleri(envs):
        try:
            t = Task.load(d)
        except Exception:  # noqa: BLE001 - bozuk yaml bir task'i atlatir, hatti durdurmaz
            continue
        if not t.prompt.strip():
            out.append(d)
    return out


def asama_uret(client, model: str, args, durum: HatDurumu) -> None:
    print(f"\n=== ASAMA uret  (aile={args.aile}, n={args.n}, "
          f"seed id {args.id_basla} ve sonrasi) ===\n")
    rng = random.Random(args.rng)
    # Karisik ailede secim seed basina yapiliyor: iki ayri kosu yerine tek
    # kosuda uretmek, id araliklarini elle bolmek zorunda birakmiyor.
    kmsk = args.karmasiklik or None
    if args.aile == "kod":
        seedler = [sample_kod(rng, args.id_basla + i, kmsk) for i in range(args.n)]
    elif args.aile == "kabuk":
        seedler = [sample(rng, args.id_basla + i) for i in range(args.n)]
    else:
        seedler = [sample_kod(rng, args.id_basla + i, kmsk)
                   if rng.random() < args.kod_orani
                   else sample(rng, args.id_basla + i)
                   for i in range(args.n)]
    out_dir = Path(args.out)
    adaylar = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as havuz:
        isler = {havuz.submit(generate_one, client, model, s, out_dir,
                              args.denemeler): (i + 1, s)
                 for i, s in enumerate(seedler)}
        for bitmis in as_completed(isler):
            sira, seed = isler[bitmis]
            aday = bitmis.result()
            adaylar.append(aday)
            # Adayin butun ciktisi tek blok: es zamanli satirlar karismasin.
            bas = f"[{sira}/{args.n}] {seed.id}  {'+'.join(seed.tools)}  {seed.konu}"
            if aday.kabul:
                son = f"    KABUL ({aday.deneme}. denemede) -> {aday.task_dir}"
                durum.uretilen.append(Path(aday.task_dir))
            elif aday.hata:
                son = f"    HATA  {aday.hata}"
            else:
                son = f"    RED   dusen kapilar: {', '.join(aday.dusen_kapilar)}"
            # Reddedilen adayin dizini de diske yazilmis oluyor. Yerinde
            # birakilirsa sonraki asamalar (prompt, bant) onu korpusun bir
            # parcasi sanip uzerinde calisiyor; karantinaya alinir.
            if not aday.kabul and aday.task_dir:
                _karantinaya(Path(aday.task_dir), args.karantina)
            print("\n".join([bas, *aday.log, son]))

    kabul = sum(1 for a in adaylar if a.kabul)
    print(f"\n  kabul: {kabul}/{len(adaylar)}")
    if kabul:
        ort = sum(a.deneme for a in adaylar if a.kabul) / kabul
        print(f"  kabul edilenlerde ortalama deneme: {ort:.1f}")


def asama_prompt(client, model: str, args, durum: HatDurumu) -> None:
    bekleyen = _promptsuzlar(Path(args.out))
    print(f"\n=== ASAMA prompt  ({len(bekleyen)} task bekliyor) ===\n")
    if not bekleyen:
        return

    def isle(d: Path):
        task = Task.load(d)
        return d, task, yaz_bir(client, model, task, args.denemeler)

    with ThreadPoolExecutor(max_workers=args.concurrency) as havuz:
        for bitmis in as_completed([havuz.submit(isle, d) for d in bekleyen]):
            d, task, sonuc = bitmis.result()
            uslup = task.metadata.get("seed", {}).get("uslup", "?")
            if sonuc.hata:
                print(f"  {task.id:34} HATA {sonuc.hata}")
            elif sonuc.yazildi and prompt_yaz(d, sonuc.prompt):
                print(f"  {task.id:34} YAZILDI  uslup={uslup}")
                durum.promptlu.append(d)
            else:
                print(f"  {task.id:34} YAZILAMADI")


def asama_kapi(args, durum: HatDurumu) -> None:
    """Prompt yazildiktan sonra TAM gauntlet -- prompt kapilari dahil.

    Uretim asamasindaki gauntlet spec_only kosuyor cunku prompt heniz yok;
    prompt_turkce ve prompt_covers_outputs ancak burada olculebiliyor.
    """
    hedef = durum.promptlu or _task_dizinleri(Path(args.out))
    print(f"\n=== ASAMA kapi  ({len(hedef)} task) ===\n")
    karantina = Path(args.karantina) if args.karantina else None
    for d in hedef:
        try:
            task = Task.load(d)
        except Exception as e:  # noqa: BLE001
            print(f"  {d.name:34} YUKLENEMEDI {type(e).__name__}")
            continue
        sonuclar, _ = run_gauntlet(task, spec_only=False)
        dusen = [r.name for r in sonuclar if not r.ok]
        if not dusen:
            print(f"  {task.id:34} GECTI")
            durum.gecen.append(d)
            continue
        print(f"  {task.id:34} RED  {', '.join(dusen)}")
        durum.reddedilen.append((d, dusen))
        _karantinaya(d, args.karantina)


def _olculen_gorevler(bant_out: str) -> set[str]:
    """Onceki (kesilmis) kosuda olculmus gorev adlari.

    Bozuk son satir tolere ediliyor: sureç yazarken oldurulmusse jsonl'in
    son satiri yarim kalmis olabilir.
    """
    yol = Path(bant_out) if bant_out else None
    if not yol or not yol.exists():
        return set()
    adlar: set[str] = set()
    for satir in yol.read_text(encoding="utf-8").splitlines():
        satir = satir.strip()
        if not satir:
            continue
        try:
            kayit = json.loads(satir)
        except json.JSONDecodeError:
            continue          # yarim yazilmis son satir
        ad = kayit.get("task_dir") or kayit.get("task_id")
        if ad:
            adlar.add(Path(ad).name)
    return adlar


def _bant_hedefleri(out: str) -> list[Path]:
    """Bant asamasinda `--out` virgulle ayrilmis birden cok klasor alabilir.

    Korpus artik tek klasor degil (uretim partileri ayri klasorlerde) ve
    hepsini TEK kosuda taramak gerekiyor: ayri kosular ayri vLLM ornekleri,
    ayri anlar ve karistirilamayan bir sira demek.
    """
    hedef: list[Path] = []
    for kok in [k.strip() for k in out.split(",") if k.strip()]:
        hedef.extend(_task_dizinleri(Path(kok)))
    return hedef


def asama_bant(client, model: str, args, durum: HatDurumu) -> None:
    """Her task'a N rollout at, pass-rate'e gore banda yerlestir.

    Bant olcumu egitilecek modelle ayni uctan yapiliyor: baska bir modelle
    olculen zorluk, egitimde gorulecek zorluk degildir.
    """
    hedef = durum.gecen or _bant_hedefleri(args.out)
    hedef = _split_suzgeci(hedef, args.split, args.bolum)
    if not hedef:
        print("olculecek task kalmadi (split suzgeci hepsini eledi)")
        return
    if args.karistir:
        # Klasor sirasiyla taramak, kosu yarida kalirsa elde yalnizca ilk
        # klasoru birakir -- ve klasorler aile/uretim partisine gore ayrildigi
        # icin o ornek temsili olmaz. Karistirilmis sirada yarim kosu bile
        # butun korpusun temsili bir ornegidir. Tohum sabit: tekrarlanabilir.
        random.Random(args.tohum).shuffle(hedef)
        print(f"  sira karistirildi (tohum={args.tohum})")
    if args.devam:
        # Colab runtime'i bir gunde iki kez yeniden atandi ve tam korpus
        # taramasi saatler suruyor. Olculmus gorevleri atla; jsonl gorev
        # basina flush edildigi icin kesilen kosunun ciktisi gecerlidir.
        olculen = _olculen_gorevler(args.bant_out)
        if olculen:
            once = len(hedef)
            hedef = [d for d in hedef if d.name not in olculen]
            print(f"  devam: {len(olculen)} gorev zaten olculmus, "
                  f"{once} -> {len(hedef)} kaldi")
    print(f"\n=== ASAMA bant  ({len(hedef)} task x {args.rollouts} rollout) ===\n")
    # Episode'lar birbirinden bagimsiz. Sirali kosmak sunucuyu bos
    # birakiyordu: 80 gorev x 8 rollout x ~5 tur = 3200 ardisik istek.
    # Havuz hem gorevler hem rollout'lar arasinda paralellik veriyor.
    # `checks:` olmayan gorev DOGRULANAMAZ: `grade()` yazilmamis bir
    # tests/check.py'yi calistirmaya calisip RuntimeError atiyor. Bunlar
    # zaten split'te "referansi gecmiyor" diye dislanmisti, ama split
    # suzgeci kapaliyken (tum korpus taramasi) yeniden iceri giriyorlar.
    dogrulanabilir, checksiz = [], []
    for d in hedef:
        try:
            (dogrulanabilir if Task.load(d).checks else checksiz).append(d)
        except Exception:  # noqa: BLE001 - bozuk task.yaml da elenmeli
            checksiz.append(d)
    if checksiz:
        print(f"  check'siz/bozuk {len(checksiz)} gorev elendi: "
              f"{', '.join(d.name for d in checksiz[:4])}"
              f"{' ...' if len(checksiz) > 4 else ''}")
    hedef = dogrulanabilir

    isler = [(d, r) for d in hedef for r in range(args.rollouts)]
    toplanan: dict[str, list] = {}
    hatali: dict[str, str] = {}
    tamam = 0

    def bir_episode(is_):
        """Tek bir episode'un hatasi butun kosuyu oldurmemeli.

        512 gorev x 8 rollout saatler suruyor; bir bozuk gorev yuzunden
        hepsini kaybetmek kabul edilemez. Hata goreve yazilir, o gorev
        raporlanir ve tarama devam eder.
        """
        d, _ = is_
        try:
            task = Task.load(d)
            return d, task.id, run_model(task, client, model, args.protocol,
                                         verbose=False), None
        except Exception as e:  # noqa: BLE001
            return d, d.name, None, f"{type(e).__name__}: {e}"

    # DIKKAT: "w" kesilen bir kosunun ciktisini siler. --devam varsa eklenir.
    kayit = (open(args.bant_out, "a" if args.devam else "w", encoding="utf-8")
             if args.bant_out else None)
    def _gorevi_yaz(task_id):
        """Bir gorevin butun rollout'lari bitince HEMEN yaz.

        Onceki hal butun episode'lar bittikten SONRA toplu yaziyordu: 512
        gorevlik tarama saatlerce kosup sonunda yaziyordu, yani kesilen
        kosudan geriye hicbir sey kalmiyordu ve --devam'in okuyacagi dosya
        hic olusmuyordu.
        """
        kayitlar = toplanan.pop(task_id)
        ozet = summarize(task_id, [e for _, e in kayitlar], args.rollouts)
        ozet["task_dir"] = str(kayitlar[0][0])
        durum.bantlar.append(ozet)
        print(f"  {task_id:34} {ozet['band']:10} "
              f"{ozet['solved']}/{ozet['rollouts']}  "
              f"kismi={ozet['mean_partial']:.2f} tur={ozet['mean_turns']:.1f}",
              flush=True)
        if kayit:
            kayit.write(json.dumps(ozet, ensure_ascii=False) + "\n")
            kayit.flush()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as havuz:
            for bitmis in as_completed([havuz.submit(bir_episode, i) for i in isler]):
                d, task_id, ep, hata = bitmis.result()
                tamam += 1
                if hata is not None:
                    if task_id not in hatali:
                        hatali[task_id] = hata
                        print(f"  HATA {task_id}: {hata[:120]}", flush=True)
                    continue
                kayitlar = toplanan.setdefault(task_id, [])
                kayitlar.append((d, ep))
                if len(kayitlar) == args.rollouts:
                    _gorevi_yaz(task_id)
                if tamam % 50 == 0:
                    print(f"  ... {tamam}/{len(isler)} episode", flush=True)

        for task_id in sorted(toplanan):      # eksik rollout'lu artiklar
            _gorevi_yaz(task_id)
    finally:
        if kayit:
            kayit.close()
    if hatali:
        print(f"\n  {len(hatali)} gorev hata verdi ve atlandi:")
        for task_id, h in list(hatali.items())[:10]:
            print(f"    {task_id:34} {h[:100]}")


def _ozet(durum: HatDurumu, args) -> None:
    print("\n" + "=" * 60)
    print("HAT OZETI")
    print(f"  uretilen spec       : {len(durum.uretilen)}")
    print(f"  prompt yazilan      : {len(durum.promptlu)}")
    print(f"  kapilardan gecen    : {len(durum.gecen)}")
    if durum.reddedilen:
        sayac: dict[str, int] = {}
        for _, dusen in durum.reddedilen:
            for k in dusen:
                sayac[k] = sayac.get(k, 0) + 1
        print(f"  kapilarda reddedilen: {len(durum.reddedilen)}  "
              + ", ".join(f"{k} x{v}"
                          for k, v in sorted(sayac.items(), key=lambda kv: -kv[1])))
    if durum.bantlar:
        gruplar: dict[str, list[str]] = {}
        for o in durum.bantlar:
            gruplar.setdefault(o["band"], []).append(o["task_id"])
        for ad in ("BANT", "OLU-kolay", "OLU-zor"):
            if ad in gruplar:
                print(f"  {ad:10} {len(gruplar[ad]):4}")
        kullanilir = len(gruplar.get("BANT", []))
        print(f"\n  EGITIMDE ISE YARAR: {kullanilir}/{len(durum.bantlar)}")
        if args.bant_out:
            print(f"  bant kayitlari -> {args.bant_out}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asamalar", default=",".join(ASAMALAR),
                        help=f"virgulle ayrilmis: {', '.join(ASAMALAR)}")
    parser.add_argument("--sandbox", choices=["docker", "singularity", "yerel"],
                        default="docker",
                        help="singularity = HPC (MN5); yerel = Docker'siz "
                             "(Colab), izolasyon yok. uret ve kapi asamalari "
                             "konteyner ister, yerelde kosmaz.")
    parser.add_argument("--n", type=int, default=10, help="uretilecek aday sayisi")
    parser.add_argument("--aile", choices=["kabuk", "kod", "karisik"],
                        default="kabuk",
                        help="kabuk = terminal gorevleri, kod = Python "
                             "yazma/onarma, karisik = --kod-orani ile bolusur")
    parser.add_argument("--karmasiklik", default="",
                        choices=["", "duz", "orta", "derin"],
                        help="kod ailesinde karmasikligi sabitle; bos "
                             "birakilirsa KOD_KARMASIKLIK agirliklariyla "
                             "ornekleniyor")
    parser.add_argument("--kod-orani", type=float, default=0.6,
                        help="--aile karisik iken kod ailesinin payi")
    parser.add_argument("--id-basla", type=int, default=0,
                        help="seed id numaralari buradan bassin (cakismayi onler)")
    parser.add_argument("--rng", type=int, default=0)
    parser.add_argument("--out", default="envs_gen")
    parser.add_argument("--karantina", default="envs_gen/_red",
                        help="kapilarda dusen task'lar buraya tasinsin (bos = tasima)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--denemeler", type=int, default=3,
                        help="bir aday/prompt icin en fazla kac onarim turu")
    parser.add_argument("--rollouts", type=int, default=8,
                        help="bant asamasinda task basina rollout")
    parser.add_argument("--protocol", choices=["auto", "native", "text"], default="auto")
    parser.add_argument("--bant-out", default="data/bant.jsonl")
    parser.add_argument("--devam", action="store_true",
                        help="--bant-out'ta zaten olculmus gorevleri atla ve "
                             "dosyaya EKLE. Bu bayrak olmadan dosya sifirdan "
                             "yazilir, yani kesilen kosunun ciktisi silinir.")
    parser.add_argument("--karistir", action="store_true",
                        help="bant asamasinda gorev sirasini karistir; yarida "
                             "kesilen kosu bile korpusun temsili ornegi olur")
    parser.add_argument("--tohum", type=int, default=1234,
                        help="--karistir icin tohum (split.py ile ayni)")
    parser.add_argument("--split", default="",
                        help="split.json yolu -- bant/degerlendirmeyi bir bolumle sinirlar")
    parser.add_argument("--bolum", default="", choices=["", "train", "holdout"],
                        help="--split verildiyse hangi bolum olculsun")
    parser.add_argument("--sicaklik", type=float, default=-1.0,
                        help="degerlendirmede 0 kullan; -1 = modelin varsayilani")
    parser.add_argument("--yanit-token", type=int, default=0,
                        help="yanit basina token butcesi (runner MAX_TOKENS). "
                             "0 = dokunma. Egitimin max_completion_length'i "
                             "ile ayni verilmezse bant daha dar bir butceyle "
                             "olculur.")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--istek-araligi", type=float, default=0.0,
                        help="iki API istegi arasinda en az bu kadar saniye bekle")
    parser.add_argument("--reasoning-effort", default="none",
                        help="reasoning modellerinde dusunme butcesi; 'none' "
                             "kapatir. Bos birakilirsa alan hic gonderilmez. "
                             "Ucun tanimadigi bir deger sessizce dusurulur.")
    parser.add_argument("--fiyat-giris", type=float, default=0.0,
                        help="$/1M giris token -- maliyet raporu icin")
    parser.add_argument("--fiyat-cikis", type=float, default=0.0,
                        help="$/1M cikis token -- maliyet raporu icin")
    args = parser.parse_args()

    # Bayraklari runner'in okudugu env'e BAGLA. Bu halka daha once hic
    # kurulmamisti: `--sicaklik` yalnizca add_argument'ta geciyordu, hicbir
    # yerde okunmuyordu, ve runner varsayilan 0.7'de kaliyordu. Sweep ve
    # eval'ler 1.0 istenirken 0.7'de kostu. runner iki degeri de cagri
    # aninda okudugu icin burada set etmek yeterli (modul zaten import
    # edilmis durumda).
    if args.sicaklik >= 0:
        os.environ["VDS_TEMPERATURE"] = str(args.sicaklik)
    if args.yanit_token > 0:
        os.environ["VDS_MAX_TOKENS"] = str(args.yanit_token)

    istenen = [a.strip() for a in args.asamalar.split(",") if a.strip()]
    bilinmeyen = [a for a in istenen if a not in ASAMALAR]
    if bilinmeyen:
        print(f"bilinmeyen asama: {', '.join(bilinmeyen)}  "
              f"(gecerli: {', '.join(ASAMALAR)})")
        return 2

    ok, info = sandbox_hazirla(args.sandbox)
    if not ok:
        print(f"{args.sandbox} sandbox kullanilamiyor: {info}")
        return 1

    # Uretim asamalari izolasyonsuz kosmaz (kapilar konteyner aciyor); bant
    # asamasi ise Colab'de yerel sandbox'la kosacak. Singularity gercek bir
    # konteyner verdigi icin bu kisit yalnizca yerel moda ait.
    if args.sandbox == "yerel":
        kutu_isteyen = [a for a in istenen if a in ("uret", "kapi")]
        if kutu_isteyen:
            print(f"yerel sandbox ile kosulamayan asamalar: "
                  f"{', '.join(kutu_isteyen)} (kapilar konteyner istiyor)")
            return 2

    model = resolve_model(args.model)
    if not model:
        print("--model verilmedi ve .env'de OPENAI_MODEL_NAME yok")
        return 2

    # Uc bir kez dogrulanir: yarim kosuda 401 yemek en pahali hata.
    ok_uc, bilgi_uc = preflight(args.base_url, args.api_key, model)
    if not ok_uc:
        print(f"UC KONTROLU BASARISIZ: {bilgi_uc}")
        return 2
    print(f"uc dogrulandi: {bilgi_uc}")
    print(f"model={model}  asamalar={','.join(istenen)}  {args.sandbox}={info}")

    sayac = TokenSayaci()
    client = _SayanClient(
        make_client(args.base_url, args.api_key, args.istek_araligi, verbose=False,
                    reasoning_effort=args.reasoning_effort),
        sayac)

    durum = HatDurumu()
    basladi = time.monotonic()
    if "uret" in istenen:
        sayac.asama = "uret"
        asama_uret(client, model, args, durum)
    if "prompt" in istenen:
        sayac.asama = "prompt"
        asama_prompt(client, model, args, durum)
    if "kapi" in istenen:
        sayac.asama = "kapi"       # kapilar API kullanmaz, sayaci bos kalir
        asama_kapi(args, durum)
    if "bant" in istenen:
        sayac.asama = "bant"
        asama_bant(client, model, args, durum)

    _ozet(durum, args)
    print(sayac.rapor(args.fiyat_giris, args.fiyat_cikis))
    print(f"\nsure: {(time.monotonic() - basladi) / 60:.1f} dk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
