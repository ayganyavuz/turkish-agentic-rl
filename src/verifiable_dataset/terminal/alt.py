"""Rakip cozum uret ve iki cozumun ayni sonuca vardigini dogrula.

    python -m verifiable_dataset.terminal.alt --all --envs envs_gen \
        --model X --base-url Y

Uretici model gorevi bir kez cozdu ve o cozumun ciktisi ground truth
oldu. Burada ayni modele ikinci bir cozum yazdiriliyor -- referansi
GORMEDEN, yalnizca prompt'tan ve mumkunse baska araclarla. Iki cozum
ayni ciktida bulusmazsa taraflardan biri yaniliyor demektir.

Bu bosluk pahaliya patliyordu: hicbir kapi referansin *dogru* olup
olmadigini sormuyordu, yalnizca calistigini. gen-s-0015'te referans
rakamlari birbirine yapistirip toplami 105562 bulmus, dogru cevap
2189.8; gorev veri setine girmis ve politika dogru cevabi verdigi icin
sekiz rollout boyunca sifir odul almisti.

Mutabakat kanit DEGIL: ayni model ayni hatayi iki kez yapabilir, cunku
hatalar bagimsiz degil. Bu yuzden rakip cozum farkli araclara
yonlendiriliyor ve uyusmazlikta karar modele birakilmiyor tek basina --
model ya cozumunu duzeltir ya da gorevin bozuk oldugunu bildirir, ve
ikinci durumda task karantinaya alinir, sessizce gecmez.
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from verifiable_dataset.terminal.equiv import run_alt
from verifiable_dataset.terminal.llm import cagir, make_client, preflight, resolve_model
from verifiable_dataset.terminal.sandbox import docker_available
from verifiable_dataset.terminal.seeds import tools_of_image
from verifiable_dataset.terminal.task import Task

BOLUM_RE = re.compile(r"^###\s+(COZUM|BOZUK)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$", re.MULTILINE)
METADATA_RE = re.compile(r"^metadata:\s*$", re.MULTILINE)

TALIMAT = """\
Sen bir Linux terminal gorevini cozen muhendissin. Sana gorev metni ve
baslangictaki dosyalar verilecek. Gorevi cozen bash komutlarini yazacaksin.

Bu gorev daha once BASKA bir yaklasimla cozuldu. Senin isin bagimsiz bir
ikinci cozum yazmak: mumkun oldugunca farkli araclar ve farkli bir yol
sec. Amac ayni sonuca baska bir yoldan varmak.

Yanitin tam olarak su iki bicimden biri olmali, baska hicbir sey yazma:

### COZUM
<bash komutlari>

ya da gorev metni kendi icinde tutarsizsa, imkansizsa veya istenen sey
baslangic dosyalariyla uretilemiyorsa:

### BOZUK
<tek paragraf gerekce>

KURALLAR:
1. /workspace icinde calisirsin, dosyalar oradadir.
2. Yalnizca imajda kurulu araclari cagir.
3. Deterministik ol: date, $RANDOM, ag erisimi yok.
4. Gorev metninin istedigi dosyalari tam olarak istenen adla uret.
5. Bicim sartlarina birebir uy -- "sadece sayi olsun", "virgulle ayrilmis",
   "baslik satiri olmasin" gibi kisitlar sonucu belirler.
6. "### BOZUK" cevabini kolay verme; once cozmeyi dene.
"""


@dataclass
class AltUretim:
    task_id: str
    kabul: bool = False
    script: str = ""
    bozuk: str = ""          # model gorevi bozuk ilan ettiyse gerekcesi
    deneme: int = 0
    hata: str = ""
    son_dusen: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def durum(self) -> str:
        if self.kabul:
            return "MUTABIK"
        if self.bozuk:
            return "BOZUK"
        if self.hata:
            return "HATA"
        return "UYUSMAZ"


def brief(task: Task) -> str:
    kurulu = ", ".join(tools_of_image(task.image))
    yollar = [o.get("path", "") for o in task.outputs if o.get("path")]
    setup = task.setup.strip()
    if len(setup) > 2500:
        setup = setup[:2500] + "\n... (kisaltildi)"
    return f"""\
Imajda kurulu araclar (baskasini cagirma):
{kurulu}

GOREV METNI:
{task.prompt.strip()}

URETILMESI GEREKEN DOSYA/DIZINLER:
{', '.join(yollar) or '(gorev metninden cikar)'}

BASLANGIC DUNYASI (bu komutlar gorev baslamadan calistirildi):
{setup}

Cozumu yaz."""


def ayristir(ham: str) -> tuple[str, str]:
    """(script, bozuk_gerekce) dondur; ikisi de bos ise bicim hatasi."""
    ham = FENCE_RE.sub("", ham)
    parcalar = BOLUM_RE.split(ham)
    if len(parcalar) < 3:
        return "", ""
    bolum, govde = parcalar[1].strip(), parcalar[2].strip()
    return (govde, "") if bolum == "COZUM" else ("", govde or "(gerekce yazilmadi)")


def geri_bildirim(sonuc, task: Task) -> str:
    """Uyusmazligi modele anlat ve iki cikis yolu sun.

    Hangi tarafin yanildigini biz bilmiyoruz: cozum de yanlis olabilir,
    gorevin ground truth'u da. Karari modele birakmak yerine ikisini de
    acikca secenek olarak koyuyoruz, boylece "bozuk" yaniti bir kacis
    degil bilincli bir bildirim oluyor.
    """
    satirlar = [
        "Yazdigin cozum gorevin beklenen ciktisini uretmedi.",
        "",
        f"Cozum exit={sonuc.exit_code}"
        + (f", stderr: {sonuc.stderr}" if sonuc.stderr else ""),
        f"Gecen kontrol: {sonuc.derived_passed}/{sonuc.derived_total}",
        "",
        "Tutmayan kontroller (beklenen = gorevin ground truth'u, bulunan = senin ciktin):",
    ]
    satirlar += [f"- {f}" for f in sonuc.failures[:6]]
    satirlar += [
        "",
        "Iki ihtimal var ve hangisi oldugunu sen degerlendireceksin:",
        "1. Cozumun yanlis. O zaman duzelt ve '### COZUM' ile yeni cozumu yaz.",
        "2. Beklenen cikti gorev metniyle celisiyor -- gorev metninin istedigi",
        "   sey ile ground truth farkli seyler. O zaman '### BOZUK' yaz ve",
        "   celiskinin ne oldugunu tek paragrafta anlat.",
        "",
        "Ground truth'u koru korune taklit etme: gorev metni ne diyorsa o dogrudur.",
    ]
    return "\n".join(satirlar)


def uret_bir(client, model: str, task: Task, max_deneme: int = 3) -> AltUretim:
    sonuc = AltUretim(task_id=task.id)
    if not task.checks:
        sonuc.hata = "turetilmis check yok; once derive calistirilmali"
        return sonuc

    mesajlar = [{"role": "system", "content": TALIMAT},
                {"role": "user", "content": brief(task)}]

    for deneme in range(1, max_deneme + 1):
        sonuc.deneme = deneme
        try:
            yanit = cagir(client, sonuc.log.append, model=model, messages=mesajlar,
                          temperature=0.7, max_tokens=4096)
        except Exception as e:  # noqa: BLE001 - bir task kosuyu bitirmemeli
            sonuc.hata = f"model cagrisi basarisiz: {type(e).__name__}: {e}"
            return sonuc

        ham = yanit.choices[0].message.content or ""
        script, bozuk = ayristir(ham)

        if bozuk:
            sonuc.bozuk = bozuk
            sonuc.log.append(f"    deneme {deneme}: model gorevi BOZUK ilan etti")
            return sonuc

        if not script:
            sonuc.log.append(f"    deneme {deneme}: yanit '### COZUM' bicimine uymuyor")
            mesajlar += [{"role": "assistant", "content": ham},
                         {"role": "user", "content":
                          "Yanitin ayristirilamadi. '### COZUM' ya da '### BOZUK' "
                          "basligiyla, bicime birebir uyarak tekrar yaz."}]
            continue

        try:
            r = run_alt(task, {"name": "rakip", "script": script}, task.checks)
        except Exception as e:  # noqa: BLE001 - sandbox hatasi adayi duserir
            sonuc.hata = f"rakip cozum calistirilamadi: {type(e).__name__}: {e}"
            return sonuc

        if r.ok:
            sonuc.kabul, sonuc.script = True, script
            sonuc.log.append(f"    deneme {deneme}: mutabik "
                             f"({r.derived_passed}/{r.derived_total})")
            return sonuc

        sonuc.son_dusen = r.failures
        sonuc.log.append(f"    deneme {deneme}: uyusmazlik "
                         f"{r.derived_passed}/{r.derived_total}")
        if deneme == max_deneme:
            return sonuc
        mesajlar += [{"role": "assistant", "content": ham},
                     {"role": "user", "content": geri_bildirim(r, task)}]

    return sonuc


def yaz_alt(task_dir: Path, script: str, name: str = "rakip-1") -> bool:
    """alt_solutions blogunu task.yaml'a ekle (metadata'nin hemen ustune)."""
    yol = task_dir / "task.yaml"
    metin = yol.read_text(encoding="utf-8")
    if re.search(r"^alt_solutions:", metin, re.MULTILINE):
        return False
    govde = "\n".join(f"      {ln}" if ln.strip() else ""
                      for ln in script.strip().split("\n"))
    blok = f"alt_solutions:\n  - name: {name}\n    script: |\n{govde}\n\n"
    if METADATA_RE.search(metin):
        metin = METADATA_RE.sub(lambda _: blok + "metadata:", metin, count=1)
    else:
        metin = metin.rstrip() + "\n\n" + blok
    yol.write_text(metin, encoding="utf-8")
    return True


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default="envs_gen")
    p.add_argument("--task", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", default="", help="adinda bu gecen task'lar")
    p.add_argument("--model", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--denemeler", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4,
                   help="es zamanli task; docker tarafi 8'den sonra yavasliyor")
    p.add_argument("--istek-araligi", type=float, default=0.0)
    p.add_argument("--karantina", default="",
                   help="mutabakat saglanmayan task'lari bu dizine tasi")
    p.add_argument("--yaz", action="store_true",
                   help="mutabik kalan rakip cozumu task.yaml'a yaz")
    args = p.parse_args()

    if args.all:
        task_dirs = sorted(d.parent for d in Path(args.envs).glob("*/task.yaml")
                           if not d.parent.name.startswith("_"))
    elif args.task:
        task_dirs = [Path(args.task)]
    else:
        print("--task ya da --all ver")
        return 2
    if args.only:
        task_dirs = [d for d in task_dirs if args.only in d.name]
    if args.limit:
        task_dirs = task_dirs[:args.limit]
    if not task_dirs:
        print("task bulunamadi")
        return 1

    args.model = resolve_model(args.model)
    if not args.model:
        print("--model verilmedi ve .env'de OPENAI_MODEL_NAME yok")
        return 2
    ok_uc, bilgi = preflight(args.base_url, args.api_key, args.model)
    if not ok_uc:
        print(f"UC KONTROLU BASARISIZ: {bilgi}")
        return 2
    print(f"uc dogrulandi: {bilgi}")

    ok, info = docker_available()
    if not ok:
        print(f"Docker daemon'a ulasilamiyor: {info}")
        return 1

    client = make_client(args.base_url, args.api_key, args.istek_araligi)
    print(f"model={args.model}  {len(task_dirs)} task  "
          f"es zamanlilik={args.concurrency}  docker={info}\n")

    def isle(task_dir: Path) -> tuple[Path, AltUretim]:
        try:
            task = Task.load(task_dir)
        except Exception as e:  # noqa: BLE001
            s = AltUretim(task_id=task_dir.name, hata=f"yuklenemedi: {e}")
            return task_dir, s
        return task_dir, uret_bir(client, args.model, task, args.denemeler)

    sonuclar: list[tuple[Path, AltUretim]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as havuz:
        isler = {havuz.submit(isle, d): d for d in task_dirs}
        for bitmis in as_completed(isler):
            task_dir, s = bitmis.result()
            sonuclar.append((task_dir, s))
            yazildi = ""
            if s.kabul and args.yaz and yaz_alt(task_dir, s.script):
                yazildi = "  -> task.yaml'a yazildi"
            print("\n".join([f"{s.task_id:34} {s.durum}", *s.log]) + yazildi)
            if s.bozuk:
                print(f"{'':34}   {s.bozuk[:220]}")
            elif s.son_dusen:
                print(f"{'':34}   {s.son_dusen[0][:220]}")
            elif s.hata:
                print(f"{'':34}   {s.hata[:220]}")

    # -- ozet ---------------------------------------------------------
    sayac: dict[str, int] = {}
    for _, s in sonuclar:
        sayac[s.durum] = sayac.get(s.durum, 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(sayac.items())))
    supheli = [(d, s) for d, s in sonuclar if not s.kabul]
    if supheli:
        print(f"\nMutabakat saglanamayan {len(supheli)} task:")
        for d, s in sorted(supheli, key=lambda x: x[1].task_id):
            print(f"  {s.task_id:34} {s.durum}")

    if args.karantina and supheli:
        hedef = Path(args.karantina)
        hedef.mkdir(parents=True, exist_ok=True)
        for d, s in supheli:
            if s.hata:      # altyapi hatasi task'in sucu degil, karantinaya alma
                continue
            d.rename(hedef / d.name)
        print(f"\nkarantina -> {hedef}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
