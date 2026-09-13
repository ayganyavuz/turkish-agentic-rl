"""tr-013'un iki varyantini uretir: kural verilmis ve kural verilmemis.

    python3 envs/_gen/tr_013_yonelme.py

Iki task ayni veriyi, ayni referansi ve ayni check'leri paylasir. Tek fark
prompt: birinde unlu uyumu kurali yaziyor (model yalnizca uygular), digerinde
yazmiyor (model Turkce bilmek zorunda). Zorluk kolu bu; ayni isi olcen iki
aday uretip kalibrasyonda hangisinin banda dustugune bakacagiz.

Uretim uc sozlesmeyi dogrulamadan yazmaz:

  1. Referans butun check'leri gecmeli.
  2. S0 (cozum calismadan) hicbir check'i gecmemeli.
  3. Naif cozumler KISMI puan almali. Bu sefer dogru cozumden ozellik
     cikararak degil, modelin gercekten uretebilecegi naif stratejileri
     (hep 'a ekle, ascii unlu kumesi, ilk unluye bak) kosturarak olculuyor
     -- tr-010'da yaptigim ablasyon olcumu reward dagilimi hakkinda yanlis
     bilgi vermisti.

SEHIR SECIMI: Kocaeli / Kirklareli / Tunceli listede YOK. Bu adlarin sonundaki
-i iyelik ekidir, TDK'ya gore yonelme hali "Kocaeli'ne" olur. Listeye alsaydik
Turkceyi dogru bilen modeli cezalandiran bir check yazmis olurduk.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from verifiable_dataset.terminal.checks import run_checks  # noqa: E402

IMAGE = "verifiable-dataset/base:latest"

# Dort durumun hepsi temsil ediliyor: kalin/ince unlu x unsuzle/unluyle bitis.
# Yanlarindaki notlar hangi hatayi yakaladiklarini soyluyor.
SEHIRLER = [
    "Ankara",        # kalin, unluyle biter   -> Ankara'ya
    "İstanbul",      # ilk unlu ince ama SON unlu kalin -> İstanbul'a
    "İzmir",         # ince, unsuzle biter    -> İzmir'e
    "Kütahya",       # ilk unlu ince, son unlu kalin, unluyle biter -> Kütahya'ya
    "Çankırı",       # 'i' unlu sayilmali     -> Çankırı'ya
    "Bolu",          # kalin, unluyle biter   -> Bolu'ya
    "Ağrı",          # 'i' unlu sayilmali     -> Ağrı'ya
    "Kastamonu",     # kalin, unluyle biter   -> Kastamonu'ya
    "Malatya",       # kalin, unluyle biter   -> Malatya'ya
    "Trabzon",       # kalin, unsuzle biter   -> Trabzon'a
    "Uşak",          # kalin, unsuzle biter   -> Uşak'a
    "Zonguldak",     # ozel adda yumusama YOK -> Zonguldak'a
    "Aydın",         # kalin, unsuzle biter   -> Aydın'a
    "Sinop",         # ozel adda yumusama YOK -> Sinop'a
    "Bayburt",       # ozel adda yumusama YOK -> Bayburt'a
    "Bingöl",        # 'o' ince unlu          -> Bingöl'e
    "Gaziantep",     # ince + yumusama YOK    -> Gaziantep'e
    "Karabük",       # 'u' ince + yumusama YOK -> Karabük'e
    "Bilecik",       # ince + yumusama YOK    -> Bilecik'e
    "Mersin",        # ince, unsuzle biter    -> Mersin'e
    "Rize",          # ince, unluyle biter    -> Rize'ye
    "Gümüşhane",     # ince, unluyle biter    -> Gümüşhane'ye
    "Niğde",         # ince, unluyle biter    -> Niğde'ye
    "Kayseri",       # ince, unluyle biter    -> Kayseri'ye
]

SETUP = "cat > sehirler.txt <<'EOF'\n" + "\n".join(SEHIRLER) + "\nEOF"

REF = r'''
KALIN, INCE = "aıou", "eiöü"
UNLU = KALIN + INCE

def tr_kucuk(s):
    # Turkce katlama: İ -> i, I -> ı. Duz .lower() "İ"yi iki kodnoktaya acar.
    return s.replace("İ", "i").replace("I", "ı").lower()

def yonelme(ad):
    harf = tr_kucuk(ad)
    son_unlu = next(h for h in reversed(harf) if h in UNLU)
    ek = "a" if son_unlu in KALIN else "e"
    if harf[-1] in UNLU:          # unluyle bitiyorsa kaynastirma y
        ek = "y" + ek
    return f"{ad}'{ek}"

with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        f.write(yonelme(s) + "\n")
'''

# -- naif temeller: modelin gercekten uretebilecegi yanlis stratejiler ----
NAIF = {
    "hep 'a ekliyor": r'''
with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        f.write(s + "'a\n")
''',
    "ascii unlu kumesi": r'''
KALIN, INCE = "aou", "ei"
UNLU = KALIN + INCE
with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        h = s.lower()
        su = next((x for x in reversed(h) if x in UNLU), "a")
        ek = "a" if su in KALIN else "e"
        if h[-1] in UNLU:
            ek = "y" + ek
        f.write(f"{s}'{ek}\n")
''',
    "ilk unluye bakiyor": r'''
KALIN, INCE = "aıou", "eiöü"
UNLU = KALIN + INCE
def tr_kucuk(s):
    return s.replace("İ", "i").replace("I", "ı").lower()
with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        h = tr_kucuk(s)
        su = next(x for x in h if x in UNLU)
        ek = "a" if su in KALIN else "e"
        if h[-1] in UNLU:
            ek = "y" + ek
        f.write(f"{s}'{ek}\n")
''',
    "kaynastirma y'yi unutuyor": r'''
KALIN, INCE = "aıou", "eiöü"
UNLU = KALIN + INCE
def tr_kucuk(s):
    return s.replace("İ", "i").replace("I", "ı").lower()
with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        h = tr_kucuk(s)
        su = next(x for x in reversed(h) if x in UNLU)
        f.write(f"{s}'{'a' if su in KALIN else 'e'}\n")
''',
    "ozel adda unsuz yumusatiyor": r'''
KALIN, INCE = "aıou", "eiöü"
UNLU = KALIN + INCE
YUMUSAK = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
def tr_kucuk(s):
    return s.replace("İ", "i").replace("I", "ı").lower()
with open("sehirler.txt", encoding="utf-8") as f:
    sehirler = [s.strip() for s in f if s.strip()]
with open("yonelme.txt", "w", encoding="utf-8") as f:
    for s in sehirler:
        h = tr_kucuk(s)
        su = next(x for x in reversed(h) if x in UNLU)
        ek = "a" if su in KALIN else "e"
        govde = s
        if h[-1] in UNLU:
            ek = "y" + ek
        elif h[-1] in YUMUSAK:
            govde = s[:-1] + YUMUSAK[h[-1]]
        f.write(f"{govde}'{ek}\n")
''',
}

ORTAK_KUYRUK = """
Sonucu `/workspace/yonelme.txt` dosyasına yaz: her satırda bir şehir, girdi
dosyasındaki sırayla, başka hiçbir şey yazma.

Kesme işareti olarak düz ASCII tırnak (') kullan.
"""

PROMPT_KURAL = """Bir Linux terminalindesin, çalışma dizinin /workspace.

`sehirler.txt` dosyasında her satırda bir şehir adı var. Her adın yönelme
hâlini (-e hâli) üret.

Kural:

  1. Eke gelen ünlü, kelimenin SON ünlüsüne göre belirlenir:
       son ünlü a, ı, o, u ise  ->  a
       son ünlü e, i, ö, ü ise  ->  e
  2. Kelime bir ünlüyle bitiyorsa ekten önce kaynaştırma harfi `y` gelir.
  3. Özel ad olduğu için ek kesme işaretiyle ayrılır ve gövde değişmez;
     sondaki p, ç, t, k harfleri yumuşatılmaz.

Örnek: Konya -> Konya'ya, Samsun -> Samsun'a, Edirne -> Edirne'ye,
Denizli -> Denizli'ye, Kars -> Kars'a, Sivas -> Sivas'a
""" + ORTAK_KUYRUK

PROMPT_BILGI = """Bir Linux terminalindesin, çalışma dizinin /workspace.

`sehirler.txt` dosyasında her satırda bir şehir adı var. Her adın yönelme
hâlini (-e hâli) üret. Türkçe yazım kurallarına uy.

Örnek: Konya -> Konya'ya
""" + ORTAK_KUYRUK

# Check merdiveni: file_contains satirlari, tek tek zor vakalari olcer.
# Boylece kurali genel olarak yakalayip bir sehri kacirmak 0 degil, kismi
# puan alir -- tam esitlik tek basina cok kaba bir sinyal olurdu.
ZOR_VAKALAR = [
    "İstanbul'a",     # ilk unlu ince, son unlu kalin
    "Kütahya'ya",     # ayni tuzak + kaynastirma y
    "Çankırı'ya",     # 'i' unlu sayilmali
    "Bingöl'e",       # 'o' ince unlu
    "Karabük'e",      # 'u' ince + ozel adda yumusama yok
    "Bayburt'a",      # ozel adda yumusama yok
]


def _run(src: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", src], cwd=cwd,
                          capture_output=True, text=True, timeout=60)


def _sh(src: str) -> str:
    return f"python3 - <<'PYEOF'\n{src.strip()}\nPYEOF"


def _block(text: str, indent: str = "  ") -> str:
    return "".join(f"{indent}{ln}\n" if ln.strip() else "\n"
                   for ln in text.strip().splitlines())


def _yaml_check(c: dict) -> str:
    parts = [f"op: {v}" if k == "op" else f"{k}: {json.dumps(v, ensure_ascii=False)}"
             for k, v in c.items()]
    return "  - {" + ", ".join(parts) + "}"


def _grade(root: Path, checks: list[dict]) -> tuple[int, list[dict]]:
    res = run_checks(root, checks, IMAGE)
    return sum(1 for r in res if r["ok"]), res


def _world(base: Path, ad: str) -> Path:
    d = base / ad
    d.mkdir()
    (d / "sehirler.txt").write_text("\n".join(SEHIRLER) + "\n", encoding="utf-8")
    return d


def _task_yaml(task_id: str, prompt: str, checks: list[dict], notlar: str) -> str:
    return (
        "# ELLE DUZENLEME. envs/_gen/tr_013_yonelme.py uretir.\n"
        f"id: {task_id}\n"
        "split: single_step\n"
        f"image: {IMAGE}\n"
        "workdir: /workspace\n"
        "max_turns: 6\n"
        "\n"
        "setup: |\n" + _block(SETUP) +
        "\n"
        "prompt: |\n" + _block(prompt) +
        "\n"
        "reference_solution: |\n" + _block(_sh(REF)) +
        "\n"
        "checks:\n" + "\n".join(_yaml_check(c) for c in checks) + "\n"
        "\n"
        "metadata:\n"
        "  domain: turkish_morphology\n"
        "  language: tr\n"
        "  n_ref_steps: 1\n"
        "  generator: ../_gen/tr_013_yonelme.py\n"
        "  notes: >\n" + _block(notlar, "    ")
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tr013-") as tmp:
        base = Path(tmp)

        ok_dir = _world(base, "ok")
        proc = _run(REF, ok_dir)
        if proc.returncode != 0:
            raise SystemExit(f"referans calismadi:\n{proc.stderr}")
        beklenen = (ok_dir / "yonelme.txt").read_text(encoding="utf-8").strip()

        checks: list[dict] = [
            {"op": "is_file", "path": "yonelme.txt"},
            {"op": "file_line_count_eq", "path": "yonelme.txt", "value": len(SEHIRLER)},
        ]
        checks += [{"op": "file_contains", "path": "yonelme.txt", "value": v}
                   for v in ZOR_VAKALAR]
        checks.append({"op": "file_content_eq", "path": "yonelme.txt", "value": beklenen})
        toplam = len(checks)

        # -- sozlesme 1 -------------------------------------------------
        gecen, _ = _grade(ok_dir, checks)
        if gecen != toplam:
            raise SystemExit(f"sozlesme 1 ihlal: referans {gecen}/{toplam}")

        # -- sozlesme 2 -------------------------------------------------
        s0_gecen, _ = _grade(_world(base, "s0"), checks)
        if s0_gecen != 0:
            raise SystemExit(f"sozlesme 2 ihlal: S0 bedava {s0_gecen} check veriyor")

        # -- sozlesme 3: naif temeller kismi puan almali -----------------
        print(f"{'referans':32} {toplam}/{toplam}  reward=1.000")
        print(f"{'S0 (cozum yok)':32} 0/{toplam}  reward=0.000")
        for i, (ad, src) in enumerate(NAIF.items()):
            d = _world(base, f"naif{i}")
            p = _run(src, d)
            if p.returncode != 0:
                raise SystemExit(f"naif temel '{ad}' cokmus:\n{p.stderr}")
            n, res = _grade(d, checks)
            if n >= toplam:
                raise SystemExit(f"sozlesme 3 ihlal: '{ad}' de geciyor, tuzak ise yaramiyor")
            # file_contains'in etiketi yalnizca yolu tasiyor, aranan degeri
            # tasimiyor; hangi zor vakanin dustugunu check sirasindan bul.
            dusen = []
            for r in res:
                if r["ok"]:
                    continue
                i_check = int(r["name"].split(":", 1)[0])
                dusen.append(ZOR_VAKALAR[i_check - 2] if 2 <= i_check < 2 + len(ZOR_VAKALAR)
                             else checks[i_check]["op"])
            print(f"{ad:32} {n}/{toplam}  reward={n / toplam:.3f}  dusen: {', '.join(dusen)}")

        print("\nbeklenen cikti:")
        for ln in beklenen.splitlines():
            print(f"  {ln}")

    yazilan = []
    for task_id, prompt, notlar in (
        ("tr-013a-yonelme-kural", PROMPT_KURAL,
         "Unlu uyumu kurali prompt'ta ACIK yazili; model yalnizca uygular. "
         "tr-013b ile ayni veri, ayni check'ler -- tek fark spec'in acikligi. "
         "Zorluk kolu bu ciftte olculuyor."),
        ("tr-013b-yonelme-bilgi", PROMPT_BILGI,
         "Kural prompt'ta YOK; model Turkce dilbilgisini bilmek zorunda. "
         "tr-013a'nin zor varyanti. Kalibrasyonda hangisi banda duserse o "
         "egitime girer."),
    ):
        d = ROOT / "envs" / task_id
        d.mkdir(parents=True, exist_ok=True)
        metin = _task_yaml(task_id, prompt, checks, notlar)
        (d / "task.yaml").write_text(metin, encoding="utf-8")

        # Serilestirme check'leri bozmasin: geri okuyup karsilastir.
        geri = yaml.safe_load(metin)
        if geri["checks"] != checks:
            raise SystemExit(f"{task_id}: yaml turu degistirdi")
        yazilan.append(d / "task.yaml")

    print()
    for p in yazilan:
        print(f"-> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
