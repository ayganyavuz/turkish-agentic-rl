"""The LLM-free gauntlet a task must survive before it enters the dataset.

    python -m verifiable_dataset.terminal.gates --all

Kapilarin hepsi Docker'la calisir, hicbiri model cagirmaz. Uretimde once
bunlar kosar: model cagiran pahali kapilara (cozulebilirlik, bant
kalibrasyonu) yalnizca buradan sag cikan adaylar ulasir.

Elle yazilmis task'lar da birer spec oldugu icin gauntlet onlara karsi
dogrulanabiliyor: 8'i de gecmeliyse, gecmeyen bir sey varsa hata kapida,
uretici modelde degil.
"""
from __future__ import annotations

import argparse
import json
import shlex
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from verifiable_dataset.terminal.derive import derive
from verifiable_dataset.terminal.prompts import denetle as prompt_denetle
from verifiable_dataset.terminal.equiv import run_alt
from verifiable_dataset.terminal.sandbox import (bash_var, docker_available,
                                                 yerel_kullan)
from verifiable_dataset.terminal.seeds import tools_used, uses_tool
from verifiable_dataset.terminal.task import Task

TURKCE_HARFLER = set("çğıöşüÇĞİÖŞÜ")


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    @property
    def mark(self) -> str:
        if self.skipped:
            return "ATLA"
        return "OK  " if self.ok else "DUSTU"


def _norm(checks: list[dict]) -> str:
    return json.dumps(sorted((json.dumps(c, sort_keys=True, ensure_ascii=False)
                              for c in checks)), ensure_ascii=False)


# -- kapilar ----------------------------------------------------------

def gate_spec_complete(task: Task, spec_only: bool = False) -> GateResult:
    eksik = []
    if not task.reference_solution.strip():
        eksik.append("reference_solution")
    if not task.goal_tr.strip():
        eksik.append("goal_tr")
    # spec asamasinda prompt heniz yazilmadi; creative writing ayri asama.
    if not spec_only:
        if not task.prompt.strip():
            eksik.append("prompt")
        if not task.checks and not task.checker.exists():
            eksik.append("checks (derive --write ile yazilir)")
    return GateResult("spec butunlugu", not eksik,
                      f"eksik alanlar: {', '.join(eksik)}" if eksik else "")


def gate_data_ascii(task: Task) -> GateResult:
    """Veri ASCII, prompt tam Turkce -- proje karari.

    Dunyanin icerigi ASCII kalirsa siralama ve kodlama surprizleri
    task'a degil, bilincli olarak secilmis bir aileye ait olur.
    """
    kirli = [ch for ch in task.setup if ord(ch) > 127]
    return GateResult("veri ASCII", not kirli,
                      f"setup icinde ASCII disi karakter: {sorted(set(kirli))[:8]}")


def gate_prompt_turkce(task: Task) -> GateResult:
    if task.metadata.get("language") != "tr":
        return GateResult("prompt turkce", True, skipped=True, detail="tr degil")
    var = TURKCE_HARFLER & set(task.prompt)
    return GateResult("prompt turkce", bool(var),
                      "prompt'ta hic Turkce karakter yok -- ASCII'lestirilmis olabilir")


def gate_prompt_covers_outputs(task: Task) -> GateResult:
    """Prompt, check'lerin sart kostugu her artefakti adiyla anmali.

    Check'ler prompt'tan once donuyor. Prompt bir cikti dosyasini hic
    anmazsa gorev cozulemez hale gelir: yardimci adini bilemedigi bir
    dosyayi uretemez, ama check onu istiyordur. Ayni denetim cozum
    sizintisina da bakiyor -- referansin komutlari prompt'a girmisse gorev
    hedef ayristirma olmaktan cikip komut kopyalamaya doner.
    """
    if not task.prompt.strip():
        return GateResult("prompt kapsami", True, skipped=True, detail="prompt yok")
    sorunlar = prompt_denetle(task, task.prompt)
    return GateResult("prompt kapsami", not sorunlar, "; ".join(sorunlar[:3]))


def gate_tool_conformance(task: Task, eksik_araclar: list[str]) -> GateResult:
    """Seed'in araclarinin hepsi kullanilmis mi, ve cagrilan her arac var mi.

    Seed araclari zorunlu: ogretmek istedigimiz yetenek yuzeyini suren sey
    bu. `sed+tr` istenmisse cozum ikisini de kullanmali, yoksa grid dolu
    gorunur ama cesitlilik sahte olur.

    Seed DISINDAKI araclari kullanmak serbest. Gercek bir cozum zaten
    mkdir, cat, diff gibi araclara uzanir; bunlari cezalandirmak dogru
    cozumleri eler. Zaten seed araclarinin hepsi sart kosuldugu icin
    "her seyi tek satir python ile coz" kacamagi da kendiliginden kapali.

    Olmayan bir araci cagirmak ise her zaman hata. Statik aramak yanlis
    alarm uretirdi (degisken adi, heredoc icerigi), bu yuzden kabugun kendi
    "command not found" mesajina bakiliyor -- `|| true` ile yutulmus olsa
    bile stderr'de duruyor.
    """
    sorun = []
    if eksik_araclar:
        sorun.append(f"imajda olmayan arac cagrildi: {eksik_araclar}")

    seed = task.metadata.get("seed")
    # Kod ailesinde seed araci her zaman python3, ama referans cozum onu
    # komut konumunda cagirmaz -- dosyayi heredoc ile yazar. "Kullanilmali"
    # kurali burada her adayi haksiz yere duserirdi; yetenek yuzeyini kod
    # ailesinde suren sey arac degil hedefin kendisi.
    if seed and seed.get("aile") == "kod":
        return GateResult("arac uygunlugu", not sorun, "; ".join(sorun),
                          skipped=not sorun)

    if seed:
        istenen = seed.get("tools", [])
        eksik = [t for t in istenen if not uses_tool(task.reference_solution, t)]
        if eksik:
            sorun.append(f"kullanilmayan seed araclari: {eksik}")
    elif not sorun:
        return GateResult("arac uygunlugu", True, skipped=True, detail="seed yok")

    return GateResult("arac uygunlugu", not sorun, "; ".join(sorun))


def gate_determinism(task: Task) -> tuple[GateResult, "object | None", list[str]]:
    """Ayni referans iki temiz dunyada ayni check'leri uretmeli.

    Timestamp, $RANDOM, hash sirasi ve locale farklari burada yakalanir --
    uretilmis task'larda elle yazilanlardan cok daha sik sizarlar.
    """
    first = derive(task)
    eksik = list(first.eksik_araclar)
    if first.error:
        return (GateResult("determinizm", False,
                           f"turetme basarisiz: {first.error}"), None, eksik)
    second = derive(task)
    if second.error:
        return (GateResult("determinizm", False,
                           f"ikinci kosu: {second.error}"), None, eksik)
    if _norm(first.checks) != _norm(second.checks):
        a = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in first.checks}
        b = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in second.checks}
        fark = sorted(a ^ b)[:3]
        return (GateResult("determinizm", False,
                           f"iki kosu farkli check uretti: {fark}"), None, eksik)
    return (GateResult("determinizm", True,
                       f"iki kosuda ayni {first.total} check"), first, eksik)


def gate_discriminates(task: Task, rep) -> GateResult:
    """Check'ler cozulmus dunyayi cozulmemisten ayirt edebiliyor mu.

    Sayilari basmak teshis icin yetmiyordu: "B 1/1, A 1/1" goren spec
    yazari neyi duzeltecegini anlamiyor ve ayni hatayi tekrarliyor. Iki
    basarisizlik sekli birbirinden cok farkli, o yuzden ayri soyleniyor.
    """
    if rep is None:
        return GateResult("ayirt etme", False, "turetme yapilamadi")
    ok = rep.ref_passed == rep.total and rep.init_passed < rep.total
    sayilar = (f"referans {rep.ref_passed}/{rep.total}, "
               f"baslangic {rep.init_passed}/{rep.total}")
    if rep.ref_passed < rep.total:
        aciklama = f"referans cozum kendi check'lerini gecmiyor ({sayilar})"
    elif rep.init_passed >= rep.total:
        aciklama = (f"gorev baslangicta zaten cozulmus ({sayilar}): SETUP "
                    f"dunyasi check'leri geciyor. Kod gorevlerinde bunun "
                    f"olagan sebebi SETUP'a birakilan 'hata'nin gercek bir "
                    f"hata olmamasi ya da testin o hatayi hic yoklamamasi.")
    else:
        aciklama = sayilar
    return GateResult("ayirt etme", ok, aciklama)


def gate_output_kinds(task: Task, rep) -> GateResult:
    """Bildirilen cikti turu, referansin gercekten urettigiyle uyusmali.

    Spec `kind: json` deyip gecersiz JSON uretebiliyor. Bu sessizce byte
    esitligine dusuyor ve task tersine doner: gecerli JSON yazan DOGRU
    cozum sifir alir, cunku beklenen deger bozuk metnin kendisidir.
    """
    if rep is None:
        return GateResult("cikti turu", False, "turetme yapilamadi")
    if not task.outputs:
        return GateResult("cikti turu", True, skipped=True,
                          detail="outputs bildirilmemis")
    return GateResult("cikti turu", not rep.sorunlar, "; ".join(rep.sorunlar[:3]))


def ucuz_hile_betigi(checks: list[dict]) -> str:
    """Check'lerde adi gecen her yolu bos olarak yaratan tembel bir betik.

    Hicbir is yapmiyor: dizinleri aciyor, dosyalari bos olarak dokunuyor,
    silinmesi gerekenleri siliyor. Bu betik check'leri geciyorsa task'in
    olcecek bir seyi yok demektir.
    """
    satirlar: list[str] = []
    for c in checks:
        op, path = c.get("op"), c.get("path")
        if not path:
            continue
        q = shlex.quote(path)
        if op in {"is_dir", "dir_entries_eq"}:
            satirlar.append(f"mkdir -p {q}")
        elif op == "not_exists":
            satirlar.append(f"rm -rf {q}")
        else:
            satirlar.append(f'mkdir -p "$(dirname {q})" && : > {q}')
    return "\n".join(dict.fromkeys(satirlar)) or "true"


def gate_cheap_hack(task: Task, rep) -> GateResult:
    """Hicbir is yapmayan bir cozum gecmemeli.

    Uretilen bir task'in referansi bos bir dosya birakmisti; turetilen check
    de "icerik bos olsun" oldu ve `touch sonuc.txt` tam puan aldi. Bant
    filtresi bunu ancak 16 ajan rollout'u harcayarak fark ederdi. Burada
    tek konteynerde, saniyeler icinde anlasiliyor.
    """
    if rep is None or not rep.checks:
        return GateResult("ucuz hile", False, "turetilen check yok")
    r = run_alt(task, {"name": "ucuz hile", "script": ucuz_hile_betigi(rep.checks)},
                rep.checks)
    return GateResult(
        "ucuz hile", not r.ok,
        f"is yapmayan betik {r.derived_passed}/{r.derived_total} check geciyor -- "
        f"check'lerin olctugu bir sey yok",
    )


def gate_ara_durum(task: Task, rep) -> GateResult:
    """Zincirin gercekten zincir oldugunu dogrula.

    Uretici modelden "birbirine bagli 2-3 hata" istemek yetmiyor: model
    pekala yan yana duran BAGIMSIZ hatalar yazip ayni sozu tutmus gibi
    gorunur. Fark ajan icin buyuk -- bagimsiz hatalar paralel, zincirli
    hatalar sirali kesif gerektirir -- ama task.yaml'a bakarak anlasilmaz.

    Olcum spec'in ARA_ADIM blogunu kullaniyor: yalnizca ILK hatayi gideren
    dunyada
      * check'lerin HEPSI gecerse zincir yok (ilk duzeltme gorevi bitiriyor),
      * SETUP dunyasindan DAHA FAZLA check gecmiyorsa ara adim bir sey
        ilerletmemis (iki hata ayni check'i tutuyor ya da ARA_ADIM etkisiz).
    Ikisi de degilse: ilerleme var ama is bitmemis -- zincir budur.
    """
    if rep is None or not rep.checks:
        return GateResult("ara durum", False, "turetilen check yok")

    zincir = int((task.metadata.get("seed", {}) or {})
                 .get("metadata", {}).get("zincir", 0) or 0)
    if not task.ara_adim.strip():
        if zincir:
            return GateResult("ara durum", False,
                              f"seed zincir={zincir} istiyor ama ARA_ADIM bolumu yok")
        return GateResult("ara durum", True, skipped=True, detail="zincirsiz gorev")

    # Tek check ile ilerleme olculemez: ara adim ya hepsini gecirir ya
    # hicbirini. Bu bir spec kusuru, zincir kusuru degil -- ayri soyle.
    if len(rep.checks) < 2 and zincir:
        return GateResult("ara durum", False,
                          f"zincir={zincir} icin tek check yetmez; regresyon "
                          f"check'i eksik (OUTPUTS'ta en az iki `run` olmali)")

    # "true" = hicbir sey yapma; SETUP dunyasinin taban olcumu.
    taban = run_alt(task, {"name": "setup", "script": "true"}, rep.checks)
    ara = run_alt(task, {"name": "ara adim", "script": task.ara_adim}, rep.checks)

    if ara.ok:
        return GateResult("ara durum", False,
                          f"ilk duzeltme tek basina {ara.derived_passed}/"
                          f"{ara.derived_total} check'i geciriyor -- zincir yok, "
                          f"tek hata var")
    if ara.derived_passed <= taban.derived_passed:
        return GateResult("ara durum", False,
                          f"ara adim ilerleme saglamiyor (SETUP "
                          f"{taban.derived_passed}/{taban.derived_total} -> ara "
                          f"{ara.derived_passed}/{ara.derived_total}) -- hatalar "
                          f"zincirli degil ya da ARA_ADIM etkisiz")
    return GateResult("ara durum", True,
                      f"SETUP {taban.derived_passed}/{taban.derived_total} -> "
                      f"ara {ara.derived_passed}/{ara.derived_total} -> "
                      f"referans {ara.derived_total}/{ara.derived_total}")


YAPISAL_BELIRTEC = {
    # Duz alt dize, duzenli ifade degil: desende tek bir kacis karakteri
    # bozulursa kapi sessizce yanlis olcer ve dogru adaylari eler.
    # Yalnizca GUVENILIR belirteci olan kaliplar burada; "closure-fabrika"
    # ve "ozyineleme" saglam ayirt edilemedigi icin atlaniyor.
    "durumlu-sinif": ("class ",),
    "generator": ("yield",),
    "context-manager": ("__enter__",),
    "dataclass-dogrulama": ("@dataclass", "__post_init__"),
    "iterator-protokolu": ("__iter__", "__next__"),
    "istisna-hiyerarsisi": ("Error(", "Exception)", "Error)"),
    "abc-protokol": ("ABC", "abstractmethod"),
    "operator-asiri-yukleme": ("__eq__", "__lt__", "__add__", "__repr__"),
    "dekorator": ("@",),
}


def gate_yapisal_uyum(task: Task) -> GateResult:
    """Seed bir Python kalibi istediyse kod o kalibi GERCEKTEN kullanmali.

    Yapisal eksenler korpusun cesitliligini tasiyor, ama talimat olmak
    yetmiyor: pilotta 19 gorevin 4'unde model kalibi yoksayip duz fonksiyon
    yazdi. Cesitlilik "istendi" ile degil "olculdu" ile artar.
    """
    seed = (task.metadata.get("seed") or {})
    if seed.get("aile") != "kod":
        return GateResult("yapisal uyum", True, skipped=True, detail="kod ailesi degil")
    kalip = (seed.get("metadata") or {}).get("kalip", "")
    desen = YAPISAL_BELIRTEC.get(kalip)
    if not desen:
        return GateResult("yapisal uyum", True, skipped=True,
                          detail=f"'{kalip}' guvenilir belirtecle olculemiyor")
    kod = (task.setup or "") + "\n" + (task.reference_solution or "")
    if any(t in kod for t in desen):
        return GateResult("yapisal uyum", True, f"kalip '{kalip}' kullanilmis")
    return GateResult("yapisal uyum", False,
                      f"seed '{kalip}' istiyor ama kodda izi yok")


def gate_equivalence(task: Task, checks: list[dict] | None) -> GateResult:
    if not task.alt_solutions:
        return GateResult("denklik", True, skipped=True, detail="alt_solutions yok")
    if checks is None:
        return GateResult("denklik", False, "turetilen check yok")
    dusen = []
    for alt in task.alt_solutions:
        r = run_alt(task, alt, checks)
        if not r.ok:
            dusen.append(f"{r.name} ({r.derived_passed}/{r.derived_total})")
    return GateResult("denklik", not dusen,
                      f"{len(task.alt_solutions) - len(dusen)}/{len(task.alt_solutions)} "
                      f"alternatif geciyor" + (f"; dusen: {dusen}" if dusen else ""))


def run_gauntlet(task: Task, spec_only: bool = False
                 ) -> tuple[list[GateResult], "object | None"]:
    """spec_only: prompt heniz yazilmamis adaylar icin prompt kapilarini atla."""
    results = [
        gate_spec_complete(task, spec_only),
        gate_data_ascii(task),
    ]
    if not spec_only:
        results.append(gate_prompt_turkce(task))
        results.append(gate_prompt_covers_outputs(task))
    # Arac uygunlugu artik referansin stderr'ine de bakiyor, o yuzden
    # turetmeden sonra calisiyor.
    det, rep, eksik_araclar = gate_determinism(task)
    results.append(gate_tool_conformance(task, eksik_araclar))
    results.append(det)
    results.append(gate_discriminates(task, rep))
    results.append(gate_output_kinds(task, rep))
    results.append(gate_cheap_hack(task, rep))
    results.append(gate_ara_durum(task, rep))
    results.append(gate_yapisal_uyum(task))
    results.append(gate_equivalence(task, rep.checks if rep else None))
    return results, rep


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sandbox", choices=["docker", "yerel"],
                        default="docker",
                        help="yerel = Docker'siz (Colab); izolasyon yok")
    parser.add_argument("--envs", default="envs")
    parser.add_argument("--spec-only", action="store_true",
                        help="prompt heniz yazilmamis adaylar icin prompt kapilarini atla")
    args = parser.parse_args()

    if args.sandbox == "yerel":
        yerel_kullan(True)
        ok, info = bash_var()
        if not ok:
            print(f"yerel sandbox icin bash gerekli: {info}")
            return 1
    else:
        ok, info = docker_available()
        if not ok:
            print(f"Docker daemon'a ulasilamiyor: {info}")
            return 1

    if args.all:
        task_dirs = sorted(p.parent for p in Path(args.envs).glob("*/task.yaml")
                           if not p.parent.name.startswith("_"))
    elif args.task:
        task_dirs = [Path(args.task)]
    else:
        print("--task ya da --all ver")
        return 2

    gecen = 0
    for task_dir in task_dirs:
        task = Task.load(task_dir)
        print(task.id)
        results, _ = run_gauntlet(task, spec_only=args.spec_only)
        for r in results:
            # Detay yalnizca aciklayici oldugu yerde: gecen bir kapinin
            # hata mesajini basmak raporu okunmaz hale getiriyordu.
            detail = f"   {r.detail}" if r.detail and (r.skipped or not r.ok) else ""
            print(f"  [{r.mark}] {r.name:18}{detail}")
        if all(r.ok for r in results):
            gecen += 1
        else:
            dusen = [r.name for r in results if not r.ok]
            print(f"  --> REDDEDILDI: {', '.join(dusen)}")
        print()

    print(f"GAUNTLET: {gecen}/{len(task_dirs)} task butun kapilardan geciyor")
    return 0 if gecen == len(task_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
