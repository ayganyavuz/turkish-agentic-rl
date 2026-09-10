"""Generate task specs from seeds and put them through the LLM-free gauntlet.

    python -m verifiable_dataset.terminal.generate --n 5 \
        --model mistral-medium-latest \
        --base-url https://api.mistral.ai/v1

Model yalnizca spec yazar: baslangic dunyasi, referans cozum, hedef
artefaktlar ve tek cumlelik amac. Check'leri yazmaz -- onlari derive.py
cozumu gercekten calistirarak cikarir. Prompt'u da yazmaz; o ayri bir
asama. Hicbir model kendi isini kendi notlandirmiyor.

Bu asamada uretilen task'lar heniz eksik: prompt'lari yok, rakip cozumleri
yok. Yani egitime hazir degiller, ama spec uretiminin calisip calismadigini
gosterirler.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from verifiable_dataset.terminal.derive import checks_yaz
from verifiable_dataset.terminal.gates import run_gauntlet
from verifiable_dataset.terminal.sandbox import docker_available
from verifiable_dataset.terminal.seeds import (Seed, sample, sample_kod,
                                              tools_of_image)
from verifiable_dataset.terminal.llm import make_client, preflight, resolve_model
from verifiable_dataset.terminal.task import Task

PIPELINE_VERSION = "spec-1"

# YAML yerine acik ayrac kullaniyoruz: modeller cok satirli bash'i blok
# skalarina dogru girintilemekte sik hata yapiyor, ayraclar affedici.
SECTION_RE = re.compile(
    r"^###\s+(SETUP|GOAL_TR|REFERENCE|OUTPUTS|ARA_ADIM)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$", re.MULTILINE)

SPEC_TALIMAT = """\
Sen dogrulanabilir terminal gorevleri tasarlayan bir muhendissin. Sana bir
seed verilecek; buna uyan bir gorev SPEC'i yazacaksin.

Yazacagin dort bolum var, baska hicbir sey yazma:

### SETUP
Gorevin basladigi dunyayi kuran bash komutlari. /workspace icinde calisir.

### GOAL_TR
Amacin tek cumlelik, susuz, kesin Turkce ifadesi. Bu bir gorev metni degil,
bir tanim: hangi dosyadan ne uretilecegini birakmadan soyler.

### REFERENCE
Gorevi cozen bash komutlari. /workspace icinde calisir.

### OUTPUTS
Hedefin parcasi olan artefaktlar, YAML listesi:
- path: <yol>
  kind: file | lines | json | dir | program
  run: <komut>        # yalnizca kind: program icin

KURALLAR (hepsi zorunlu):
1. Referans cozum GERCEKTEN calismali. Calismayan spec dogrudan elenir.
2. Seed'deki araclarin hepsi referans cozumde kullanilmali.
3. Seed'de python3 yoksa python3 KULLANMA. Tek satirlik python kacamagi yok.
4. Deterministik ol: date, $RANDOM, ag erisimi, `ls` siralamasina bagimlilik
   yok. Ayni referans iki kez calisinca ayni sonucu vermeli.
5. Veri ASCII olmali: dosya adlarinda ve dosya iceriklerinde Turkce karakter
   kullanma. Yalnizca GOAL_TR tam Turkce yazilir.
6. SETUP verisi kisa tutulsun (en fazla 40 satir), seed'de bukulme=olcek
   degilse.
7. OUTPUTS yalnizca hedefin parcasi olan artefaktlari listeler. Referansin
   yol boyunca yazdigi ara betikleri (orn. analiz.py) LISTELEME -- onlar
   sart kosulursa tek satirda cozen dogru cozumler haksiz yere duser.
8. Gorev baslangicta cozulmus olmamali: SETUP hedef ciktiyi olusturmasin.
9. Bildirdigin kind gercekle uyusmali. kind: json dediysen referansin
   urettigi dosya GECERLI JSON olmali -- elle string birlestirerek JSON
   uydurma, virguller ve parantezler dogru olsun. Emin degilsen kind: file
   kullan ve duz metin uret.

ORNEK (seed: araclar=[grep], konu=sunucu-loglari, hedef=ozetle,
kaynak=tek-dosya, cikti=tek-sayi):

### SETUP
cat > kayitlar.log <<'LOG'
BILGI sistem basladi
HATA disk dolu
BILGI kullanici giris yapti
HATA baglanti koptu
UYARI bellek az
HATA zaman asimi
LOG

### GOAL_TR
kayitlar.log dosyasinda HATA gecen satir sayisini sonuc.txt'ye sadece sayi
olarak yaz

### REFERENCE
grep -c 'HATA' kayitlar.log > sonuc.txt

### OUTPUTS
- path: sonuc.txt
  kind: file
"""



# -- kod ailesi -------------------------------------------------------
# Ayri bir talimat, cunku kabuk talimatinin yarisi (arac demeti, girdi
# bicimi, kesif) burada ya anlamsiz ya da yanlis yonlendirici. Asil fark
# dogrulamada: bir programin dogrulugu diskteki metninde degil calisinca
# ne urettiginde, o yuzden `kind: program` + `run` zorunlu.

SPEC_TALIMAT_KOD = 'Sen dogrulanabilir Python programlama gorevleri tasarlayan bir muhendissin.\nSana bir seed verilecek; buna uyan bir gorev SPEC\'i yazacaksin.\n\nYazacagin dort bolum var, baska hicbir sey yazma:\n\n### SETUP\nGorevin basladigi dunyayi kuran bash komutlari. /workspace icinde calisir.\nKaynak dosyalari burada olusturulur.\n\n### GOAL_TR\nAmacin tek cumlelik, susuz, kesin Turkce ifadesi. Hangi dosyanin\nyazilacagini/duzeltilecegini birakmadan soyler.\n\n### REFERENCE\nGorevi cozen bash komutlari. /workspace icinde calisir. Genellikle dogru\nPython dosyasini heredoc ile yazar.\n\n### OUTPUTS\n- path: <ajanin yazacagi/duzeltecegi .py dosyasi>\n  kind: program\n  run: <dogrulama komutu>\n\nKURALLAR (hepsi zorunlu):\n1. Referans cozum GERCEKTEN calismali; `run` komutu referans dunyasinda\n   exit 0 dondurmeli. Calismayan spec elenir.\n2. `run` komutu testi ICINDE tasimali (python3 -c \'...\'), testi calisma\n   dizinindeki bir dosyadan OKUMAMALI. Ajan dosya sistemini degistirerek\n   notunu yukseltememeli.\n3. `run` komutu BASARILI durumda SABIT VE BOS OLMAYAN bir metin basmali\n   (orn. sonunda print("TAMAM")). Yalnizca assert kullanip hicbir sey\n   basmayan bir test ISE YARAMAZ: assert patlayinca da stdout bos kalir,\n   yani bozuk dunya ile dogru dunya ayni ciktiyi verir ve check hicbir\n   seyi olcmez.\n4. `run` ciktisi deterministik olmali: zaman, rastgelelik, bellek adresi,\n   kume/sozluk siralamasina bagimlilik yok.\n5. Yalnizca Python 3 standart kutuphanesi. pytest, numpy KURULU DEGIL;\n   test icin `assert` ya da `unittest` kullan.\n6. Kod ve tanimlayicilar Ingilizce; yalnizca GOAL_TR Turkce yazilir.\n   Dosya adlari ve dosya icerikleri ASCII olmali.\n7. Gorev baslangicta cozulmus olmamali: `run` komutu SETUP\'tan hemen\n   sonra BASARISIZ olmali, referanstan sonra basarili.\n8. SETUP kisa tutulsun; toplam 80 satiri gecmesin.\n9. EN SIK YAPILAN HATA: SETUP\'taki kod ile REFERENCE\'taki kod ayni\n   oluyor. hata-bul gorevlerinde SETUP\'a biraktigin kod GERCEKTEN\n   YANLIS olmali ve `run` testi o yanlisligi YAKALAMALI. Spec\'i\n   yazmadan once kendine sor: bu test bozuk kodda hangi satirda\n   patlar? Cevabin yoksa hata birakmamissin demektir.\n10. `run` degeri TEK SATIR olmali -- OUTPUTS bir YAML blogu, coka\n   satirli bir komut onu bozar. Uzun testi \';\' ile tek satirda birlestir.\n11. Paket/modul adi olarak Python standart kutuphanesindeki adlari\n   KULLANMA (calendar, json, csv, time, string, types, code...).\n   Golgeleme sessiz ve teshisi zor hatalar uretir.\n12. YOLLAR GORELI olmali. `/workspace` yazma -- ne SETUP\'ta, ne\n   REFERENCE\'ta, ne `run` icinde. Gorev baska bir calisma dizininde de\n   kosabilmeli. (`python3 -c` calisma dizinini zaten sys.path\'e koyar,\n   sys.path.insert\'e gerek yok.)\n13. `run` komutu calisma dizinindeki HICBIR .py dosyasini import ederek\n   test verisi almamali (orn. `from test_data import CART` YASAK).\n   Ajan o dosyayi degistirip testi gecebilirdi. Test verisi komutun\n   kendi icinde, satir ici yazilir; yalnizca ajanin duzeltecegi/yazacagi\n   hedef modul import edilir.\n\nHEDEFE GORE SEKIL (bunlar tarif, ilke degil -- birebir uygula):\n\nhata-bul   : SETUP birbirini cagiran 2-3 modul yazar ve iclerinden BIRINDE\n             seed\'deki tipte SESSIZ bir hata birakir: kod calisir ama YANLIS\n             sonuc uretir. REFERENCE ayni dosyayi DOGRU haliyle yeniden\n             yazar. SETUP\'taki bozuk satir ile REFERENCE\'taki dogru satir\n             BIRBIRINDEN FARKLI OLMALI -- iki metni yan yana koy ve farki\n             gordugunden emin ol.\nyarisma    : SETUP girdi dosyalarini ve (cok-dosya ise) yardimci modulleri\n             yazar. SETUP hedef .py dosyasini HIC OLUSTURMAZ; ajan onu\n             sifirdan yazar. REFERENCE o dosyayi heredoc ile yazar.\nveri-yapisi: SETUP iskeleti yazar ama HER METODUN GOVDESI\n             `raise NotImplementedError` olmali -- calisan hicbir\n             gerceklestirme birakma. (cok-dosya ise iskeleti kullanan\n             modulleri de yazar.) REFERENCE dosyayi tam gerceklestirmeyle\n             yeniden yazar.\n\nHER UC HEDEFTE DE: `run` komutu SETUP\'tan hemen sonra calistirildiginda\nBASARISIZ olmali. Spec\'i yazdiktan sonra kendine sor: SETUP dunyasinda bu\nkomut neden patlar? Cevabin yoksa gorev bastan cozulmus demektir ve elenir.\n\nORNEK (seed: hedef=hata-bul, ayrinti=off-by-one, yapi=cok-dosya, konu=siparis-kayitlari):\n\n### SETUP\nmkdir -p shop\ncat > shop/orders.py <<\'ORDERS\'\ndef parse_orders(text):\n    rows = []\n    for line in text.strip().splitlines():\n        name, qty, price = line.split(\',\')\n        rows.append((name, int(qty), float(price)))\n    return rows\nORDERS\ncat > shop/report.py <<\'REPORT\'\nfrom shop.orders import parse_orders\n\n\ndef top_orders(text, k):\n    rows = parse_orders(text)\n    rows.sort(key=lambda r: r[1] * r[2], reverse=True)\n    return [r[0] for r in rows[:k - 1]]\nREPORT\ncat > data.csv <<\'CSV\'\nwidget,3,4.0\ngadget,1,50.0\nbolt,10,0.5\nCSV\n\n### GOAL_TR\nshop/report.py icindeki top_orders fonksiyonunu, en yuksek tutarli k\nsiparisin adini dogru dondurecek bicimde duzelt\n\n### REFERENCE\ncat > shop/report.py <<\'REPORT\'\nfrom shop.orders import parse_orders\n\n\ndef top_orders(text, k):\n    rows = parse_orders(text)\n    rows.sort(key=lambda r: r[1] * r[2], reverse=True)\n    return [r[0] for r in rows[:k]]\nREPORT\n\n### OUTPUTS\n- path: shop/report.py\n  kind: program\n  run: python3 -c \'from shop.report import top_orders as f; t=open("data.csv").read(); assert f(t,2)==["gadget","widget"], f(t,2); assert f(t,1)==["gadget"]; assert f(t,3)==["gadget","widget","bolt"]; print("TAMAM")\'\n'


def seed_brief_kod(seed: Seed) -> str:
    """Kod ailesi icin seed ozeti.

    Ayrinti ekseni (hata tipi / algoritmik kalip / veri yapisi) yon
    veriyor ama baglamiyor: ayni yonde birbirinden farkli gorevler
    cikmasi icin ayrintiyi modelin secmesi isteniyor.
    """
    ayrinti = seed.metadata.get("ayrinti", "")
    etiket = {"hata-bul": "hata tipi", "yarisma": "algoritmik kalip",
              "veri-yapisi": "veri yapisi"}.get(seed.hedef, "ayrinti")
    return (
        "Yalnizca Python 3 standart kutuphanesi var (pytest, numpy YOK).\n\n"
        "Seed:\n"
        f"  hedef     : {seed.hedef}\n"
        f"  {etiket:10}: {ayrinti}\n"
        f"  konu      : {seed.konu}\n"
        f"  yapi      : {seed.kaynak}\n"
        f"  py kalibi : {seed.metadata.get('kalip', 'duz-fonksiyon')}\n"
        f"  veri akisi: {seed.metadata.get('akis', 'bellek-ici')}\n"
        f"  topoloji  : {seed.metadata.get('topoloji', 'iki-modul-zincir')}\n"
        f"  zorluk    : {seed.bukulme}\n"
        f"  uslup     : {seed.uslup}\n"
        f"  adim      : {seed.adim} (referans cozum kabaca bu kadar adim surmeli)\n\n"
        f"'{ayrinti}' verilen yon; ayrintiyi sen sec ve alisilmis ornekten "
        "kacin -- ayni yonde birbirinden farkli gorevler uretilecek.\n\n"
        "YAPISAL EKSENLER BAGLAYICIDIR (ayrinti gibi serbest degil):\n"
        "  py kalibi  : kodun SEKLI. durumlu-sinif istendiyse cozum bir sinif ve\n"
        "               metotlari olmali; generator istendiyse yield kullanmali;\n"
        "               context-manager __enter__/__exit__ tanimlamali. Duz\n"
        "               fonksiyon yazip gecme.\n"
        "  veri akisi : verinin NEREDEN geldigi. dosya-okuma ise SETUP veri\n"
        "               dosyasi yazar ve kod onu acar; stdin-boru ise kod\n"
        "               stdin'den okur; ic-ice-yapi ise veri ic ice sozluk/liste.\n"
        "  topoloji   : modullerin BAGLANTISI. uc-modul-zincir = A'yi B, B'yi C\n"
        "               cagirir; elmas = B ve C ortak D yi kullanir;\n"
        "               paket-alt-paket = pkg/ ve pkg/alt/ dizinleri.\n"
        "\n"
        "Bu uc eksen korpusun yapisal cesitliligini tasiyor; birine uymayan\n"
        "aday elenir.\n"
        "\n"
        "Bu seed'e uyan bir SPEC yaz.")

def seed_brief(seed: Seed) -> str:
    kurulu = ", ".join(tools_of_image(seed.image))
    return f"""\
Imajda kurulu araclar (BASKASINI CAGIRMA -- jq, jshon, gawk, git, sqlite3,
zip gibi araclar YOK):
{kurulu}

Seed:
  araclar   : {', '.join(seed.tools)}   (hepsi referansta kullanilmali)
  konu      : {seed.konu}
  hedef     : {seed.hedef}
  girdi     : {seed.girdi}
  kaynak    : {seed.kaynak}
  cikti     : {seed.cikti}
  kesif     : {seed.kesif}
  direnc    : {seed.direnc}
  bukulme   : {seed.bukulme}
  adim      : {seed.adim} (referans cozum kabaca bu kadar adim surmeli)

Bu seed'e uyan bir SPEC yaz."""



# -- karmasiklik talimatlari ------------------------------------------
# Kabul oranini yukselten sey soyut ilke degil mekanik tarif oldu
# (bkz. VERI-URETIMI.md, kod ailesi 1/6 -> %40). Buradaki metinler de
# "sunu yap" diye yaziliyor, "sunu hedefle" diye degil.

KARMASIKLIK_TALIMAT = {
    "duz": "",
    "orta": """
KARMASIKLIK: ORTA -- bu gorev cok adimli olmali.

ONCE BUNU OKU -- USTTEKI BOLUM LISTESI BU GOREV ICIN GECERSIZ.
Bu gorevde DORT degil BES bolum yazacaksin, tam bu sirayla:
### SETUP, ### GOAL_TR, ### REFERENCE, ### ARA_ADIM, ### OUTPUTS
ARA_ADIM'i unutmak adayin ELENMESIDIR; en sik yapilan hata budur.

14. ZINCIRLEME HATA (2 hata). SETUP birbirini cagiran modullere IKI hata
    biraksin ve ikincisi birincisi giderilmeden GORUNMESIN: 1. hata
    ustteki fonksiyonu erken dondurdugu/patlatti icin 2. hatanin bulundugu
    kod yoluna hic girilmiyor olsun. Iki hata AYRI DOSYADA olsun.
    Kendine sor: 1. hatayi duzeltince test hangi YENI mesajla dusuyor?
    Cevabin yoksa iki bagimsiz hata yazmissin, zincir degil.
15. ARA_ADIM bolumu ZORUNLU. Yalnizca 1. hatayi gideren bash komutlarini
    yaz (2. dosyaya DOKUNMA). SETUP + ARA_ADIM dunyasinda `run` HALA
    DUSMELI -- ama farkli bir sebeple.
16. REGRESYON KORUMASI. SETUP, ZATEN DOGRU calisan ikinci bir davranis
    da icersin ve OUTPUTS'a bunu olcen AYRI bir `run` check'i koy. Bu
    check SETUP dunyasinda GECMELI ve referanstan sonra da GECMELI.
    Boylece "dosyayi sil bastan yaz" kestirmesi eskiyi bozarak dusuyor.
17. BAGIMLI MODULLER. Moduller birbirini gercekten cagirsin (A'yi B,
    B'yi C kullansin); yan yana duran bagimsiz dosyalar yazma.

OUTPUTS BU SEKILDE OLMALI -- EN AZ IKI `run` GIRDISI (birebir uygula):

### OUTPUTS
- path: pkg/report.py
  kind: program
  run: python3 -c 'from pkg.report import summary as f; assert f("a,1")=="a=1", f("a,1"); print("TAMAM")'
- path: pkg/report.py
  kind: program
  run: python3 -c 'from pkg.report import fmt as f; assert f(2)=="2.00"; print("REGRESYON-TAMAM")'

Birinci girdi zincirin sonunu olcer (SETUP dunyasinda DUSER).
Ikinci girdi regresyon korumasidir (SETUP dunyasinda da GECER).
TEK `run` girdisi yazarsan aday ELENIR: tek check ile ara adimin ilerleme
saglayip saglamadigi olculemez.
""",
    "derin": """
KARMASIKLIK: DERIN -- bu gorev uzun ve cok adimli olmali.

ONCE BUNU OKU -- USTTEKI BOLUM LISTESI BU GOREV ICIN GECERSIZ.
Bu gorevde DORT degil BES bolum yazacaksin, tam bu sirayla:
### SETUP, ### GOAL_TR, ### REFERENCE, ### ARA_ADIM, ### OUTPUTS
ARA_ADIM'i unutmak adayin ELENMESIDIR; en sik yapilan hata budur.

14. ZINCIRLEME HATA (3 hata). SETUP birbirini cagiran modullere UC hata
    biraksin; her biri bir oncekisi giderilmeden GORUNMESIN. Uc hata UC
    AYRI DOSYADA olsun. Kendine sor: 1. hatayi duzeltince test hangi YENI
    mesajla duser, 2.'yi duzeltince hangi mesajla? Iki cevabin da yoksa
    bagimsiz hatalar yazmissin, zincir degil.
15. ARA_ADIM bolumu ZORUNLU. Yalnizca 1. hatayi gideren bash komutlarini
    yaz (diger dosyalara DOKUNMA). SETUP + ARA_ADIM dunyasinda `run`
    HALA DUSMELI -- ama farkli bir sebeple.
16. REGRESYON KORUMASI. SETUP, ZATEN DOGRU calisan ikinci bir davranis
    da icersin ve OUTPUTS'a bunu olcen AYRI bir `run` check'i koy. Bu
    check SETUP dunyasinda GECMELI ve referanstan sonra da GECMELI.
17. BAGIMLI MODULLER. Moduller birbirini gercekten cagirsin (A'yi B,
    B'yi C kullansin); yan yana duran bagimsiz dosyalar yazma.
18. KESIF ZORLAMASI. GOAL_TR hangi dosyanin bozuk oldugunu SOYLEMESIN;
    yalnizca gozlenen YANLIS DAVRANISI tarif etsin ("rapor toplamlari
    tutmuyor" gibi). Ajan bozuk modulu testi kosturup bulmali.
    DIKKAT: OUTPUTS yine de duzeltilecek dosyalarin yolunu vermeli --
    kesif ajan icin zor, notlandirma icin degil.

OUTPUTS BU SEKILDE OLMALI -- EN AZ IKI `run` GIRDISI (birebir uygula):

### OUTPUTS
- path: pkg/report.py
  kind: program
  run: python3 -c 'from pkg.report import summary as f; assert f("a,1")=="a=1", f("a,1"); print("TAMAM")'
- path: pkg/report.py
  kind: program
  run: python3 -c 'from pkg.report import fmt as f; assert f(2)=="2.00"; print("REGRESYON-TAMAM")'

Birinci girdi zincirin sonunu olcer (SETUP dunyasinda DUSER).
Ikinci girdi regresyon korumasidir (SETUP dunyasinda da GECER).
TEK `run` girdisi yazarsan aday ELENIR: tek check ile ara adimin ilerleme
saglayip saglamadigi olculemez.
""",
}

# -- ayristirma -------------------------------------------------------

@dataclass
class Spec:
    setup: str = ""
    goal_tr: str = ""
    reference_solution: str = ""
    outputs: list[dict] = field(default_factory=list)
    # Zincirleme gorevlerde YALNIZCA ilk hatayi gideren bash blogu.
    # Ara-durum kapisi bununla "kismi cozum hala dusuyor mu" diye bakiyor;
    # bu blok olmadan cascade iddiasi dogrulanamaz.
    ara_adim: str = ""
    error: str = ""


def parse_spec(text: str) -> Spec:
    """Ayracli yaniti bolumlere ayir; model fence ya da giris cumlesi eklemis olabilir."""
    text = FENCE_RE.sub("", text)
    parcalar = SECTION_RE.split(text)
    if len(parcalar) < 3:
        return Spec(error="yanitta ### bolum ayraci bulunamadi")

    bolumler: dict[str, str] = {}
    for i in range(1, len(parcalar) - 1, 2):
        bolumler[parcalar[i].strip()] = parcalar[i + 1].strip()

    eksik = [b for b in ("SETUP", "GOAL_TR", "REFERENCE", "OUTPUTS") if b not in bolumler]
    if eksik:
        return Spec(error=f"eksik bolumler: {', '.join(eksik)}")

    try:
        outputs = yaml.safe_load(bolumler["OUTPUTS"]) or []
    except yaml.YAMLError as e:
        return Spec(error=f"OUTPUTS YAML olarak cozulemedi: {e}")
    if not isinstance(outputs, list) or not all(isinstance(o, dict) for o in outputs):
        return Spec(error="OUTPUTS bir sozluk listesi olmali")

    return Spec(
        setup=bolumler["SETUP"],
        goal_tr=" ".join(bolumler["GOAL_TR"].split()),
        reference_solution=bolumler["REFERENCE"],
        outputs=outputs,
        ara_adim=bolumler.get("ARA_ADIM", ""),
    )


# -- task.yaml yazimi -------------------------------------------------

def _blok(anahtar: str, govde: str) -> str:
    icerik = "\n".join(f"  {ln}" if ln.strip() else "" for ln in govde.rstrip().split("\n"))
    return f"{anahtar}: |\n{icerik}\n\n"


def write_task_yaml(task_dir: Path, seed: Seed, spec: Spec) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    task_id = task_dir.name
    max_turns = 6 + 2 * seed.adim

    metin = (
        f"id: {task_id}\n"
        f"split: {'single_step' if seed.adim <= 1 else 'multi_step'}\n"
        f"image: {seed.image}\n"
        f"workdir: /workspace\n"
        f"max_turns: {max_turns}\n\n"
        "# Bu task uretildi ve heniz tamamlanmadi: prompt creative writing\n"
        "# asamasinda, alt_solutions ise rakip cozum asamasinda eklenecek.\n"
        "prompt: \"\"\n\n"
    )
    metin += _blok("setup", spec.setup)
    metin += "goal_tr: >\n" + "\n".join(f"  {ln}" for ln in
                                        _sar(spec.goal_tr, 74)) + "\n\n"
    metin += _blok("reference_solution", spec.reference_solution)
    if spec.ara_adim.strip():
        metin += _blok("ara_adim", spec.ara_adim)
    metin += "outputs:\n"
    for o in spec.outputs:
        metin += "  - " + yaml.safe_dump(o, default_flow_style=True, sort_keys=False,
                                         allow_unicode=True, width=10 ** 6).strip() + "\n"
    metin += "\n"
    metin += yaml.safe_dump(
        {"metadata": {
            "domain": seed.hedef,
            "language": "tr",
            "generated": True,
            "pipeline": PIPELINE_VERSION,
            "seed": seed.to_dict(),
        }},
        sort_keys=False, allow_unicode=True, default_flow_style=False)

    path = task_dir / "task.yaml"
    path.write_text(metin, encoding="utf-8")
    return path


def _sar(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# -- uretim -----------------------------------------------------------

@dataclass
class Aday:
    seed: Seed
    kabul: bool = False
    deneme: int = 0
    task_dir: Path | None = None
    hata: str = ""
    dusen_kapilar: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def generate_one(client, model: str, seed: Seed, out_dir: Path,
                 max_deneme: int = 3, verbose: bool = True) -> Aday:
    aday = Aday(seed=seed)

    def kaydet(satir: str) -> None:
        # Es zamanli kosuda dogrudan print etmek 8 parcacigin
        # satirlarini birbirine karistiriyor; aday bitince topluca basilir.
        aday.log.append(satir)
    kod = seed.aile == "kod"
    mesajlar = [
        {"role": "system",
         "content": ((SPEC_TALIMAT_KOD
                      + KARMASIKLIK_TALIMAT.get(
                          seed.metadata.get("karmasiklik", "duz"), ""))
                     if kod else SPEC_TALIMAT)},
        {"role": "user", "content": (seed_brief_kod if kod else seed_brief)(seed)},
    ]
    task_dir = out_dir / f"gen-{seed.id}-{seed.hedef}"

    for deneme in range(1, max_deneme + 1):
        aday.deneme = deneme
        try:
            yanit = client.chat.completions.create(
                model=model, messages=mesajlar, temperature=0.7, max_tokens=16384)
        except Exception as e:  # noqa: BLE001 - API hatasi adayi duserir, kosuyu degil
            aday.hata = f"model cagrisi basarisiz: {type(e).__name__}: {e}"
            return aday

        ham = yanit.choices[0].message.content or ""
        spec = parse_spec(ham)
        if spec.error:
            geri = f"Yanitin ayristirilamadi: {spec.error}. Formata birebir uy."
            if verbose:
                kaydet(f"    deneme {deneme}: {spec.error}")
            mesajlar += [{"role": "assistant", "content": ham},
                         {"role": "user", "content": geri}]
            continue

        write_task_yaml(task_dir, seed, spec)
        aday.task_dir = task_dir

        try:
            task = Task.load(task_dir)
        except Exception as e:  # noqa: BLE001
            aday.hata = f"uretilen task.yaml yuklenemedi: {e}"
            return aday

        try:
            sonuclar, rep = run_gauntlet(task, spec_only=True)
        except Exception as e:  # noqa: BLE001 - bir aday kosuyu bitirmemeli
            hata = f"{type(e).__name__}: {e}"
            aday.dusen_kapilar = ["gauntlet coktu"]
            if verbose:
                kaydet(f"    deneme {deneme}: gauntlet coktu -- {hata[:160]}")
            if deneme == max_deneme:
                aday.hata = f"gauntlet coktu: {hata}"
                return aday
            mesajlar += [{"role": "assistant", "content": ham},
                         {"role": "user", "content":
                          f"Spec degerlendirilirken hata olustu: {hata}. "
                          f"Daha basit ve saglam bir spec yaz."}]
            continue
        dusen = [r for r in sonuclar if not r.ok]
        if not dusen:
            # Check'ler dosyaya yazilmazsa runner ve sweep onlari goremez;
            # gauntlet kendi turettigi icin sorun fark edilmeden gecerdi.
            if rep is not None and rep.checks:
                checks_yaz(task_dir, rep.checks)
            aday.kabul = True
            aday.dusen_kapilar = []
            return aday

        aday.dusen_kapilar = [r.name for r in dusen]
        if verbose:
            for r in dusen:
                kaydet(f"    deneme {deneme}: [{r.name}] {r.detail[:120]}")
        if deneme == max_deneme:
            return aday

        geri = ("Spec su kapilardan gecemedi, duzelt ve TAM spec'i bastan yaz:\n"
                + "\n".join(f"- {r.name}: {r.detail}" for r in dusen))
        mesajlar += [{"role": "assistant", "content": ham},
                     {"role": "user", "content": geri}]

    return aday


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="uretilecek aday sayisi")
    parser.add_argument("--model", default="",
                        help="bos birakilirsa .env'deki OPENAI_MODEL_NAME")
    parser.add_argument("--base-url", default="",
                        help="bos birakilirsa .env'deki OPENAI_BASE_URL")
    parser.add_argument("--api-key", default="",
                        help="bos birakilirsa .env okunur; kendi sunucun icin gereksiz")
    parser.add_argument("--out", default="envs_gen")
    parser.add_argument("--rng", type=int, default=0)
    parser.add_argument("--id-basla", type=int, default=0,
                        help="seed id numaralari buradan bassin; mevcut "
                             "task'larin uzerine yazmamak icin")
    parser.add_argument("--denemeler", type=int, default=3,
                        help="bir aday icin en fazla kac onarim turu")
    parser.add_argument("--gecikme", type=float, default=1.0,
                        help="adaylar arasi bekleme")
    parser.add_argument("--istek-araligi", type=float, default=0.0,
                        help="iki API istegi arasinda en az bu kadar saniye bekle")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="es zamanli aday sayisi")
    args = parser.parse_args()

    args.model = resolve_model(args.model)
    if not args.model:
        print("--model verilmedi ve .env'de OPENAI_MODEL_NAME yok")
        return 2

    ok_uc, bilgi_uc = preflight(args.base_url, args.api_key, args.model)
    if not ok_uc:
        print(f"UC KONTROLU BASARISIZ: {bilgi_uc}")
        return 2
    print(f"uc dogrulandi: {bilgi_uc}")

    ok, info = docker_available()
    if not ok:
        print(f"Docker daemon'a ulasilamiyor: {info}")
        print("Docker Desktop'i baslatip tekrar dene.")
        return 1

    client = make_client(args.base_url, args.api_key, args.istek_araligi)

    out_dir = Path(args.out)
    rng = random.Random(args.rng)
    print(f"model={args.model}  n={args.n}  cikti={out_dir}/  docker={info}\n")

    def basligi(seed: Seed, sira: int) -> str:
        return (f"[{sira}/{args.n}] {seed.id}  {'+'.join(seed.tools)}  {seed.konu}  "
                f"{seed.hedef}  kaynak={seed.kaynak} cikti={seed.cikti} "
                f"kesif={seed.kesif} adim={seed.adim}")

    def sonucu(aday: Aday) -> str:
        if aday.kabul:
            return f"    KABUL ({aday.deneme}. denemede)  -> {aday.task_dir}"
        if aday.hata:
            return f"    HATA  {aday.hata}"
        return f"    RED   dusen kapilar: {', '.join(aday.dusen_kapilar)}"

    seedler = [sample(rng, args.id_basla + i) for i in range(args.n)]
    adaylar: list[Aday] = []

    if args.concurrency > 1:
        # Her aday kendi konteynerlerini acar; es zamanlilik cekirdek
        # sayisini asmasin, yoksa gauntlet'ler birbirini yavaslatir.
        print(f"es zamanlilik: {args.concurrency}\n")
        with ThreadPoolExecutor(max_workers=args.concurrency) as havuz:
            isler = {havuz.submit(generate_one, client, args.model, seed,
                                  out_dir, args.denemeler): (i + 1, seed)
                     for i, seed in enumerate(seedler)}
            for bitmis in as_completed(isler):
                sira, seed = isler[bitmis]
                aday = bitmis.result()
                adaylar.append(aday)
                # Adayin butun ciktisi tek blok halinde: satirlar karismasin.
                print("\n".join([basligi(seed, sira), *aday.log, sonucu(aday)]))
    else:
        for i, seed in enumerate(seedler):
            print(basligi(seed, i + 1))
            aday = generate_one(client, args.model, seed, out_dir, args.denemeler)
            adaylar.append(aday)
            print("\n".join([*aday.log, sonucu(aday)]))
            if args.gecikme and i < args.n - 1:
                time.sleep(args.gecikme)

    kabul = sum(1 for a in adaylar if a.kabul)
    print(f"\nKABUL ORANI: {kabul}/{len(adaylar)}")
    if kabul:
        ortalama = sum(a.deneme for a in adaylar if a.kabul) / kabul
        print(f"Kabul edilenlerde ortalama deneme sayisi: {ortalama:.1f}")
    sayac: dict[str, int] = {}
    for a in adaylar:
        for kapi in a.dusen_kapilar:
            sayac[kapi] = sayac.get(kapi, 0) + 1
    if sayac:
        print("En cok duseren kapilar: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(sayac.items(), key=lambda kv: -kv[1])))
    print(f"\nKabul edilenler heniz eksik: prompt ve alt_solutions sonraki asamalarda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
