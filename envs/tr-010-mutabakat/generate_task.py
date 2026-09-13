"""tr-010-mutabakat'i uretir: veriyi kurar, referansi calistirir, task.yaml yazar.

    python3 envs/tr-010-mutabakat/generate_task.py

Bu task'in beklenen degerleri elle hesaplanamaz -- 12 dosyalik bir birlestirme,
sqlite join'i ve tarih bazli kur cevrimi var. Bu yuzden check degerleri elle
YAZILMAZ, CALISTIRILARAK turetilir: veri uretilir, referans cozum uzerinde
kosturulur, ortaya cikan rapor.json check'lere donusur.

Uretim uc sozlesmeyi dogrulamadan task.yaml yazmaz:

  1. Referans cozum check'lerin HEPSINI gecmeli.
  2. Baslangic durumu (S0, cozum calismadan) check'lerin HICBIRINI gecmemeli --
     yoksa check bos bir odul verir.
  3. Naif cozum (Turkce harf katlamasi yerine duz .lower()) tuzaga dusmeli --
     yoksa task'in ogretmeyi hedefledigi sey olculmuyor demektir.

Setup ve referans, task.yaml'a gomulen tam da su kaynak metinlerdir; boylece
konteynerde uretilen veri host'ta uretilenle bit bazinda ayni olur.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from verifiable_dataset.terminal.checks import run_checks  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
IMAGE = "verifiable-dataset/base:latest"


# -- 1. dunyayi kuran kaynak ------------------------------------------
#
# Konteynerde `python3 - <<'PY'` ile calisir. Rastgelelik icin sabit bir LCG
# kullaniyoruz: random modulunun ureteci surumler arasi stabil olsa da, veri
# uretimini yorumlayici surumune hic bagli birakmamak daha ucuz bir sigorta.

GEN = r'''
import csv, os, sqlite3, datetime as dt

class R:
    """Surumden bagimsiz deterministik ureteci (64-bit LCG)."""
    def __init__(self, seed): self.s = seed & 0xFFFFFFFFFFFFFFFF
    def _n(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 33
    def below(self, n): return self._n() % n
    def pick(self, xs): return xs[self._n() % len(xs)]
    def unit(self): return self._n() / 2147483648.0

def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()

KAYITLI = [
    "İsmail Çetin", "Işıl Yılmaz", "İlknur Doğan", "Irmak Kara", "Şahin Öztürk",
    "Halil Kılıç", "Nazlı Aydın", "Cihan Şimşek", "Bilge Arslan", "Ayşegül Kaya",
    "Zeynep Yıldız", "Mustafa Şen", "Elif Karaca", "Kerem Bilgin", "Selin Uçar",
    "Barış Gündüz", "Pınar Tekin", "Emre Sarı", "Gizem Aksoy", "Onur Balcı",
    "Sıla Demirtaş", "Yiğit Erdoğan", "Melis Turan", "Tolga Çınar", "Aslı Yavuz",
    "Burak Şahin", "Ceren Ilgaz", "Deniz Akın", "Ebru Kılıçarslan", "Ferhat Özdemir",
    "Gökhan Işık", "Hande Yalçın", "İbrahim Polat", "Jale Sönmez",
]
KAYITSIZ = ["Uğur Tanrıkulu", "Şeyma Bozkurt", "Levent Ilıcalı", "Nihal Öğüt"]
SEHIRLER = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kayseri", "Şanlıurfa"]

os.makedirs("veri", exist_ok=True)

# -- kurlar: yalnizca is gunleri, resmi tatiller yok ---------------------
TATIL = {
    "2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23",
    "2024-05-01", "2024-05-19", "2024-06-16", "2024-06-17", "2024-06-18",
    "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
}
r = R(20240115)
kur_satirlari = []
gun = dt.date(2023, 12, 29)          # siparisler 2024'te; her tarihin oncesi var
son = dt.date(2024, 12, 31)
oran = 29.40
while gun <= son:
    iso = gun.isoformat()
    oran += 0.035 + r.unit() * 0.06
    if gun.weekday() < 5 and iso not in TATIL:
        kur_satirlari.append((iso, f"{oran:.4f}"))
    gun += dt.timedelta(days=1)

with open("veri/kurlar.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tarih", "usd_try"])
    w.writerows(kur_satirlari)

# -- musteri veritabani: adlar TURKCE BUYUK harfle -----------------------
db = "veri/musteriler.db"
if os.path.exists(db):
    os.remove(db)
con = sqlite3.connect(db)
con.execute("CREATE TABLE musteriler (musteri_no INTEGER PRIMARY KEY, ad TEXT NOT NULL, sehir TEXT)")
con.executemany(
    "INSERT INTO musteriler VALUES (?, ?, ?)",
    [(1000 + i, tr_upper(ad), r.pick(SEHIRLER)) for i, ad in enumerate(KAYITLI)],
)
con.commit()
con.close()

# -- siparis dosyalari ---------------------------------------------------
VARSAYILAN = ["siparis_no", "tarih", "musteri", "para_birimi", "tutar"]
DUZEN = {
    3: ["tutar", "siparis_no", "tarih", "musteri", "para_birimi"],
    5: ["musteri", "tutar", "para_birimi", "siparis_no", "tarih"],
}
BOZUK = ["", "   ", "yok", "-", "belirsiz"]

herkes = KAYITLI + KAYITSIZ
sira = 0
for ay in range(1, 13):
    gun_sayisi = (dt.date(2024, ay, 28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    satirlar = []
    for _ in range(25):
        sira += 1
        tarih = dt.date(2024, ay, 1 + r.below(gun_sayisi.day)).isoformat()
        musteri = r.pick(herkes)
        usd = r.below(100) < 28
        # USD siparisler kucuk kalemler; ayni araligi kullanmak TL toplamini
        # kurun yaninda gorunmez kilardi.
        deger = (100 + r.below(790000) / 100.0) if usd else (500 + r.below(24950000) / 100.0)
        if r.below(100) < 5:                       # gecersiz: bos / sayi disi
            tutar = r.pick(BOZUK)
        elif r.below(100) < 2:                     # gecersiz: negatif
            tutar = f"-{deger:.2f}"
        elif ay in (2, 9):                         # 1.234,56 bicimi
            tutar = f"{deger:,.2f}".translate(str.maketrans(",.", ".,"))
        else:                                      # 1234.56 bicimi
            tutar = f"{deger:.2f}"
        satirlar.append({
            "siparis_no": f"S{sira:05d}",
            "tarih": tarih,
            "musteri": musteri,
            "para_birimi": "USD" if usd else "TL",
            "tutar": tutar,
        })

    alanlar = DUZEN.get(ay, VARSAYILAN)
    enc = "utf-8-sig" if ay == 3 else "utf-8"      # 3. ayda BOM var
    delim = ";" if ay == 7 else ","                # 7. ay noktali virgullu
    with open(f"veri/satis_2024_{ay:02d}.csv", "w", encoding=enc, newline="") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, delimiter=delim)
        w.writeheader()
        for s in satirlar:
            w.writerow({k: s[k] for k in alanlar})

# -- spec: prompt'ta degil, burada --------------------------------------
with open("KURALLAR.md", "w", encoding="utf-8") as f:
    f.write("""# Mutabakat Kuralları

## 1. Kapsam
`veri/satis_2024_01.csv` ... `veri/satis_2024_12.csv` dosyalarındaki bütün
sipariş satırları. Her dosyanın ilk satırı başlıktır ve sayılmaz.

Dosyalar aynı sistemden çıkmadı: sütun sırası her dosyada aynı değildir,
bir dosya noktalı virgül ayraçlıdır, bir dosya BOM ile başlar. Sütunlara
adlarıyla eriş, konumlarıyla değil.

## 2. Tutar biçimi
İki biçim bir arada bulunur ve ikisi de aynı değeri ifade eder:

    1234.56     nokta ondalık ayraç
    1.234,56    nokta binlik ayraç, virgül ondalık ayraç

## 3. Geçersiz satırlar
`tutar` alanı boş olan, sayıya çevrilemeyen ya da **negatif** olan satırlar
geçersizdir. Geçersiz satırlar hiçbir toplama girmez, ama sayılır.

## 4. Para birimi ve kur
`para_birimi` TL ise tutar olduğu gibi alınır.

`para_birimi` USD ise sipariş **kendi tarihindeki** kurla TL'ye çevrilir.
`veri/kurlar.csv` günlük USD/TRY kurunu tutar, ama yalnızca iş günleri için:
hafta sonları ve resmî tatiller dosyada yoktur. Siparişin tarihi dosyada
yoksa, o tarihten **önceki en yakın günün** kuru kullanılır.

## 5. Yuvarlama
Ara hesaplarda yuvarlama yapma. Yalnızca rapora yazdığın nihai tutarları
2 ondalık basamağa yuvarla.

## 6. Müşteri eşleştirme
`veri/musteriler.db` bir SQLite veritabanıdır; `musteriler` tablosundaki `ad`
sütunu kayıtlı müşterileri tutar. CSV'lerdeki müşteri adı ile veritabanındaki
ad aynı kişiyi gösterir, ama yazım biçimleri farklıdır: veritabanı adları
büyük harfle yazılmıştır.

Eşleştirme **Türkçe harf kurallarına** göre yapılır. Türkçede:

    I harfinin küçüğü  ı  (i değil)
    İ harfinin küçüğü  i  (i̇ değil)

Yani "İSMAİL ÇETİN" ile "İsmail Çetin" aynı müşteridir; "IŞIL YILMAZ" ile
"Işıl Yılmaz" aynı müşteridir. Karşılaştırmadan önce baştaki ve sondaki
boşlukları at.

## 7. rapor.json
Sonucu `/workspace/rapor.json` dosyasına yaz. Tam olarak şu alanlar bulunmalı:

    toplam_satir           bütün dosyalardaki sipariş satırı sayısı
                           (geçersizler dahil, başlıklar hariç)
    gecersiz_satir         3. maddeye göre geçersiz satır sayısı
    toplam_tl              para_birimi TL olan geçerli siparişlerin toplamı
    toplam_usd_tl          para_birimi USD olan geçerli siparişlerin
                           TL karşılıklarının toplamı
    en_cok_harcayan        toplam TL harcaması en yüksek olan müşterinin
                           CSV'de yazıldığı hâliyle adı (kayıtlı olması
                           gerekmez; USD siparişler çevrilmiş hâliyle sayılır)
    eslesmeyen_musteriler  CSV'lerde geçen ama veritabanında kaydı olmayan
                           müşterilerin CSV'deki hâliyle adları, alfabetik
                           sıralı liste (geçersiz satırlardaki adlar da sayılır)
    kayitli_ciro           veritabanında kaydı OLAN müşterilerin geçerli
                           siparişlerinin TL cinsinden toplamı
""")
'''


# -- 2. referans cozum -------------------------------------------------

REF = r"""
import bisect, csv, json, sqlite3

def tr_lower(s):
    return s.replace("İ", "i").replace("I", "ı").lower().strip()

def tutar_oku(ham):
    s = (ham or "").strip()
    if not s:
        return None
    if "," in s:                       # 1.234,56 -> 1234.56
        s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v < 0 else v        # negatif tutar gecersiz (KURALLAR md.3)

satirlar = []
for ay in range(1, 13):
    yol = f"veri/satis_2024_{ay:02d}.csv"
    # utf-8-sig BOM'suz dosyalarda da zararsiz; ayraci ornekten anla.
    with open(yol, encoding="utf-8-sig", newline="") as f:
        ornek = f.read(4096)
        f.seek(0)
        ayrac = ";" if ornek.count(";") > ornek.count(",") else ","
        satirlar.extend(csv.DictReader(f, delimiter=ayrac))

kur = {}
with open("veri/kurlar.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        kur[r["tarih"]] = float(r["usd_try"])
gunler = sorted(kur)

def kur_of(tarih):
    # ISO tarihlerde metin siralamasi kronolojik; o gun yoksa oncesine dus.
    i = bisect.bisect_right(gunler, tarih) - 1
    return kur[gunler[i]]

con = sqlite3.connect("veri/musteriler.db")
kayitli = {tr_lower(ad) for (ad,) in con.execute("SELECT ad FROM musteriler")}
con.close()

gecersiz = 0
harcama = {}
toplam_tl = 0.0
toplam_usd = 0.0
kayitli_ciro = 0.0
gorulen = set()

for r in satirlar:
    ad = (r["musteri"] or "").strip()
    gorulen.add(ad)
    v = tutar_oku(r.get("tutar"))
    if v is None:
        gecersiz += 1
        continue
    if (r["para_birimi"] or "").strip().upper() == "USD":
        tl = v * kur_of((r["tarih"] or "").strip())
        toplam_usd += tl
    else:
        tl = v
        toplam_tl += tl
    harcama[ad] = harcama.get(ad, 0.0) + tl
    if tr_lower(ad) in kayitli:
        kayitli_ciro += tl

rapor = {
    "toplam_satir": len(satirlar),
    "gecersiz_satir": gecersiz,
    "toplam_tl": round(toplam_tl, 2),
    "toplam_usd_tl": round(toplam_usd, 2),
    "en_cok_harcayan": max(harcama, key=lambda a: harcama[a]),
    "eslesmeyen_musteriler": sorted(a for a in gorulen if tr_lower(a) not in kayitli),
    "kayitli_ciro": round(kayitli_ciro, 2),
}
with open("rapor.json", "w", encoding="utf-8") as f:
    json.dump(rapor, f, ensure_ascii=False, indent=2)
"""

# Naif varyant: Turkce katlama yerine duz .lower(). Uretimde tuzagin gercekten
# ayirt edip etmedigini olcmek icin kullaniliyor, task'a girmiyor.
TR_LOWER_LINE = 's.replace("İ", "i").replace("I", "ı").lower().strip()'
NAIVE = REF.replace(TR_LOWER_LINE, "s.lower().strip()")
assert NAIVE != REF, "naif varyant uretilemedi"


PROMPT = """Bir Linux terminalindesin, çalışma dizinin /workspace.

Bu dizinde bir yılın satış arşivi duruyor: aylık sipariş dosyaları, günlük
döviz kurları ve bir müşteri veritabanı. Dosyalar farklı sistemlerden geldiği
için biçimleri birbirini tutmuyor.

Mutabakat kuralları `KURALLAR.md` dosyasında yazılı. Önce onu oku; hangi
satırın geçersiz sayıldığı, tutarların nasıl okunacağı, kurun hangi güne göre
alınacağı ve müşterilerin nasıl eşleştirileceği orada tanımlı.

Kurallara göre mutabakatı çıkar ve sonucu `/workspace/rapor.json` dosyasına
yaz. Hangi alanların bulunması gerektiği de `KURALLAR.md` içinde.

İşin bittiğinde kısaca ne yaptığını ve hangi biçim sorunlarıyla karşılaştığını
özetle.
"""

METADATA = """  domain: data_reconciliation
  language: tr
  n_ref_steps: 1
  generator: generate_task.py
  notes: >
    Spec prompt'ta degil KURALLAR.md'de; model once kesfetmek zorunda.
    Tuzaklar: 3. ay BOM + sutun sirasi degisik, 5. ay sutun sirasi degisik,
    7. ay noktali virgul ayrac, 2. ve 9. ay 1.234,56 bicimi, negatif tutar
    yalnizca KURALLAR'da gecersiz sayiliyor, kur dosyasinda hafta sonu ve
    tatil yok (onceki gune dusmek gerekiyor), musteri adlari veritabaninda
    TURKCE buyuk harfle. Son madde asil kapi: duz .lower() ile katlama
    "İSMAİL" -> "i̇smai̇l" uretip eslesmeyi bozar, Turkce katlama sart.
    Check degerleri elle yazilmadi, generate_task.py referansi calistirip
    turetti.
"""


def _sh(src: str) -> str:
    """Kaynagi konteynerde calistiracak tek satirlik kabuk komutuna sar."""
    return f"python3 - <<'PYEOF'\n{src.strip()}\nPYEOF"


def _run(src: str, cwd: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", src],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(f"kaynak calismadi:\n{proc.stderr}")


def _checks_for(rapor: dict) -> list[dict]:
    """Referansin urettigi rapordan check merdivenini kur.

    Sira kolaydan zora: reward = passed/total oldugu icin yarida kalan bir
    cozumun kismi puan almasi, GRPO'da grup ici varyansin kaynagi.
    """
    p = "rapor.json"
    return [
        {"op": "is_file", "path": p},
        {
            "op": "json_field_eq",
            "path": p,
            "field": "toplam_satir",
            "value": rapor["toplam_satir"],
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "gecersiz_satir",
            "value": rapor["gecersiz_satir"],
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "toplam_tl",
            "value": rapor["toplam_tl"],
            "tol": 0.05,
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "toplam_usd_tl",
            "value": rapor["toplam_usd_tl"],
            "tol": 0.05,
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "en_cok_harcayan",
            "value": rapor["en_cok_harcayan"],
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "eslesmeyen_musteriler",
            "value": rapor["eslesmeyen_musteriler"],
            "as_set": True,
        },
        {
            "op": "json_field_eq",
            "path": p,
            "field": "kayitli_ciro",
            "value": rapor["kayitli_ciro"],
            "tol": 0.05,
        },
    ]


def _grade(root: Path, checks: list[dict]) -> tuple[int, int, list[dict]]:
    res = run_checks(root, checks, IMAGE)
    return sum(1 for r in res if r["ok"]), len(res), res


def _block(text: str, indent: str = "  ") -> str:
    return "".join(
        f"{indent}{ln}\n" if ln.strip() else "\n" for ln in text.strip().splitlines()
    )


def _yaml_check(c: dict) -> str:
    parts = [
        f"{k}: {json.dumps(v, ensure_ascii=False)}" if k != "op" else f"op: {v}"
        for k, v in c.items()
    ]
    return "  - {" + ", ".join(parts) + "}"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tr010-") as tmp:
        base = Path(tmp)

        # -- referans dunyasi: veri + cozum -> beklenen degerler ----------
        ok_dir = base / "ok"
        ok_dir.mkdir()
        _run(GEN, ok_dir)
        _run(REF, ok_dir)
        rapor = json.loads((ok_dir / "rapor.json").read_text(encoding="utf-8"))
        checks = _checks_for(rapor)

        passed, total, _ = _grade(ok_dir, checks)
        if passed != total:
            raise SystemExit(f"sozlesme 1 ihlal: referans {passed}/{total}")

        # -- S0: cozum calismadan hicbir check gecmemeli ------------------
        s0 = base / "s0"
        s0.mkdir()
        _run(GEN, s0)
        s0_passed, _, _ = _grade(s0, checks)
        if s0_passed != 0:
            raise SystemExit(f"sozlesme 2 ihlal: S0 bedava {s0_passed} check veriyor")

        # -- naif cozum tuzaga dusmeli -----------------------------------
        nv = base / "naive"
        nv.mkdir()
        _run(GEN, nv)
        _run(NAIVE, nv)
        nv_passed, _, nv_res = _grade(nv, checks)
        if nv_passed >= total:
            raise SystemExit(
                "sozlesme 3 ihlal: naif katlama da geciyor, tuzak ise yaramiyor"
            )

        print(f"referans : {passed}/{total}")
        print(f"S0       : {s0_passed}/{total}")
        print(
            f"naif fold: {nv_passed}/{total}  (dusen: "
            f"{', '.join(r['name'].split(':', 1)[1] for r in nv_res if not r['ok'])})"
        )
        print()
        for k, v in rapor.items():
            print(f"  {k:<22} {json.dumps(v, ensure_ascii=False)[:120]}")

    yaml = (
        "# ELLE DUZENLEME. generate_task.py uretir; check degerleri referans\n"
        "# cozum calistirilarak turetilmistir.\n"
        "id: tr-010-mutabakat\n"
        "split: multi_step\n"
        f"image: {IMAGE}\n"
        "workdir: /workspace\n"
        "max_turns: 20\n"
        "\n"
        "setup: |\n" + _block(_sh(GEN)) + "\n"
        "prompt: |\n" + _block(PROMPT) + "\n"
        "reference_solution: |\n" + _block(_sh(REF)) + "\n"
        "checks:\n" + "\n".join(_yaml_check(c) for c in checks) + "\n"
        "\n"
        "metadata:\n" + METADATA
    )
    out = TASK_DIR / "task.yaml"
    out.write_text(yaml, encoding="utf-8")
    print(f"\n-> {out}  ({len(yaml.splitlines())} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
