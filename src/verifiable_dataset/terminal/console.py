"""Episode izlerini terminale okunakli sekilde basar.

Renkler ve kutu karakterleri; cikti bir TTY degilse, NO_COLOR ayarliysa ya da
konsol UTF-8 konusamiyorsa kendiliginden kapanir. Boylece `... > log.txt`
escape dizisi degil duz metin birakir.

Burasi yalnizca gorunumden sorumlu. Hicbir fonksiyon modelin gordugu metni ya
da reward'i etkilemez: uzun ciktilar SADECE ekranda kisaltilir, sandbox'in
modele verdigi gozlem ve JSONL kaydi tam kalir.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap

MAX_WIDTH = 96
DISPLAY_LINES = 14      # bir komut ciktisindan ekranda gosterilecek satir
DISPLAY_COLS = 400      # tek satirin ekranda kesilecegi uzunluk

INDENT = "  "

_flags: dict[str, bool | None] = {"color": None, "unicode": None}

_UNICODE_GLYPHS = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│", "turn": "▌", "think": "┆",
    "ok": "✔", "fail": "✘", "dot": "·",
    "on": "▰", "off": "▱",
}
_ASCII_GLYPHS = {
    "tl": "+", "tr": "+", "bl": "+", "br": "+",
    "h": "-", "v": "|", "turn": "|", "think": ":",
    "ok": "+", "fail": "x", "dot": "-",
    "on": "#", "off": ".",
}

_CODES = {
    "reset": "0", "bold": "1", "dim": "2",
    "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "gray": "90",
}


def _color_on() -> bool:
    if _flags["color"] is None:
        if os.environ.get("NO_COLOR"):
            _flags["color"] = False
        elif os.environ.get("FORCE_COLOR"):
            _flags["color"] = True
        else:
            _flags["color"] = bool(getattr(sys.stdout, "isatty", lambda: False)())
    return bool(_flags["color"])


def _unicode_on() -> bool:
    # main() akisi utf-8'e reconfigure ettikten SONRA cagrildigi icin bu
    # kontrol tembel; import aninda bakarsak eski kodlamayi gorurduk.
    if _flags["unicode"] is None:
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        _flags["unicode"] = "utf" in enc
    return bool(_flags["unicode"])


def g(name: str) -> str:
    return (_UNICODE_GLYPHS if _unicode_on() else _ASCII_GLYPHS)[name]


def paint(text: str, *styles: str) -> str:
    if not styles or not _color_on():
        return text
    prefix = "".join(f"\033[{_CODES[s]}m" for s in styles)
    return f"{prefix}{text}\033[0m"


def width() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, MAX_WIDTH)


def duration(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _wrap(text: str, cols: int) -> list[str]:
    """Paragraf sinirlarini koruyarak sar."""
    lines: list[str] = []
    for para in text.strip().splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, max(20, cols)) or [""])
    return lines


def _gutter(glyph: str, style: str) -> str:
    return INDENT + paint(glyph, style) + " "


def _body(lines: list[str], glyph: str, gutter_style: str, *text_styles: str) -> None:
    pre = _gutter(glyph, gutter_style)
    for line in lines:
        print(pre + paint(line, *text_styles) if line else pre.rstrip())


# -- ust seviye bloklar -----------------------------------------------

def header(title: str, rows: list[tuple[str, str]]) -> None:
    inner = width() - 2
    label = f" {title} "
    fill = max(0, inner - 1 - len(label))
    print(paint(g("tl") + g("h"), "gray")
          + paint(label, "bold", "cyan")
          + paint(g("h") * fill + g("tr"), "gray"))
    for key, value in rows:
        value = value[:max(0, inner - 11)]
        pad = " " * max(0, inner - 10 - len(value))
        print(paint(g("v"), "gray") + " " + paint(f"{key:<9}", "gray")
              + value + pad + paint(g("v"), "gray"))
    print(paint(g("bl") + g("h") * inner + g("br"), "gray"))


def rollout(index: int, total: int) -> None:
    label = f" rollout {index}/{total} "
    print()
    print(paint(g("h") * 2 + label + g("h") * max(0, width() - 2 - len(label)), "gray"))


def turn(number: int) -> None:
    print()
    print(INDENT + paint(g("turn"), "blue") + " " + paint(f"tur {number}", "bold", "blue"))


def thinking(text: str) -> None:
    """Modelin akil yurutmesi (vLLM reasoning_content ya da metin protokolu)."""
    if not text.strip():
        return
    _body(_wrap(text, width() - 6), g("think"), "gray", "gray")


def says(text: str) -> None:
    """Modelin kullaniciya donuk anlatimi."""
    if not text.strip():
        return
    _body(_wrap(text, width() - 6), g("v"), "gray")


def command(cmd: str) -> None:
    lines = cmd.strip().splitlines()
    pre = _gutter(g("v"), "gray")
    print(pre + paint("$ ", "green") + paint(lines[0], "bold"))
    # Cok satirli heredoc'lari da vurgulamak ekrani bogar; devam satirlari sade.
    for line in lines[1:DISPLAY_LINES]:
        print(pre + "  " + line)
    if len(lines) > DISPLAY_LINES:
        print(pre + "  " + paint(f"... +{len(lines) - DISPLAY_LINES} satir", "gray"))


def exec_result(exit_code: int, stdout: str, stderr: str,
                elapsed: float, timed_out: bool = False) -> None:
    pre = _gutter(g("v"), "gray") + "  "
    status_style = "green" if exit_code == 0 else "red"
    meta = f"exit {exit_code} {g('dot')} {duration(elapsed)}"
    if timed_out:
        meta += " (zaman asimi)"
    print(pre + paint(meta, status_style))

    streams = [(stdout, ()), (stderr, ("red",))]
    shown = 0
    dropped = 0
    for text, styles in streams:
        for line in text.rstrip().splitlines():
            if shown >= DISPLAY_LINES:
                dropped += 1
                continue
            print(pre + paint(line[:DISPLAY_COLS], "dim", *styles))
            shown += 1
    if dropped:
        print(pre + paint(f"... +{dropped} satir (tam cikti modele gitti)", "gray"))
    elif shown == 0:
        print(pre + paint("(cikti yok)", "gray"))


def verdict(solved: bool, reward: float, passed: int, total: int,
            turns: int, elapsed: float, checks: list[dict],
            error: str | None = None, attempted: bool = True) -> None:
    style = "green" if solved else "red"
    word = "SOLVED" if solved else "FAILED"
    mark = g("ok") if solved else g("fail")
    print()
    print(INDENT + paint(f"{mark} {word}", "bold", style)
          + paint(f"   reward {reward:.2f}   {passed}/{total} check"
                  f"   {turns} tur   {duration(elapsed)}", "gray"))

    # Gecen kosuda tik isareti dizmek gurultu; dokumu sadece basarisizken yap.
    # Gecen check'ler yine de listelenir: reward=passed/total oldugu icin
    # modelin nereye kadar geldigini gormek gerekiyor.
    # Hic komut calismadiysa (orn. model sunucusuna ulasilamadi) check listesi
    # bilgi tasimaz -- hepsi zaten baslangic durumunda kaldigi icin fail.
    if not solved and attempted:
        for check in checks:
            ok = check["ok"]
            mark = paint(g("ok"), "green") if ok else paint(g("fail"), "bold", "red")
            print(INDENT * 2 + mark + " " + paint(check["name"], "dim" if ok else "red"))
            if not ok and check.get("detail"):
                for line in _wrap(check["detail"], width() - 8):
                    print(INDENT * 3 + " " + paint(line, "gray"))
    if error:
        print(INDENT * 2 + paint(f"! {error}", "yellow"))


def summary(solved_flags: list[bool], rewards: list[float],
            band: str | None = None) -> None:
    n = len(solved_flags)
    bar = "".join(g("on") if s else g("off") for s in solved_flags)
    n_solved = sum(solved_flags)
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    style = "green" if n_solved == n else ("red" if n_solved == 0 else "yellow")
    print()
    print(INDENT + paint(f"pass@{n}", "bold") + f"  {n_solved}/{n}  "
          + paint(bar, style)
          + paint(f"   ortalama reward {mean:.3f}", "gray"))
    if band:
        band_style = "green" if band == "BANT" else "yellow"
        print(INDENT + paint(f"zorluk bandi: {band}", band_style))


# -- tek satirlik mesajlar --------------------------------------------

def note(text: str) -> None:
    print(paint(text, "gray"))


def warn(text: str) -> None:
    print(paint(f"! {text}", "yellow"))


def error(text: str) -> None:
    print(paint(f"{g('fail')} {text}", "bold", "red"))
