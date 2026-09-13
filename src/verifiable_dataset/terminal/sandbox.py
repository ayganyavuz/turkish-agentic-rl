"""Terminal sandbox: Docker, Singularity ya da (izolasyonsuz) yerel.

One container per episode, kept alive across turns so filesystem state
persists. Commands run via ``docker exec``, which means the working
directory and exported environment variables do NOT carry over between
turns -- each exec is a fresh shell. That is fine for filesystem tasks;
tasks that need a persistent venv or background processes will need a
PTY-attached session instead (see notes in README).

Uc mod da ayni arayuzu veriyor (``start/exec/yukle/snapshot/stop``), boylece
cagri noktalari hangi motorun kostugunu bilmiyor; secim ``mod_sec`` ile tek
yerden yapiliyor.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

DOCKER = shutil.which("docker") or "docker"
SINGULARITY = shutil.which("singularity") or "singularity"
# Cozumlenmis yol sart: Windows'ta ciplak "bash" WSL'in bash.exe'sine
# gidiyor ve Windows bicimli calisma dizinini goremiyor. Colab'de zaten
# /bin/bash.
BASH = shutil.which("bash") or "bash"

# Ag izolasyonu (`--net --network none`) ayricaliksiz network namespace
# istiyor; cogu HPC kurulumunda kapali ve kapaliyken instance hic
# baslamiyor. Bu yuzden Docker'in aksine varsayilan KAPALI -- kosuyu
# tumden bloke etmektense izolasyonu bilerek kaybediyoruz.
# VDS_AG_IZOLE=1 ile acilir; MN5'te bir kez deneyip kalici karar verin.
AG_IZOLE = os.environ.get("VDS_AG_IZOLE", "") == "1"


class SandboxError(RuntimeError):
    """Raised when the container itself misbehaves (not when a command fails)."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    def to_observation(self, max_chars: int = 4000) -> str:
        """Render the result the way the agent sees it."""
        parts = [f"exit_code: {self.exit_code}"]
        if self.timed_out:
            parts.append("(komut zaman asimina ugradi)")
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.rstrip()}")
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(cikti yok)")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (kesildi, toplam {len(text)} karakter)"
        return text


def _run(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _erisimi_ac(kok: Path, degisen: list[tuple[Path, int]]) -> None:
    """Okunamayan dosya/dizinlere gecici okuma izni ver, eskisini not et.

    Dizinler icine inmeden once acilir; yoksa okunamayan bir dizinin alti
    hic gezilemez.
    """
    try:
        mod = stat.S_IMODE(kok.lstat().st_mode)
    except OSError:
        return
    if kok.is_symlink():
        return
    gerek = stat.S_IRUSR | (stat.S_IXUSR if kok.is_dir() else 0)
    if mod & gerek != gerek:
        degisen.append((kok, mod))
        try:
            os.chmod(kok, mod | gerek)
        except OSError:
            degisen.pop()
            return
    if kok.is_dir():
        for alt in sorted(kok.iterdir()):
            _erisimi_ac(alt, degisen)


def _snapshot_kopyala(kaynak: Path, hedef: Path) -> None:
    """Modlari koruyarak kopyala -- okunamayan dosyalara ragmen.

    Docker snapshot'i daemon root olarak aliyor, yani 0000 modlu bir dosya
    sorunsuz kopyalaniyor. Singularity ve yerel modda kopyalama bizim
    kullanicimizla kosuyor: gorevin bilerek okunamaz yaptigi tek bir dosya
    (`okunmayan_dosya` gibi) butun snapshot'i EACCES ile dusuruyor ve gorev
    bozukmus gibi FAIL veriyordu.

    Modlar hem kaynakta hem hedefte geri konuyor -- yoksa izinleri sinayan
    bir check, biz actigimiz icin yanlislikla gecerdi. Geri koyma derinden
    yuzeye: once cocuk, sonra dizin, aksi halde kapanan dizinin altina bir
    daha girilemez.
    """
    degisen: list[tuple[Path, int]] = []
    _erisimi_ac(Path(kaynak), degisen)
    try:
        shutil.copytree(kaynak, hedef, symlinks=True)
    finally:
        for yol, eski in reversed(degisen):
            try:
                os.chmod(yol, eski)
            except OSError:
                pass
    for yol, eski in reversed(degisen):
        try:
            os.chmod(Path(hedef) / yol.relative_to(kaynak), eski)
        except OSError:
            pass


def _icerigi_kopyala(kaynak: Path, hedef: Path) -> None:
    """`docker cp src/. cid:dst` gibi: dizinin kendisini degil icerigini tasi."""
    hedef.mkdir(parents=True, exist_ok=True)
    for oge in Path(kaynak).iterdir():
        varis = hedef / oge.name
        if oge.is_dir() and not oge.is_symlink():
            shutil.copytree(oge, varis, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(oge, varis, follow_symlinks=False)


class DockerSandbox:
    """Start a container, exec commands in it, snapshot it, tear it down."""

    yerel = False

    def __init__(
        self,
        image: str,
        workdir: str = "/workspace",
        cpus: str = "1",
        memory: str = "512m",
        pids_limit: int = 256,
        network: str = "none",
    ):
        self.image = image
        self.workdir = workdir
        self.cpus = cpus
        self.memory = memory
        self.pids_limit = pids_limit
        self.network = network
        self.container_id: str | None = None

    # -- lifecycle ----------------------------------------------------

    def start(self) -> str:
        name = f"vds-{uuid.uuid4().hex[:12]}"
        args = [
            DOCKER, "run", "-d", "--rm",
            "--name", name,
            "--network", self.network,
            "--cpus", self.cpus,
            "--memory", self.memory,
            "--pids-limit", str(self.pids_limit),
            "--workdir", self.workdir,
            self.image,
            "sleep", "infinity",
        ]
        proc = _run(args, timeout=120)
        if proc.returncode != 0:
            raise SandboxError(f"container baslatilamadi: {proc.stderr.strip()}")
        self.container_id = proc.stdout.strip()
        return self.container_id

    def stop(self) -> None:
        if not self.container_id:
            return
        _run([DOCKER, "kill", self.container_id], timeout=60)
        self.container_id = None

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- operations ---------------------------------------------------

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        """Run a shell command inside the container."""
        if not self.container_id:
            raise SandboxError("container calismiyor")
        args = [
            DOCKER, "exec", "--workdir", self.workdir,
            self.container_id, "bash", "-lc", command,
        ]
        try:
            proc = _run(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="", exit_code=124, timed_out=True)
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)

    def yukle(self, kaynak: Path) -> None:
        """Host'taki bir dizinin icerigini calisma dizinine kopyala."""
        if not self.container_id:
            raise SandboxError("container calismiyor")
        proc = _run(
            [DOCKER, "cp", f"{kaynak}/.", f"{self.container_id}:{self.workdir}"],
            timeout=120,
        )
        if proc.returncode != 0:
            raise SandboxError(f"kopyalanamadi: {proc.stderr.strip()}")

    def snapshot(self, dest: Path) -> Path:
        """Copy the workdir out to the host for grading.

        Grading never runs inside the container, so a tampered interpreter
        or a planted checker cannot influence the reward.
        """
        if not self.container_id:
            raise SandboxError("container calismiyor")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        proc = _run(
            [DOCKER, "cp", f"{self.container_id}:{self.workdir}", str(dest)],
            timeout=120,
        )
        if proc.returncode != 0:
            raise SandboxError(f"snapshot alinamadi: {proc.stderr.strip()}")
        return dest


# -- singularity sandbox ----------------------------------------------
# MareNostrum 5'te Docker yok: root ve kalici bir daemon istiyor, cok
# kullanicili HPC'de verilmiyor. BSC `singularity` modulu sunuyor
# (ACC'de 3.11.5 ve 4.1.5).

def sif_yolu(image: str) -> Path:
    """Imaj adini `.sif` dosyasina cevir: `vds/base:latest` -> `base.sif`.

    Dizin VDS_SIF_DIZIN'den geliyor. Gorev dosyalarindaki `image:` alani
    Docker adi olarak kaliyor -- korpusu Singularity ugruna yeniden
    yazmak, ayni gorev tanimini iki motorda kosturamamak demekti.
    """
    ad = image.rsplit("/", 1)[-1].split(":", 1)[0]
    return Path(os.environ.get("VDS_SIF_DIZIN", ".")) / f"{ad}.sif"


class SingularitySandbox:
    """DockerSandbox ile ayni arayuz, HPC'de Singularity uzerinde.

    Docker'dan iki yapisal farki var:

    1. `.sif` salt okunur bir squashfs. Docker'in yazilabilir katmani
       yerine calisma dizini host'tan bind ediliyor; ustune de
       `--writable-tmpfs`, yoksa /workspace disina yazan gorevler
       Docker'da gecerken burada patlardi. Yan etkisi iyi: snapshot artik
       konteynerden kopyalama degil, dizin zaten host'ta.
    2. `--containall` sart. Singularity varsayilanda $HOME, $PWD ve
       /tmp'yi konteynere bagliyor -- onsuz modelin urettigi komut
       dogrudan GPFS home dizinine yazar. Docker'dakinden daha tehlikeli
       bir varsayilan oldugu icin secenek degil, sabit.

    Docker'daki --cpus/--memory/--pids-limit karsiligi yok; kaynak siniri
    isi saran Slurm cgroup'una kaliyor.
    """

    yerel = False

    def __init__(self, image: str, workdir: str = "/workspace",
                 network: str = "none", **_yoksay):
        self.image = image
        self.workdir = workdir
        self.network = network
        self.root: Path | None = None
        self.name: str | None = None

    # -- lifecycle ----------------------------------------------------

    def start(self) -> str:
        sif = sif_yolu(self.image)
        if not sif.exists():
            raise SandboxError(
                f"sif bulunamadi: {sif} (imaj: {self.image}). Dizustunde "
                f"`docker save {self.image} -o base.tar`, makinede "
                "`singularity build base.sif docker-archive://base.tar`; "
                "dizini VDS_SIF_DIZIN ile goster.")
        self.root = Path(tempfile.mkdtemp(prefix="vds-sing-"))
        ad = f"vds-{uuid.uuid4().hex[:12]}"
        args = [SINGULARITY, "instance", "start", "--containall",
                "--writable-tmpfs", "--bind", f"{self.root}:{self.workdir}"]
        if self.network == "none" and AG_IZOLE:
            args += ["--net", "--network", "none"]
        args += [str(sif), ad]
        proc = _run(args, timeout=120)
        if proc.returncode != 0:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None
            raise SandboxError(f"instance baslatilamadi: {proc.stderr.strip()}")
        self.name = ad
        return ad

    def stop(self) -> None:
        if self.name:
            _run([SINGULARITY, "instance", "stop", self.name], timeout=60)
            self.name = None
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None

    def __enter__(self) -> "SingularitySandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- operations ---------------------------------------------------

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        if not self.name:
            raise SandboxError("instance calismiyor")
        # --cleanenv sart ve exec'e ayrica verilmeli: --containall'i
        # `instance start`a vermek sonraki exec'lerin ortamini temizlemiyor.
        # Modul yuklu bir kabuktan kosunca host'un PYTHONHOME'u iceri sizip
        # konteynerin python'unu host stdlib'ini aramaya gonderiyordu --
        # "ModuleNotFoundError: No module named 'encodings'" ile olen
        # referans cozumler, gorev bozuk sanilacak sekilde FAIL veriyordu.
        # Yan faydasi: Docker da yalnizca imajin kendi ENV'ini verir, yani
        # bu iki motoru ayirmak yerine birbirine yaklastiriyor.
        args = [SINGULARITY, "exec", "--cleanenv", "--pwd", self.workdir,
                f"instance://{self.name}", "bash", "-lc", command]
        try:
            proc = _run(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="", exit_code=124, timed_out=True)
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr,
                          exit_code=proc.returncode)

    def yukle(self, kaynak: Path) -> None:
        if not self.root:
            raise SandboxError("instance calismiyor")
        _icerigi_kopyala(Path(kaynak), self.root)

    def snapshot(self, dest: Path) -> Path:
        if not self.root:
            raise SandboxError("instance calismiyor")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        _snapshot_kopyala(self.root, dest)
        return dest


# -- yerel sandbox ----------------------------------------------------
# Colab'de Docker yok: VM'in kendisi ayricaliksiz bir konteyner ve daemon
# kurulamiyor. Egitim oraya tasinacagi icin ayni harness'in Docker'siz
# kosabilmesi gerekiyor.
#
# IZOLASYON YOK. Komutlar dogrudan host'ta calisiyor; modelin urettigi
# `rm -rf /` gercekten zarar verir. Colab VM'i atilabilir oldugu icin bu
# kabul edildi, ama en ucuz korumalar konuldu: bariz yikici komutlarin
# reddi, per-komut zaman asimi ve HOME/TMPDIR'in kok dizine baglanmasi.
#
# `docker exec` semantigi BILEREK korunuyor: her komut taze bir kabukta
# kosuyor, yani `cd` ve export edilen degiskenler turlar arasi tasinmiyor.
# Bunu "duzeltmek" mevcut bant olcumlerini gecersiz kilardi.

MODLAR = ("docker", "singularity", "yerel")
VARSAYILAN_MOD = "docker"


def mod_sec(ad: str) -> None:
    """Bundan sonra acilan sandbox'larin motorunu sec."""
    global VARSAYILAN_MOD
    if ad not in MODLAR:
        raise ValueError(f"bilinmeyen sandbox modu: {ad}")
    VARSAYILAN_MOD = ad


def yerel_kullan(deger: bool = True) -> None:
    """Mod ikiliyken kalan cagri noktalari icin ince sarmalayici."""
    mod_sec("yerel" if deger else "docker")


# Tam bir guvenlik siniri degil -- kaza eseri yikimi durdurmak icin.
TEHLIKELI_DESENLER = [
    re.compile(kalip) for kalip in (
        r"\brm\s+(-[a-zA-Z]*\s+)*/(\s|$)",     # rm -rf /
        r"\brm\s+(-[a-zA-Z]*\s+)*(/etc|/usr|/bin|/var|/home|~)\b",
        r"\bmkfs\b", r"\bdd\s+[^|]*of=/dev/",
        r"\b(shutdown|reboot|halt|poweroff)\b",
        r">\s*/dev/(sd|nvme|hd)",
        r":\(\)\s*\{.*\|.*&.*\}\s*;",          # fork bomb
        r"\bsudo\b",
    )
]


class LocalSandbox:
    """DockerSandbox ile ayni arayuz, konteyner yerine gecici dizin."""

    yerel = True

    def __init__(self, image: str, workdir: str = "/workspace", **_yoksay):
        # image yalnizca raporlama icin tutuluyor; yerel modda katman yok.
        self.image = image
        self.workdir = workdir
        self.root: Path | None = None

    # -- lifecycle ----------------------------------------------------

    def start(self) -> str:
        self.root = Path(tempfile.mkdtemp(prefix="vds-yerel-"))
        return str(self.root)

    def stop(self) -> None:
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None

    def __enter__(self) -> "LocalSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- operations ---------------------------------------------------

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        if not self.root:
            raise SandboxError("sandbox baslatilmadi")
        for desen in TEHLIKELI_DESENLER:
            if desen.search(command):
                return ExecResult(
                    stdout="", stderr=f"komut reddedildi (yerel sandbox): {command}",
                    exit_code=126)
        ortam = dict(os.environ)
        # Kok disina yazma egilimini azalt: ev dizini ve gecici dizin
        # sandbox'in icinde.
        ortam.update({"HOME": str(self.root), "TMPDIR": str(self.root),
                      "PWD": str(self.root)})
        try:
            proc = subprocess.run(
                [BASH, "-lc", command], cwd=self.root, env=ortam,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="", exit_code=124, timed_out=True)
        except OSError as e:
            raise SandboxError(f"komut calistirilamadi: {e}") from e
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr,
                          exit_code=proc.returncode)

    def yukle(self, kaynak: Path) -> None:
        if not self.root:
            raise SandboxError("sandbox baslatilmadi")
        _icerigi_kopyala(Path(kaynak), self.root)

    def snapshot(self, dest: Path) -> Path:
        if not self.root:
            raise SandboxError("sandbox baslatilmadi")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        _snapshot_kopyala(self.root, dest)
        return dest


SINIFLAR = {"docker": DockerSandbox, "singularity": SingularitySandbox,
            "yerel": LocalSandbox}


def sandbox_yap(image: str, workdir: str = "/workspace",
                yerel: bool | None = None, mod: str | None = None):
    """Moda gore uc sandbox sinifindan birini dondur."""
    if mod is None:
        mod = VARSAYILAN_MOD if yerel is None else ("yerel" if yerel else "docker")
    return SINIFLAR[mod](image=image, workdir=workdir)


def bash_var() -> tuple[bool, str]:
    """Yerel modun tek on kosulu: bir bash bulunmasi."""
    return (True, BASH) if shutil.which("bash") else (False, "bash bulunamadi")


def singularity_var() -> tuple[bool, str]:
    """Singularity calistirilabiliyor mu (imaj varligi start'ta bakiliyor)."""
    try:
        proc = _run([SINGULARITY, "--version"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "singularity calistirilamadi"
    return True, proc.stdout.strip()


def sandbox_hazirla(mod: str) -> tuple[bool, str]:
    """Modu sec ve on kosulunu dogrula.

    Dort ayri giris noktasi ayni if/else'i kopyaliyordu; dorduncu mod
    eklenirken birini unutmak kacinilmazdi.
    """
    mod_sec(mod)
    if mod == "yerel":
        return bash_var()
    if mod == "singularity":
        ok, bilgi = singularity_var()
        if ok and not AG_IZOLE:
            bilgi += "  (ag izolasyonu kapali; VDS_AG_IZOLE=1 ile denenebilir)"
        return ok, bilgi
    return docker_available()


def docker_available() -> tuple[bool, str]:
    """Check that the Docker daemon is reachable."""
    try:
        proc = _run([DOCKER, "info", "--format", "{{.ServerVersion}}"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "docker daemon yanit vermiyor"
    return True, proc.stdout.strip()
