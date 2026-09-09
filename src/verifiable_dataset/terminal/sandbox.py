"""Docker-backed terminal sandbox.

One container per episode, kept alive across turns so filesystem state
persists. Commands run via ``docker exec``, which means the working
directory and exported environment variables do NOT carry over between
turns -- each exec is a fresh shell. That is fine for filesystem tasks;
tasks that need a persistent venv or background processes will need a
PTY-attached session instead (see notes in README).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

DOCKER = shutil.which("docker") or "docker"
# Cozumlenmis yol sart: Windows'ta ciplak "bash" WSL'in bash.exe'sine
# gidiyor ve Windows bicimli calisma dizinini goremiyor. Colab'de zaten
# /bin/bash.
BASH = shutil.which("bash") or "bash"


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

VARSAYILAN_YEREL = False


def yerel_kullan(deger: bool = True) -> None:
    """Bundan sonra acilan sandbox'lar yerel olsun (Docker yerine)."""
    global VARSAYILAN_YEREL
    VARSAYILAN_YEREL = deger


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

    def snapshot(self, dest: Path) -> Path:
        if not self.root:
            raise SandboxError("sandbox baslatilmadi")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root, dest, symlinks=True)
        return dest


def sandbox_yap(image: str, workdir: str = "/workspace",
                yerel: bool | None = None):
    """Moda gore DockerSandbox ya da LocalSandbox dondur."""
    if yerel is None:
        yerel = VARSAYILAN_YEREL
    return (LocalSandbox if yerel else DockerSandbox)(image=image, workdir=workdir)


def bash_var() -> tuple[bool, str]:
    """Yerel modun tek on kosulu: bir bash bulunmasi."""
    return (True, BASH) if shutil.which("bash") else (False, "bash bulunamadi")


def docker_available() -> tuple[bool, str]:
    """Check that the Docker daemon is reachable."""
    try:
        proc = _run([DOCKER, "info", "--format", "{{.ServerVersion}}"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "docker daemon yanit vermiyor"
    return True, proc.stdout.strip()
