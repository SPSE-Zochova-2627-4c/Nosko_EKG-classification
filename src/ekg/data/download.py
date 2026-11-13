"""Stiahnutie PTB-XL v1.0.3 z PhysioNet s overením kontrolných súčtov (issue #3).

Spustenie:  uv run ekg-download            (iba 100 Hz časť + metadáta)
            uv run ekg-download --with-500hz
            uv run ekg-download --keep-zip  (ZIP sa po rozbalení nezmaže)

Postup:
1. stiahne ZIP (1,84 GB) do data/raw/ – prerušené sťahovanie pokračuje tam, kde skončilo,
2. stiahne oficiálny zoznam SHA256 súčtov z PhysioNet,
3. rozbalí iba potrebné súbory do data/raw/ptb-xl/,
4. overí SHA256 každého rozbaleného súboru; pri akejkoľvek chybe skončí s chybou,
5. zapíše data/raw/ptb-xl/DOWNLOAD_INFO.json (verzia, zdroj, súčet ZIP, dátum).

Opakované spustenie už stiahnuté dáta iba znovu overí.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from ekg import paths

VERSION = "1.0.3"
ZIP_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
SUMS_URL = f"https://physionet.org/files/ptb-xl/{VERSION}/SHA256SUMS.txt"
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
CITATION = (
    "Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). "
    "PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). "
    "PhysioNet. https://doi.org/10.13026/kfzx-aw45"
)
METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "RECORDS")

RAW_DIR = paths.DATA_DIR / "raw"
PTBXL_DIR = RAW_DIR / "ptb-xl"
ZIP_PATH = RAW_DIR / f"ptb-xl-{VERSION}.zip"
INFO_FILE = "DOWNLOAD_INFO.json"
CHUNK = 1 << 20  # 1 MB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stiahne URL do dest; ak existuje dest.part, pokračuje od jeho konca."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "ekg-maturita/0.1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and response.status != 206:  # server nepodporuje pokračovanie
            start = 0
        total = start + int(response.headers.get("Content-Length", 0))
        with (
            open(part, "ab" if start else "wb") as f,
            tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for block in iter(lambda: response.read(CHUNK), b""):
                f.write(block)
                bar.update(len(block))
    part.rename(dest)


def parse_sums(text: str) -> dict[str, str]:
    """'<sha256> <cesta>' na každom riadku -> {cesta: sha256}."""
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    return sums


def wanted(name: str, with_500hz: bool) -> bool:
    return (
        name in METADATA_FILES
        or name == "SHA256SUMS.txt"
        or name.startswith("records100/")
        or (with_500hz and name.startswith("records500/"))
    )


def extract(zip_path: Path, dest: Path, with_500hz: bool) -> int:
    """Rozbalí potrebné súbory; odstráni spoločný koreňový priečinok v ZIPe."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        roots = {m.filename.split("/", 1)[0] for m in members}
        prefix = roots.pop() + "/" if len(roots) == 1 else ""
        selected = [m for m in members if wanted(m.filename.removeprefix(prefix), with_500hz)]
        for member in tqdm(selected, desc="rozbaľujem", unit="súbor"):
            target = dest / member.filename.removeprefix(prefix)
            if dest.resolve() not in target.resolve().parents:  # ochrana pred ../ v ZIPe
                raise RuntimeError(f"Podozrivá cesta v ZIPe: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
    return len(selected)


def verify(dest: Path, sums: dict[str, str], with_500hz: bool) -> list[str]:
    """Vráti zoznam problémov (chýbajúci súbor / nesprávny súčet). Prázdny = všetko OK."""
    expected = {name: digest for name, digest in sums.items() if wanted(name, with_500hz)}
    problems = []
    for name, digest in tqdm(sorted(expected.items()), desc="overujem SHA256", unit="súbor"):
        path = dest / name
        if not path.is_file():
            problems.append(f"chýba: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"nesprávny SHA256: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stiahne a overí PTB-XL v1.0.3.")
    parser.add_argument("--with-500hz", action="store_true", help="rozbaliť aj 500 Hz záznamy")
    parser.add_argument("--keep-zip", action="store_true", help="po rozbalení nemazať ZIP")
    args = parser.parse_args(argv)

    PTBXL_DIR.mkdir(parents=True, exist_ok=True)
    sums_path = PTBXL_DIR / "SHA256SUMS.txt"
    info_path = PTBXL_DIR / INFO_FILE

    if not sums_path.exists():
        print(f"Sťahujem zoznam kontrolných súčtov: {SUMS_URL}")
        download(SUMS_URL, sums_path)
    sums = parse_sums(sums_path.read_text())

    already = info_path.exists() and (
        not args.with_500hz or json.loads(info_path.read_text()).get("with_500hz")
    )
    zip_sha = None
    if not already:
        if not ZIP_PATH.exists():
            print(f"Sťahujem PTB-XL {VERSION} (1,84 GB): {ZIP_URL}")
            download(ZIP_URL, ZIP_PATH)
        print("Počítam SHA256 ZIP súboru…")
        zip_sha = sha256_file(ZIP_PATH)
        count = extract(ZIP_PATH, PTBXL_DIR, args.with_500hz)
        print(f"Rozbalených súborov: {count}")
    else:
        print("Dataset už je stiahnutý – iba ho overujem.")

    problems = verify(PTBXL_DIR, sums, args.with_500hz)
    if problems:
        print(f"\nOverenie ZLYHALO ({len(problems)} problémov), prvých 20:")
        for problem in problems[:20]:
            print("  -", problem)
        print("Skús znova; ak problém pretrváva, zmaž data/raw/ a spusti sťahovanie odznova.")
        return 1

    if not already:
        info = {
            "dataset": "PTB-XL",
            "version": VERSION,
            "source": ZIP_URL,
            "checksums": SUMS_URL,
            "license": LICENSE,
            "citation": CITATION,
            "zip_sha256": zip_sha,
            "with_500hz": args.with_500hz,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
        if not args.keep_zip:
            ZIP_PATH.unlink()
            print("ZIP zmazaný (ponechaj ho cez --keep-zip).")

    n = sum(1 for name in sums if wanted(name, args.with_500hz))
    print(f"\nOK: všetkých {n} súborov má správny SHA256. Dáta sú v {PTBXL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
