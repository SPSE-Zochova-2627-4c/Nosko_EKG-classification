"""Testy sťahovania a načítania PTB-XL na malom umelom datasete (bez internetu)."""

import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest
import wfdb

from ekg import contract
from ekg.data import download, ptbxl

# poradie zvodov schválne iné ako v kontrakte – loader ho musí opraviť
SHUFFLED_LEADS = list(reversed(contract.LEADS))


def write_fake_ptbxl(root, n_records=3):
    """Vytvorí miniatúrny PTB-XL: 2 CSV + n WFDB záznamov 100 Hz."""
    rng = np.random.default_rng(0)
    rows = []
    for ecg_id in range(1, n_records + 1):
        rel = f"records100/00000/{ecg_id:05d}_lr"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        signal = rng.normal(0, 0.2, size=(1000, 12)).round(3)
        wfdb.wrsamp(
            record_name=f"{ecg_id:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=SHUFFLED_LEADS,
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str((root / rel).parent),
        )
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": 100.0 + ecg_id,
                "scp_codes": "{'NORM': 100.0, 'SR': 0.0}",
                "filename_lr": rel,
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"description": ["normal ECG", "sinus rhythm"], "diagnostic_class": ["NORM", None]},
        index=pd.Index(["NORM", "SR"]),
    ).to_csv(root / "scp_statements.csv")
    for name in ("LICENSE.txt", "RECORDS"):
        (root / name).write_text("test\n")


def make_sums(root):
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()} {p.relative_to(root).as_posix()}"
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "ptb-xl"
    write_fake_ptbxl(root)
    return root


# --- načítanie ---------------------------------------------------------------


def test_load_database_parses_scp_codes(fake_root):
    db = ptbxl.load_database(fake_root)
    assert len(db) == 3
    assert db.index.name == "ecg_id"
    assert db.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}


def test_load_signal_matches_contract_and_reorders_leads(fake_root):
    x = ptbxl.load_signal("records100/00000/00001_lr", fake_root)
    assert x.shape == (12, 1000) and x.dtype == np.float32
    raw = wfdb.rdrecord(str(fake_root / "records100/00000/00001_lr"))
    # zvod "I" bol v súbore posledný
    np.testing.assert_allclose(x[0], raw.p_signal[:, 11], atol=1e-6)


def test_verify_dataset_ok(fake_root):
    assert ptbxl.verify_dataset(fake_root) == []


def test_verify_dataset_reports_missing_record(fake_root):
    (fake_root / "records100/00000/00002_lr.hea").unlink()
    problems = ptbxl.verify_dataset(fake_root)
    assert len(problems) == 1 and "ecg_id 2" in problems[0]


def test_verify_dataset_reports_unknown_scp_code(fake_root):
    db = pd.read_csv(fake_root / "ptbxl_database.csv")
    db.loc[0, "scp_codes"] = "{'XYZ': 50.0}"
    db.to_csv(fake_root / "ptbxl_database.csv", index=False)
    assert any("XYZ" in p for p in ptbxl.verify_dataset(fake_root))


def test_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(ptbxl.DatasetNotFoundError, match="ekg-download"):
        ptbxl.load_database(tmp_path)


# --- sťahovanie: rozbalenie a overenie súčtov --------------------------------


@pytest.fixture
def fake_zip(tmp_path, fake_root):
    """ZIP so spoločným koreňovým priečinkom a 500 Hz súborom navyše, ako na PhysioNet."""
    (fake_root / "records500/00000").mkdir(parents=True)
    (fake_root / "records500/00000/00001_hr.dat").write_bytes(b"500hz")
    sums = download.parse_sums(make_sums(fake_root))
    zip_path = tmp_path / "ptb-xl.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in fake_root.rglob("*"):
            if p.is_file():
                zf.write(p, f"ptb-xl-1.0.3/{p.relative_to(fake_root).as_posix()}")
    return zip_path, sums


def test_extract_only_100hz_and_verify(tmp_path, fake_zip):
    zip_path, sums = fake_zip
    dest = tmp_path / "out"
    download.extract(zip_path, dest, with_500hz=False)
    assert (dest / "ptbxl_database.csv").is_file()
    assert (dest / "records100/00000/00001_lr.dat").is_file()
    assert not (dest / "records500").exists()
    assert download.verify(dest, sums, with_500hz=False) == []


def test_verify_detects_corrupted_and_missing(tmp_path, fake_zip):
    zip_path, sums = fake_zip
    dest = tmp_path / "out"
    download.extract(zip_path, dest, with_500hz=False)
    (dest / "records100/00000/00001_lr.dat").write_bytes(b"poskodene")
    (dest / "scp_statements.csv").unlink()
    problems = download.verify(dest, sums, with_500hz=False)
    assert "nesprávny SHA256: records100/00000/00001_lr.dat" in problems
    assert "chýba: scp_statements.csv" in problems


def test_extract_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "zly.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ptb-xl/records100/../../../mimo.txt", "x")
        zf.writestr("ptb-xl/ptbxl_database.csv", "x")
    with pytest.raises(RuntimeError, match="Podozrivá"):
        download.extract(zip_path, tmp_path / "out", with_500hz=False)


def test_parse_sums():
    text = "abc123 ptbxl_database.csv\n\ndef456 records100/00000/00001_lr.dat\n"
    assert download.parse_sums(text) == {
        "ptbxl_database.csv": "abc123",
        "records100/00000/00001_lr.dat": "def456",
    }
