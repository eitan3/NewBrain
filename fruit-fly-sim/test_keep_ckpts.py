"""--keep-ckpts: only the newest N numbered checkpoints stay; last.pt / final.pt always stay. CPU only, no torch work.

    python -m pytest test_keep_ckpts.py -q
"""
import os

import pytest

import train_lm


def _fake_run(tmp_path, steps):
    for s in steps:
        (tmp_path / f"ckpt_{s}.pt").write_bytes(b"x")
    for n in ("last.pt", "final.pt", "train.log", "ckpt_best.pt"):
        (tmp_path / n).write_bytes(b"x")
    return tmp_path


@pytest.fixture
def trashed(monkeypatch):
    """The Recycle Bin call is mocked: records the path and deletes the file."""
    got = []

    def fake(p):
        got.append(os.path.basename(str(p)))
        os.remove(p)

    monkeypatch.setattr(train_lm, "send_to_trash", fake)
    return got


def _names(d):
    return sorted(p.name for p in d.iterdir())


def test_keeps_newest_n(tmp_path, trashed):
    run = _fake_run(tmp_path, [250, 500, 1000, 1500, 2000, 2500, 3000, 10000])  # 10000 > 3000 numerically, not lexically
    removed = train_lm.prune_ckpts(run, 3)
    assert sorted(trashed) == ["ckpt_1000.pt", "ckpt_1500.pt", "ckpt_2000.pt", "ckpt_250.pt", "ckpt_500.pt"]
    assert len(removed) == 5
    assert _names(run) == sorted(["ckpt_2500.pt", "ckpt_3000.pt", "ckpt_10000.pt",
                                  "last.pt", "final.pt", "train.log", "ckpt_best.pt"])


def test_zero_keeps_all(tmp_path, trashed):
    run = _fake_run(tmp_path, [1, 2, 3, 4, 5, 6, 7])
    before = _names(run)
    assert train_lm.prune_ckpts(run, 0) == []
    assert trashed == [] and _names(run) == before


def test_fewer_than_n(tmp_path, trashed):
    run = _fake_run(tmp_path, [100, 200])
    train_lm.prune_ckpts(run, 5)
    assert trashed == [] and "ckpt_100.pt" in _names(run)


def test_trash_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    run = _fake_run(tmp_path, [1, 2, 3])

    def boom(p):
        raise OSError("nope")

    monkeypatch.setattr(train_lm, "send_to_trash", boom)
    assert train_lm.prune_ckpts(run, 1) == []
    assert "could not remove ckpt_1.pt" in capsys.readouterr().out
