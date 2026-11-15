"""M5.4 历史索引/配置持久化纯模块测试（task_history）。

覆盖：登记/去重/最新在前/保留首次 created_at、终止状态更新（缺失静默）、查找、
条数上限、缺失/损坏优雅降级、配置读写回环。不导入 Tkinter，不触碰真实任务目录。
"""
from __future__ import annotations

import json

from bookclean import task_history as th


def test_record_and_find(tmp_path):
    p = tmp_path / "idx.json"
    e = th.record_task(p, task_dir="/t/a", output_path="/out/a.pdf",
                       settings={"preset": "auto"}, label="A", last_state="running")
    assert e["task_dir"] == "/t/a"
    assert e["output_path"] == "/out/a.pdf"
    assert e["settings"] == {"preset": "auto"}
    assert e["label"] == "A"
    assert e["last_state"] == "running"
    assert e["created_at"]
    found = th.find_entry(p, "/t/a")
    assert found is not None and found["label"] == "A"
    assert th.find_entry(p, "/t/none") is None


def test_upsert_dedup_preserves_created_at(tmp_path):
    p = tmp_path / "idx.json"
    e1 = th.record_task(p, task_dir="/t/a", label="A", last_state="running")
    created = e1["created_at"]
    th.record_task(p, task_dir="/t/b", label="B")
    e2 = th.record_task(p, task_dir="/t/a", label="A2", last_state="completed",
                        output_path="/out/a.pdf")
    entries = th.load_history(p)
    assert len(entries) == 2  # 去重，不新增
    assert e2["created_at"] == created  # 保留首次创建时间
    assert entries[0]["task_dir"] == "/t/a"  # 更新的条目置顶（最新在前）
    assert entries[0]["last_state"] == "completed"
    assert entries[0]["label"] == "A2"


def test_update_state_existing_and_absent(tmp_path):
    p = tmp_path / "idx.json"
    th.record_task(p, task_dir="/t/a", last_state="running")
    th.update_task_state(p, "/t/a", "cancelled")
    assert th.find_entry(p, "/t/a")["last_state"] == "cancelled"
    th.update_task_state(p, "/t/a", "completed", output_path="/out/a.pdf")
    e = th.find_entry(p, "/t/a")
    assert e["last_state"] == "completed" and e["output_path"] == "/out/a.pdf"
    # 缺失条目静默跳过，不新增
    th.update_task_state(p, "/t/none", "failed")
    assert th.find_entry(p, "/t/none") is None
    assert len(th.load_history(p)) == 1


def test_max_entries_cap(tmp_path):
    p = tmp_path / "idx.json"
    for i in range(th.MAX_ENTRIES + 10):
        th.record_task(p, task_dir=f"/t/{i}", label=str(i))
    entries = th.load_history(p)
    assert len(entries) == th.MAX_ENTRIES
    assert entries[0]["task_dir"] == f"/t/{th.MAX_ENTRIES + 9}"  # 最新在前


def test_load_history_missing_and_corrupt(tmp_path):
    assert th.load_history(tmp_path / "none.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert th.load_history(bad) == []
    wrong_type = tmp_path / "wrong.json"
    wrong_type.write_text(json.dumps({"a": 1}), encoding="utf-8")  # dict 而非 list
    assert th.load_history(wrong_type) == []
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps([{"task_dir": "/t/a"}, "junk", 5, {"no_dir": 1}]), encoding="utf-8")
    assert th.load_history(mixed) == [{"task_dir": "/t/a"}]  # 过滤非 dict/无 task_dir


def test_config_roundtrip_and_corrupt(tmp_path):
    p = tmp_path / "cfg.json"
    assert th.load_config(p) == {}
    th.save_config(p, {"preset": "old_book", "compression": "compact"})
    assert th.load_config(p) == {"preset": "old_book", "compression": "compact"}
    p.write_text("[]", encoding="utf-8")  # 类型不符 → 空 dict
    assert th.load_config(p) == {}
    p.write_text("{ broken", encoding="utf-8")
    assert th.load_config(p) == {}


def test_label_defaults_to_dir_name(tmp_path):
    p = tmp_path / "idx.json"
    e = th.record_task(p, task_dir="/t/task_20261113_000000_abc123")
    assert e["label"] == "task_20261113_000000_abc123"