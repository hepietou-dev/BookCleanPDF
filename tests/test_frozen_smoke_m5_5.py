"""M5.5 打包冒烟 harness（dev 模式）：在隔离目录复跑 frozen_smoke 全链路契约。

frozen 二进制验证由 `BookCleanPDF.exe --smoke` 承担；此处保证 harness 本身
（矩阵/重开渲染/续跑复用/异常文件/原件只读）在 dev 下可回归，不依赖打包产物。
"""

import json
from pathlib import Path

from bookclean.frozen_smoke import run_smoke


def test_frozen_smoke_dev_matrix(tmp_path, monkeypatch):
    monkeypatch.delenv("BOOKCLEAN_SMOKE_PAGES", raising=False)
    result = run_smoke(root=tmp_path)

    assert result["ok"], result
    assert result["frozen"] is False
    assert result["originals_unchanged"] is True
    assert result["frozen_paths_ok"] is True

    assert len(result["runs"]) == 4
    combos = {(r["color"], r["compression"]) for r in result["runs"]}
    assert combos == {("preserve", "balanced"), ("grayscale", "balanced"),
                      ("preserve", "compact"), ("preserve", "lossless")}
    for run in result["runs"]:
        assert run["ok"], run
        assert run["terminal"] == "completed"
        assert run["validation_status"] == "passed"
        assert run["total_pages"] == 3
        assert run["processed"] == 3
        assert run["fallbacks"] == 0

    assert result["reopen_render"]["ok"] is True
    assert result["reopen_render"]["page_count"] == 3
    assert result["resume_reuse"]["ok"] is True
    assert result["resume_reuse"]["reused"] == 3
    assert result["resume_reuse"]["newly_processed"] == 0
    assert result["error_file"]["ok"] is True

    on_disk = json.loads(
        (Path(result["data_root"]) / "smoke_result.json").read_text(encoding="utf-8"))
    assert on_disk["ok"] is True