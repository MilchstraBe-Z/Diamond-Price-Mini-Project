import json

from diamonds.cli import main


def test_cli_writes_report(sample_csv, tmp_path, capsys):
    output = tmp_path / "report.json"

    exit_code = main(["train", "--csv", str(sample_csv), "--output", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) == {"data", "metrics", "top_features"}
    assert set(report["metrics"]) == {"baseline", "forest"}
    assert set(report["metrics"]["forest"]) == {"r2", "mae"}
    assert len(report["top_features"]) == 5
    assert "forest" in capsys.readouterr().out


def test_cli_bad_csv_exits_with_error(tmp_path, capsys):
    exit_code = main(["train", "--csv", str(tmp_path / "missing.csv")])

    assert exit_code == 1
    assert "error" in capsys.readouterr().err
