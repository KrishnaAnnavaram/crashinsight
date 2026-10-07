"""The CLI works end to end on synthetic files. No network and no download."""
import pytest

from crashinsight.cli import main


@pytest.fixture()
def data_dir(tmp_path):
    assert main(["synth", "--out", str(tmp_path / "d"), "--crashes", "300", "--seed", "2"]) == 0
    return tmp_path / "d"


def test_validate_and_describe(data_dir, capsys):
    assert main(["validate", "--data", str(data_dir)]) == 0
    assert "codebook coverage: OK" in capsys.readouterr().out
    assert main(["describe", "--data", str(data_dir)]) == 0
    assert "KA rate by body_group" in capsys.readouterr().out


def test_validate_flags_unknown_codes(data_dir, capsys):
    path = data_dir / "VEHICLE.CSV"
    text = path.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    col = header.index("MAN_COLL")
    first = text[1].split(",")
    first[col] = "5"
    text[1] = ",".join(first)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert main(["validate", "--data", str(data_dir)]) == 2
    assert "MAN_COLL: codes without a label: [5]" in capsys.readouterr().out


def test_train_evaluate_report(data_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CRASHINSIGHT_N_SPLITS", "3")
    out = tmp_path / "model"
    assert main(["train", "--data", str(data_dir), "--model", "logreg", "--bootstrap", "10", "--out", str(out)]) == 0
    assert (out / "model_card.md").exists() and (out / "metrics.json").exists()
    assert main(["evaluate", "--data", str(data_dir), "--model", str(out / "model.joblib")]) == 0
    assert "ka_pr_auc" in capsys.readouterr().out
    report = tmp_path / "report.md"
    assert main(["report", "--data", str(data_dir), "--bootstrap", "4", "--out", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "Survey-weighted KA rate" in text and "majority baseline" in text


def test_age_effect_command(data_dir, capsys):
    assert main(["age-effect", "--data", str(data_dir), "--bootstrap", "4"]) == 0
    assert "Adjusted odds ratio" in capsys.readouterr().out


def test_missing_data_folder_is_a_clean_error(tmp_path, capsys):
    assert main(["validate", "--data", str(tmp_path / "none")]) == 1
    assert "crashinsight synth" in capsys.readouterr().err
