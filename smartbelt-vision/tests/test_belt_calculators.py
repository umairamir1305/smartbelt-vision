"""Tests for belt_calculators.py: the Belt Friction Calculator and the Power Difference Calculator (not from the video)."""

import math

import pytest

import belt_calculators


def run(monkeypatch, capsys, *arguments):
    monkeypatch.setattr("sys.argv", ["belt_calculators.py", *[str(argument) for argument in arguments]])
    code = belt_calculators.main()
    return code, capsys.readouterr().out


def value_after(label, text):
    line = [line for line in text.splitlines() if line.startswith(label)][0]
    return line.split(" : ")[1].split()[0]


def test_friction_coefficient_with_a_wrap_angle_in_degrees(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, "friction", "--t1", 120, "--t2", 60, "--wrap-deg", 180)
    assert code == 0
    assert "Belt Friction Calculator (model-based, NOT measured from the video)" in out
    assert float(value_after("Friction coefficient mu", out)) == pytest.approx(math.log(2) / math.pi, abs=1e-4)  # 0.2206
    assert value_after("Tension ratio T1 / T2", out) == "2.000"
    assert "180.0 degrees" in out and "(typed in)" in out
    assert "The video cannot measure it" in out


def test_friction_coefficient_with_the_wrap_angle_from_the_geometry(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, "friction", "--t1", 120, "--t2", 60, "--r-small-mm", 25, "--r-large-mm", 50,
                    "--centre-distance-mm", 200)
    assert code == 0
    wrap = math.pi - 2 * math.asin(25 / 200)  # small pulley
    assert float(value_after("Friction coefficient mu", out)) == pytest.approx(math.log(2) / wrap, abs=1e-4)
    assert "worked out for the small pulley of an open belt" in out

    code, out = run(monkeypatch, capsys, "friction", "--t1", 120, "--t2", 60, "--r-small-mm", 25, "--r-large-mm", 50,
                    "--centre-distance-mm", 200, "--pulley", "large")
    assert float(value_after("Friction coefficient mu", out)) == pytest.approx(math.log(2) / (math.pi + 2 * math.asin(25 / 200)), abs=1e-4)
    assert "large pulley" in out


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["friction", "--t1", 50, "--t2", 60, "--wrap-deg", 180], "must not be smaller"),
        (["friction", "--t1", 120, "--t2", 60], "Give the wrap angle"),
        (["friction", "--t1", 120, "--t2", 60, "--r-small-mm", 25, "--r-large-mm", 50], "Give the wrap angle"),
        (["friction", "--t1", 120, "--t2", 0, "--wrap-deg", 180], "must be positive"),
        (["friction", "--t1", 120, "--t2", 60, "--wrap-deg", 0], "wrap angle must be positive"),
        (["friction", "--t1", 120, "--t2", 60, "--r-small-mm", 25, "--r-large-mm", 50, "--centre-distance-mm", 20], "centre distance"),
        (["power", "--torque-in", 0, "--rpm-in", 120, "--torque-out", 1, "--rpm-out", 60], "input power"),
        (["power", "--torque-in", -1, "--rpm-in", 120, "--torque-out", 1, "--rpm-out", 60], "cannot be negative"),
    ],
)
def test_bad_values_give_a_plain_error(monkeypatch, capsys, arguments, message):
    code, out = run(monkeypatch, capsys, *arguments)
    assert code == 1 and message in out


def test_power_difference(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, "power", "--torque-in", 2.0, "--rpm-in", 120, "--torque-out", 3.7, "--rpm-out", 60)
    assert code == 0
    assert "Power Difference Calculator (model-based, NOT measured from the video)" in out
    assert float(value_after("Input power", out)) == pytest.approx(25.13, abs=0.01)
    assert float(value_after("Output power", out)) == pytest.approx(23.25, abs=0.01)
    assert float(value_after("Estimated power difference", out)) == pytest.approx(1.88, abs=0.01)
    assert float(value_after("Estimated efficiency", out)) == pytest.approx(92.5, abs=0.1)
    assert "The video cannot measure torque or power" in out


def test_the_calculators_never_claim_to_be_measured_from_the_video(monkeypatch, capsys):
    for arguments in (["friction", "--t1", 120, "--t2", 60, "--wrap-deg", 180],
                      ["power", "--torque-in", 2.0, "--rpm-in", 120, "--torque-out", 3.7, "--rpm-out", 60]):
        _, out = run(monkeypatch, capsys, *arguments)
        assert "NOT measured from the video" in out
        assert "typed in" in out
