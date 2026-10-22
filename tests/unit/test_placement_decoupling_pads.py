"""Placement gate measures decoupling pad-to-pad, not by footprint centroid (issue #1380)."""

from __future__ import annotations

from typing import Any

from kicad_mcp.tools.design_intent_state import DecouplingPairIntent, ProjectDesignIntent
from kicad_mcp.tools.validation import _PlacementEvaluator


def _pad(number: str, net: str, x: float, y: float, pintype: str = "") -> dict[str, object]:
    return {"number": number, "net": net, "pintype": pintype, "x_mm": x, "y_mm": y}


def _fp(
    x: float,
    y: float,
    *,
    name: str = "Package_SO:SOIC-8",
    value: str = "",
    pads: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "value": value,
        "x_mm": x,
        "y_mm": y,
        "width_mm": 2.0,
        "height_mm": 2.0,
    }
    if pads is not None:
        entry["pads"] = pads
        entry["pad_nets"] = {str(p["number"]): str(p["net"]) for p in pads if p["net"]}
        entry["net_names"] = sorted({str(p["net"]) for p in pads if p["net"]})
    return entry


def _evaluate(
    footprints: dict[str, dict[str, Any]],
    pairs: list[DecouplingPairIntent],
) -> tuple[list[str], int, list[str]]:
    evaluator = _PlacementEvaluator.__new__(_PlacementEvaluator)
    evaluator.footprints = footprints
    evaluator.intent = ProjectDesignIntent(decoupling_pairs=pairs)
    evaluator.warnings = []
    violations, checked = evaluator._evaluate_decoupling()
    return violations, checked, evaluator.warnings


def _pair(ic: str = "U1", caps: tuple[str, ...] = ("C1",), limit: float = 6.0):
    return DecouplingPairIntent(ic_ref=ic, cap_refs=list(caps), max_distance_mm=limit)


def _cap(x: float, y: float, supply_dx: float = -1.0, net: str = "+3V3") -> dict[str, Any]:
    return _fp(
        x,
        y,
        name="Capacitor_SMD:C_0603",
        value="100n",
        pads=[_pad("1", net, x + supply_dx, y), _pad("2", "GND", x - supply_dx, y)],
    )


def test_centroid_far_but_pad_near_cap_does_not_fail() -> None:
    # Centroids are 9 mm apart (over the 6 mm limit); the supply pads are 2.5 mm apart.
    footprints = {
        "U1": _fp(
            0.0,
            0.0,
            value="STM32F072",
            pads=[_pad("1", "+3V3", 4.0, 0.0, "power_in"), _pad("8", "GND", -4.0, 0.0)],
        ),
        "C1": _cap(9.0, 0.0, supply_dx=-2.5),
    }
    violations, checked, warnings = _evaluate(footprints, [_pair()])
    assert violations == []
    assert checked == 1
    assert not any("heuristic" in w for w in warnings)


def test_pad_far_same_net_cap_fails_with_per_pin_distance() -> None:
    footprints = {
        "U1": _fp(
            0.0,
            0.0,
            value="STM32F072",
            pads=[_pad("1", "+3V3", 1.0, 0.0, "power_in"), _pad("8", "GND", -1.0, 0.0)],
        ),
        "C1": _cap(20.0, 0.0),
    }
    violations, checked, _ = _evaluate(footprints, [_pair()])
    assert checked == 1
    assert len(violations) == 1
    message = violations[0]
    assert message.startswith("U1:")
    assert "nearest decoupling cap" in message
    assert "+3V3" in message
    assert "pin 1 -> C1.1" in message
    assert "18.00 mm" in message
    assert "limit 6.00 mm" in message
    assert "pad-centre" in message


def test_other_same_net_capacitor_on_board_satisfies_pair() -> None:
    # The inferred pair names C1 (wrong rail, centroid-nearest); C7 sits on +3V3 near the pin.
    footprints = {
        "U1": _fp(0.0, 0.0, pads=[_pad("1", "+3V3", 1.0, 0.0, "power_in")]),
        "C1": _cap(3.0, 0.0, net="+5V"),
        "C7": _cap(1.0, 3.0),
    }
    violations, _, _ = _evaluate(footprints, [_pair()])
    assert violations == []


def test_esd_protector_is_not_judged_by_generic_rule() -> None:
    footprints = {
        "U4": _fp(
            0.0,
            0.0,
            name="Package_TO_SOT_SMD:SOT-23-6",
            value="USBLC6-2SC6",
            pads=[_pad("5", "VBUS", 0.0, 0.0, "power_in"), _pad("2", "GND", 1.0, 0.0)],
        ),
        "C1": _cap(30.0, 0.0, net="VBUS"),
    }
    violations, checked, warnings = _evaluate(footprints, [_pair(ic="U4")])
    assert violations == []
    assert checked == 0
    assert len(warnings) == 1
    assert "ESD/TVS protection" in warnings[0]
    assert "USBLC6-2SC6" in warnings[0]
    assert "not applied" in warnings[0]


def test_missing_pad_data_falls_back_to_heuristic_warning() -> None:
    footprints = {
        "U1": _fp(10.0, 10.0, name="Package_QFP:TQFP-48"),
        "C1": _fp(30.0, 20.0, name="Capacitor_SMD:C_0603"),
    }
    violations, checked, warnings = _evaluate(footprints, [_pair(limit=4.0)])
    assert violations == []
    assert checked == 1
    assert len(warnings) == 1
    assert "heuristic" in warnings[0]
    assert "centroid" in warnings[0]
    assert "22.36 mm" in warnings[0]


def test_heuristic_centroid_within_limit_is_silent() -> None:
    footprints = {
        "U1": _fp(10.0, 10.0),
        "C1": _fp(12.0, 10.0, name="Capacitor_SMD:C_0603"),
    }
    violations, _, warnings = _evaluate(footprints, [_pair()])
    assert violations == []
    assert warnings == []


def test_far_pin_on_satisfied_net_is_warned_not_failed() -> None:
    footprints = {
        "U1": _fp(
            0.0,
            0.0,
            pads=[
                _pad("1", "+3V3", 0.0, 0.0, "power_in"),
                _pad("24", "+3V3", 0.0, 12.0, "power_in"),
            ],
        ),
        "C1": _cap(2.0, 0.0),
    }
    violations, _, warnings = _evaluate(footprints, [_pair()])
    assert violations == []
    assert len(warnings) == 1
    assert "24" in warnings[0]
    assert "+3V3" in warnings[0]


def test_explicit_cap_missing_from_board_still_fails() -> None:
    footprints = {"U1": _fp(0.0, 0.0, pads=[_pad("1", "+3V3", 0.0, 0.0, "power_in")])}
    violations, _, _ = _evaluate(footprints, [_pair(caps=("C9",))])
    assert any("missing or unresolved decoupling caps -> C9" in v for v in violations)


def test_ic_missing_from_board_still_fails() -> None:
    violations, checked, _ = _evaluate({"C1": _cap(0.0, 0.0)}, [_pair()])
    assert violations == ["Decoupling IC ref 'U1' is missing on the board."]
    assert checked == 0
