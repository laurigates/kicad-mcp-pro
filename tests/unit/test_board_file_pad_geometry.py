"""Board parser exposes per-pad board coordinates, nets and pin types (issue #1380)."""

from __future__ import annotations

import pytest

from kicad_mcp.tools.board_file import _parse_board_footprint_blocks

_BOARD = """(kicad_pcb
\t(version 20250216)
\t(footprint "Package_QFP:TQFP-48"
\t\t(layer "F.Cu")
\t\t(at 100 200 90)
\t\t(property "Reference" "U1" (at 0 0 0) (layer "F.SilkS"))
\t\t(property "Value" "MCU" (at 0 1 0) (layer "F.Fab"))
\t\t(pad "1" smd rect (at 1 0 90) (size 1 1) (layers "F.Cu") (net 1 "+3V3") (pintype "power_in"))
\t\t(pad "2" smd rect (at 0 2) (size 1 1) (layers "F.Cu") (net 2 "GND"))
\t\t(pad "49" smd rect (at 0 0) (size 3 3) (layers "F.Cu") (net 2 "GND"))
\t\t(pad "49" smd rect (at 0.5 0.5) (size 1 1) (layers "F.Cu") (net 2 "GND"))
\t\t(pad "" np_thru_hole circle (at -3 -3) (size 1 1) (layers "*.Cu"))
\t)
)
"""


def _pads() -> list[dict[str, object]]:
    footprints = _parse_board_footprint_blocks(_BOARD)
    return list(footprints["U1"]["pads"])


def test_pad_positions_are_transformed_to_board_coordinates() -> None:
    pad1 = _pads()[0]
    assert pad1["number"] == "1"
    assert pad1["net"] == "+3V3"
    assert pad1["pintype"] == "power_in"
    # Same transform as the courtyard bounds: local (1, 0) at 90 deg -> (100, 199).
    assert pad1["x_mm"] == pytest.approx(100.0)
    assert pad1["y_mm"] == pytest.approx(199.0)


def test_pad_without_pintype_reports_empty_string() -> None:
    pad2 = _pads()[1]
    assert pad2["pintype"] == ""
    assert pad2["net"] == "GND"
    # local (0, 2) at 90 deg -> (102, 200)
    assert pad2["x_mm"] == pytest.approx(102.0)
    assert pad2["y_mm"] == pytest.approx(200.0)


def test_duplicate_pad_numbers_and_netless_pads_are_kept() -> None:
    pads = _pads()
    assert [pad["number"] for pad in pads] == ["1", "2", "49", "49", ""]
    assert pads[-1]["net"] == ""


def test_existing_pad_net_keys_are_unchanged() -> None:
    entry = _parse_board_footprint_blocks(_BOARD)["U1"]
    assert entry["net_names"] == ["+3V3", "GND"]
    assert entry["pad_nets"] == {"1": "+3V3", "2": "GND", "49": "GND"}


def test_footprint_without_root_position_has_no_pad_geometry() -> None:
    board = (
        "(kicad_pcb\n"
        '\t(footprint "R"\n'
        '\t\t(property "Reference" "R1" (at 0 0 0))\n'
        '\t\t(pad "1" smd rect (at 0 0) (size 1 1) (net 1 "N"))\n'
        "\t)\n)\n"
    )
    assert _parse_board_footprint_blocks(board)["R1"]["pads"] == []
