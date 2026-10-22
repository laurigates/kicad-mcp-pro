"""Regression tests for upstream issue #1388: routing must never short USB D+ and D-.

An older server let ``sch_route_wire_between_pins`` write a route after the router
reported ``obstacle_bypass_failed``; the STM32F072 benchmark ended up with the
USB-C D+/D- pins (J1:A6/A7/B6/B7) and the USBLC6-2SC6 I/O pins merged into one
net.  These tests use hermetic stub libraries shaped like the benchmark parts and
assert that every route either keeps D+ and D- apart or is refused as a failed
tool result that leaves the schematic byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.types import CallToolResult

from kicad_mcp.server import build_server
from kicad_mcp.tools import schematic as schematic_module
from kicad_mcp.tools.schematic import _build_connectivity_groups, get_pin_positions
from tests.conftest import call_tool_text

pytestmark = pytest.mark.anyio

# USB-C receptacle stub: the duplicate A/B data pins interleave on one edge, the
# way they do on real USB2.0 receptacle symbols, so D+ routes must pass D- pins.
USB_C_LIB = """(kicad_symbol_lib (version 20250316) (generator pytest)
  (symbol "USB_C_Receptacle_USB2.0_16P"
    (property "Reference" "J" (id 0) (at 0 12.7 0))
    (property "Value" "USB_C_Receptacle_USB2.0_16P" (id 1) (at 0 -12.7 0))
    (symbol "USB_C_Receptacle_USB2.0_16P_0_1"
      (rectangle (start -7.62 -10.16) (end 7.62 10.16))
    )
    (symbol "USB_C_Receptacle_USB2.0_16P_1_1"
      (pin bidirectional line (at 10.16 5.08 180) (length 2.54) (name "D+") (number "A6"))
      (pin bidirectional line (at 10.16 2.54 180) (length 2.54) (name "D-") (number "A7"))
      (pin bidirectional line (at 10.16 0 180) (length 2.54) (name "D+") (number "B6"))
      (pin bidirectional line (at 10.16 -2.54 180) (length 2.54) (name "D-") (number "B7"))
    )
  )
)
"""

USBLC6_LIB = """(kicad_symbol_lib (version 20250316) (generator pytest)
  (symbol "USBLC6-2SC6"
    (property "Reference" "U" (id 0) (at 0 7.62 0))
    (property "Value" "USBLC6-2SC6" (id 1) (at 0 -7.62 0))
    (symbol "USBLC6-2SC6_0_1"
      (rectangle (start -5.08 -5.08) (end 5.08 5.08))
    )
    (symbol "USBLC6-2SC6_1_1"
      (pin passive line (at -7.62 2.54 0) (length 2.54) (name "I/O1") (number "1"))
      (pin power_in line (at -7.62 0 0) (length 2.54) (name "GND") (number "2"))
      (pin passive line (at -7.62 -2.54 0) (length 2.54) (name "I/O2") (number "3"))
      (pin passive line (at 7.62 -2.54 180) (length 2.54) (name "I/O2") (number "4"))
      (pin power_in line (at 7.62 0 180) (length 2.54) (name "VBUS") (number "5"))
      (pin passive line (at 7.62 2.54 180) (length 2.54) (name "I/O1") (number "6"))
    )
  )
)
"""

STM32_LIB = """(kicad_symbol_lib (version 20250316) (generator pytest)
  (symbol "STM32F072C8Tx"
    (property "Reference" "U" (id 0) (at 0 12.7 0))
    (property "Value" "STM32F072C8Tx" (id 1) (at 0 -12.7 0))
    (symbol "STM32F072C8Tx_0_1"
      (rectangle (start -10.16 -10.16) (end 10.16 10.16))
    )
    (symbol "STM32F072C8Tx_1_1"
      (pin bidirectional line (at -12.7 5.08 0) (length 2.54) (name "PA10") (number "31"))
      (pin bidirectional line (at -12.7 2.54 0) (length 2.54) (name "PA12") (number "33"))
      (pin bidirectional line (at -12.7 0 0) (length 2.54) (name "PA11") (number "32"))
      (pin bidirectional line (at -12.7 -2.54 0) (length 2.54) (name "PA9") (number "30"))
    )
  )
)
"""

J1 = ("Connector", "USB_C_Receptacle_USB2.0_16P", 50.8, 50.8)
U2 = ("Power_Protection", "USBLC6-2SC6", 88.9, 50.8)
U1 = ("MCU_ST_STM32F0", "STM32F072C8Tx", 127.0, 50.8)

DP_PINS = {("J1", "A6"), ("J1", "B6"), ("U2", "1"), ("U2", "6"), ("U1", "33")}
DM_PINS = {("J1", "A7"), ("J1", "B7"), ("U2", "3"), ("U2", "4"), ("U1", "32")}


def _write_libraries(sample_project: Path) -> None:
    symbols_dir = sample_project.parent / "symbols"
    (symbols_dir / "Connector.kicad_sym").write_text(USB_C_LIB, encoding="utf-8")
    (symbols_dir / "Power_Protection.kicad_sym").write_text(USBLC6_LIB, encoding="utf-8")
    (symbols_dir / "MCU_ST_STM32F0.kicad_sym").write_text(STM32_LIB, encoding="utf-8")


def _placement(part: tuple[str, str, float, float], reference: str) -> dict[str, Any]:
    library, symbol, x, y = part
    return {
        "library": library,
        "symbol_name": symbol,
        "x_mm": x,
        "y_mm": y,
        "reference": reference,
        "value": symbol,
    }


def _pin(part: tuple[str, str, float, float], number: str) -> tuple[float, float]:
    library, symbol, x, y = part
    return get_pin_positions(library, symbol, x, y, 0, 1)[number]


async def _usb_server(sample_project: Path) -> FastMCP:
    _write_libraries(sample_project)
    server = build_server("schematic")
    await call_tool_text(server, "kicad_set_project", {"project_dir": str(sample_project)})
    await call_tool_text(
        server,
        "sch_build_circuit",
        {"symbols": [_placement(J1, "J1"), _placement(U2, "U2"), _placement(U1, "U1")]},
    )
    return server


def _groups(sch_file: Path) -> list[set[tuple[str, str]]]:
    return [
        {(str(pin["reference"]), str(pin["pin"])) for pin in group["pins"]}
        for group in _build_connectivity_groups(sch_file)
    ]


def _assert_dp_dm_separate(sch_file: Path) -> None:
    for pins in _groups(sch_file):
        assert not (pins & DP_PINS and pins & DM_PINS), f"D+ and D- shorted: {sorted(pins)}"


async def _route(server: FastMCP, ref1: str, pin1: str, ref2: str, pin2: str) -> CallToolResult:
    result = await server.call_tool(
        "sch_route_wire_between_pins",
        {"ref1": ref1, "pin1": pin1, "ref2": ref2, "pin2": pin2},
    )
    assert isinstance(result, CallToolResult)
    return result


def _text(result: CallToolResult) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content)


async def _assert_refused_unchanged(
    server: FastMCP, sch_file: Path, ref1: str, pin1: str, ref2: str, pin2: str
) -> str:
    before = sch_file.read_bytes()
    result = await _route(server, ref1, pin1, ref2, pin2)
    text = _text(result)
    assert result.is_error is True, text
    assert text.startswith("SCHEMATIC_ROUTING_UNSAFE: Route refused"), text
    assert "No schematic changes were made." in text
    assert sch_file.read_bytes() == before
    return text


async def test_usb_dp_dm_routes_stay_separate(sample_project: Path, mock_kicad: MagicMock) -> None:
    server = await _usb_server(sample_project)
    sch_file = sample_project / "demo.kicad_sch"

    for ref1, pin1, ref2, pin2 in (
        ("J1", "A6", "U2", "1"),
        ("J1", "A7", "U2", "3"),
        ("J1", "B6", "U2", "1"),
        ("J1", "B7", "U2", "3"),
        ("U2", "6", "U1", "33"),
        ("U2", "4", "U1", "32"),
    ):
        # Every one of these routes has a clean path, so the post-route guard
        # must not over-refuse; a short after any step fails immediately.
        result = await _route(server, ref1, pin1, ref2, pin2)
        assert result.is_error is False, _text(result)
        assert "Routed" in _text(result)
        _assert_dp_dm_separate(sch_file)

    groups = _groups(sch_file)
    dp = next(pins for pins in groups if ("J1", "A6") in pins)
    dm = next(pins for pins in groups if ("J1", "A7") in pins)
    assert {("J1", "A6"), ("J1", "B6"), ("U2", "1")} <= dp
    assert {("J1", "A7"), ("J1", "B7"), ("U2", "3")} <= dm
    assert not dp & dm
    mcu_dp = next(pins for pins in groups if ("U2", "6") in pins)
    mcu_dm = next(pins for pins in groups if ("U2", "4") in pins)
    assert {("U2", "6"), ("U1", "33")} <= mcu_dp
    assert {("U2", "4"), ("U1", "32")} <= mcu_dm
    assert not mcu_dp & mcu_dm


async def test_route_refuses_named_usb_net_merge(
    sample_project: Path, mock_kicad: MagicMock
) -> None:
    server = await _usb_server(sample_project)
    sch_file = sample_project / "demo.kicad_sch"
    a6 = _pin(J1, "A6")
    u2_3 = _pin(U2, "3")
    await call_tool_text(server, "sch_add_label", {"name": "USB_DP", "x_mm": a6[0], "y_mm": a6[1]})
    await call_tool_text(
        server, "sch_add_label", {"name": "USB_DM", "x_mm": u2_3[0], "y_mm": u2_3[1]}
    )

    text = await _assert_refused_unchanged(server, sch_file, "J1", "A6", "U2", "3")

    assert "USB_DP" in text
    assert "USB_DM" in text
    _assert_dp_dm_separate(sch_file)


async def test_route_refuses_to_absorb_foreign_label_on_path(
    sample_project: Path, mock_kicad: MagicMock
) -> None:
    server = await _usb_server(sample_project)
    sch_file = sample_project / "demo.kicad_sch"
    start = _pin(U2, "6")
    end = _pin(U1, "33")
    assert start[1] == end[1]  # the direct run is a single straight segment
    await call_tool_text(
        server,
        "sch_add_label",
        {"name": "USB_DM", "x_mm": round((start[0] + end[0]) / 2 / 1.27) * 1.27, "y_mm": start[1]},
    )

    text = await _assert_refused_unchanged(server, sch_file, "U2", "6", "U1", "33")

    assert "USB_DM" in text


def _crafted_route(*points: tuple[float, float]) -> list[tuple[float, float, float, float]]:
    return [(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:], strict=False)]


def _defect_routes() -> dict[str, tuple[list[tuple[float, float, float, float]], str]]:
    """Router outputs (U2:6 -> U1:33, D+) that would each join the D- net or a foreign net."""
    s = _pin(U2, "6")  # D+ start
    e = _pin(U1, "33")  # D+ end (PA12)
    pa11 = _pin(U1, "32")  # D- (PA11), one pitch below PA12
    dm = _pin(U2, "4")  # D- wire runs right from here
    lane = dm[1]  # y of the pre-drawn D- wire
    dm_end_x = dm[0] + 10.16
    above = s[1] - 2.54
    below = lane + 5.08
    x1 = s[0] + 2.54
    x2 = s[0] + 5.08
    x_exit = e[0] - 2.54
    return {
        "collinear_overlap": (
            _crafted_route(s, (x1, s[1]), (x1, lane), (x_exit, lane), (x_exit, e[1]), e),
            "U2:4",
        ),
        "t_junction_vertex": (
            _crafted_route(
                s, (x2, s[1]), (x2, lane), (x2, below), (x_exit, below), (x_exit, e[1]), e
            ),
            "U2:4",
        ),
        "ends_on_wire_endpoint": (
            _crafted_route(
                s,
                (dm_end_x, s[1]),
                (dm_end_x, lane),
                (dm_end_x, below),
                (x_exit, below),
                (x_exit, e[1]),
                e,
            ),
            "U2:4",
        ),
        "through_foreign_pin_tip": (
            _crafted_route(s, (x1, s[1]), (x1, below), (e[0], below), (e[0], pa11[1]), e),
            "U1:32",
        ),
        "through_label_anchor": (
            _crafted_route(s, (x1, s[1]), (x1, above), (x_exit, above), (x_exit, e[1]), e),
            "USB_DM",
        ),
        "through_power_symbol": (
            _crafted_route(s, (x1, s[1]), (x1, above), (x_exit, above), (x_exit, e[1]), e),
            "GND",
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "collinear_overlap",
        "t_junction_vertex",
        "ends_on_wire_endpoint",
        "through_foreign_pin_tip",
        "through_label_anchor",
        "through_power_symbol",
    ],
)
async def test_post_route_guard_catches_router_defects(
    sample_project: Path, mock_kicad: MagicMock, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    server = await _usb_server(sample_project)
    sch_file = sample_project / "demo.kicad_sch"
    dm = _pin(U2, "4")
    await call_tool_text(
        server,
        "sch_add_wire",
        {"x1_mm": dm[0], "y1_mm": dm[1], "x2_mm": dm[0] + 10.16, "y2_mm": dm[1]},
    )
    s = _pin(U2, "6")
    anchor = (s[0] + 10.16, s[1] - 2.54)
    if case == "through_label_anchor":
        await call_tool_text(
            server, "sch_add_label", {"name": "USB_DM", "x_mm": anchor[0], "y_mm": anchor[1]}
        )
    if case == "through_power_symbol":
        await call_tool_text(
            server, "sch_add_power_symbol", {"name": "GND", "x_mm": anchor[0], "y_mm": anchor[1]}
        )
    segments, foreign = _defect_routes()[case]
    monkeypatch.setattr(
        schematic_module,
        "_route_avoiding_obstacles",
        lambda *_args, **_kwargs: (list(segments), None),
    )

    text = await _assert_refused_unchanged(server, sch_file, "U2", "6", "U1", "33")

    assert "would merge another net" in text
    assert foreign in text
    _assert_dp_dm_separate(sch_file)


async def test_obstacle_bypass_failure_is_a_failed_tool_result(
    sample_project: Path, mock_kicad: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = await _usb_server(sample_project)
    sch_file = sample_project / "demo.kicad_sch"
    s = _pin(J1, "A6")
    e = _pin(U2, "3")
    monkeypatch.setattr(
        schematic_module,
        "_route_avoiding_obstacles",
        lambda *_args, **_kwargs: (
            _crafted_route(s, (s[0] + 2.54, s[1]), (s[0] + 2.54, e[1]), e),
            "WARNING: obstacle_bypass_failed",
        ),
    )

    text = await _assert_refused_unchanged(server, sch_file, "J1", "A6", "U2", "3")

    assert "obstacle_bypass_failed" in text
