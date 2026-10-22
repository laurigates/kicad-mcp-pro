from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from kicad_mcp.errors import SchematicRoutingUnsafeError, ToolValidationError
from kicad_mcp.schematic.connectivity_authoring import (
    BoundingBoxLike,
    SchematicConnectivityAuthoringService,
    route_merge_violation,
)


@dataclass(frozen=True)
class FakeTarget:
    path: Path
    description: str = "root"


@dataclass(frozen=True)
class FakeBoundingBox:
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0


@dataclass
class ServiceHarness:
    service: SchematicConnectivityAuthoringService
    writes: list[tuple[Path | None, str]]
    calls: list[tuple[str, Any]]


def _harness(
    tmp_path: Path,
    *,
    parsed: dict[str, Any] | None = None,
    pin_positions: dict[tuple[str, str], dict[str, tuple[float, float]]] | None = None,
    aliases: dict[tuple[str, str], dict[str, tuple[float, float]]] | None = None,
    route_segments: list[tuple[float, float, float, float]] | None = None,
    route_warning: str | None = None,
    power_net: Callable[[str], bool] | None = None,
    power_lib: str | None = '(symbol "power:3V3")',
    connectivity_groups: Callable[[str], list[dict[str, Any]]] | None = None,
) -> ServiceHarness:
    target = FakeTarget(tmp_path / "board.kicad_sch")
    target.path.write_text("(kicad_sch\n\t(sheet_instances)\n)", encoding="utf-8")
    parsed_value = parsed or {"uuid": "root-uuid", "symbols": [], "power_symbols": []}
    position_map = pin_positions or {}
    alias_map = aliases or {}
    writes: list[tuple[Path | None, str]] = []
    calls: list[tuple[str, Any]] = []

    def transactional_write(mutator: Callable[[str], str], path: Path | None = None) -> str:
        before = target.path.read_text(encoding="utf-8")
        after = mutator(before)
        writes.append((path, after))
        return str(path or target.path)

    def wire_block(x1: float, y1: float, x2: float, y2: float) -> str:
        calls.append(("wire_block", (x1, y1, x2, y2)))
        return f"WIRE({x1},{y1}->{x2},{y2})"

    def label_block(
        name: str,
        x: float,
        y: float,
        rotation: int = 0,
        global_label: bool = False,
        shape: str | None = None,
        kind: str | None = None,
        justify: str | None = None,
    ) -> str:
        effective_kind = kind or ("global_label" if global_label else "label")
        calls.append(("label_block", (name, x, y, rotation, effective_kind, shape)))
        return f"LABEL({name},{x},{y},{rotation},{effective_kind},{shape})"

    def place_symbol_block(
        *,
        lib_id: str,
        x: float,
        y: float,
        reference: str,
        value: str,
        rotation: int,
        project_name: str,
        root_uuid: str,
    ) -> str:
        payload: dict[str, object] = {
            "lib_id": lib_id,
            "x": x,
            "y": y,
            "reference": reference,
            "value": value,
            "rotation": rotation,
            "project_name": project_name,
            "root_uuid": root_uuid,
        }
        calls.append(("place_symbol_block", payload))
        return f"POWER({value},{x},{y})"

    def append_before_sheet_instances(content: str, block: str) -> str:
        return content.replace("\t(sheet_instances)", f"\t{block}\n\t(sheet_instances)", 1)

    def get_positions(
        library: str,
        symbol_name: str,
        _x: float,
        _y: float,
        _rotation: int,
        _unit: int,
    ) -> dict[str, tuple[float, float]]:
        return dict(position_map.get((library, symbol_name), {}))

    def get_aliases(
        library: str,
        symbol_name: str,
        _x: float,
        _y: float,
        _rotation: int,
        _unit: int,
    ) -> dict[str, tuple[float, float]]:
        return dict(alias_map.get((library, symbol_name), {}))

    def route(
        start: tuple[float, float],
        end: tuple[float, float],
        obstacles: list[BoundingBoxLike],
        snap_to_grid: bool,
        occupied: list[tuple[float, float, float, float]] | None = None,
        start_normal: tuple[float, float] | None = None,
        end_normal: tuple[float, float] | None = None,
        boundaries: list[BoundingBoxLike] | None = None,
    ) -> tuple[list[tuple[float, float, float, float]], str | None]:
        calls.append(
            (
                "route",
                (
                    start,
                    end,
                    obstacles,
                    snap_to_grid,
                    occupied,
                    start_normal,
                    end_normal,
                    boundaries,
                ),
            )
        )
        return list(route_segments or []), route_warning

    def split_lib_id(lib_id: str) -> tuple[str, str]:
        library, symbol_name = lib_id.split(":", 1)
        return library, symbol_name

    optional: dict[str, Any] = {}
    if connectivity_groups is not None:
        optional["route_connectivity_groups"] = connectivity_groups

    service = SchematicConnectivityAuthoringService(
        resolve_target=lambda sheet, sheet_file: target,
        parse_schematic=lambda path: parsed_value,
        project_name=lambda: "Demo",
        new_uuid=lambda: "abcd-1234",
        get_pin_positions=get_positions,
        get_pin_alias_positions=get_aliases,
        pin_label_stub_direction=lambda point, origin, all_points: (1.0, 0.0),
        is_origin_pin_power_symbol=lambda symbol_name, value: True,
        is_power_net=power_net or (lambda name: name in {"3V3", "GND"}),
        load_lib_symbol=lambda library, name: power_lib,
        wire_block=wire_block,
        power_symbol_rotation_from_vector=lambda ux, uy: 270,
        place_symbol_block=place_symbol_block,
        terminal_rotation_from_vector=lambda ux, uy: 0,
        label_block=label_block,
        append_before_sheet_instances=append_before_sheet_instances,
        transactional_write=transactional_write,
        reload_schematic=lambda: "Reloaded",
        format_target_detail=lambda resolved: f"Target schematic: {resolved.path.name}",
        active_schematic_file=lambda: target.path,
        split_lib_id=split_lib_id,
        get_symbol_bboxes=lambda content: [FakeBoundingBox()],
        foreign_wire_segments=lambda content, anchors: [],
        count_net_crossings=lambda segments, occupied: 0,
        get_pin_outward_normals=lambda library, symbol, rotation, unit: {},  # noqa: ARG005
        route_avoiding_obstacles=route,
        run_auto_add_missing_junctions=lambda: "Inserted 2 missing junction(s).",
        snap_tolerance_mm=0.001,
        **optional,
    )
    return ServiceHarness(service=service, writes=writes, calls=calls)


def _symbol(reference: str = "U1", lib_id: str = "Device:R") -> dict[str, Any]:
    return {
        "reference": reference,
        "lib_id": lib_id,
        "value": "R",
        "x": 10.0,
        "y": 20.0,
        "rotation": 0,
        "unit": 1,
    }


def test_add_pin_labels_reports_invalid_and_missing_connections_without_write(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, parsed={"uuid": "root", "symbols": [], "power_symbols": []})

    result = harness.service.add_pin_labels(
        [
            {"reference": "U1", "pin": "1"},
            {"reference": "U404", "pin": "1", "net": "SIG"},
        ]
    )

    assert result == (
        "No pin labels were added.\n"
        "SKIP {'reference': 'U1', 'pin': '1'}: needs reference, pin, net\n"
        "U404.1: reference not found"
    )
    assert harness.writes == []


def test_add_pin_labels_writes_signal_stub_and_target_detail(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    result = harness.service.add_pin_labels(
        [{"reference": "U1", "pin": "1", "net": "SIG"}],
        stub_mm=5.08,
        global_labels=False,
        sheet="Power",
    )

    assert result == (
        "Reloaded\nTarget schematic: board.kicad_sch\n"
        "Added 1 pin terminal(s) with stubs:\n"
        "U1.1 -> SIG @ (17.08, 20.0)"
    )
    assert len(harness.writes) == 1
    path, content = harness.writes[0]
    assert path == tmp_path / "board.kicad_sch"
    assert "WIRE(12.0,20.0->17.08,20.0)" in content
    assert "LABEL(SIG,17.08,20.0,0,label,None)" in content


def test_add_pin_labels_emits_hierarchical_label_with_shape(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    result = harness.service.add_pin_labels(
        [{"reference": "U1", "pin": "1", "net": "SIG", "shape": "input"}],
        label_kind="hierarchical",
    )

    assert "U1.1 -> SIG @ (17.08, 20.0)" in result
    content = harness.writes[0][1]
    assert "LABEL(SIG,17.08,20.0,0,hierarchical_label,input)" in content
    assert ("label_block", ("SIG", 17.08, 20.0, 0, "hierarchical_label", "input")) in harness.calls


def test_add_pin_labels_label_kind_overrides_global_labels(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    harness.service.add_pin_labels(
        [{"reference": "U1", "pin": "1", "net": "SIG"}],
        global_labels=True,
        label_kind="local",
    )

    assert "LABEL(SIG,17.08,20.0,0,label,None)" in harness.writes[0][1]


def test_add_pin_labels_defaults_to_global_label(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    harness.service.add_pin_labels([{"reference": "U1", "pin": "1", "net": "SIG"}])

    assert "LABEL(SIG,17.08,20.0,0,global_label,None)" in harness.writes[0][1]


def test_add_pin_labels_rejects_invalid_label_kind(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    with pytest.raises(ValueError, match="label_kind must be one of"):
        harness.service.add_pin_labels(
            [{"reference": "U1", "pin": "1", "net": "SIG"}],
            label_kind="sheet",
        )


def test_add_pin_labels_resolves_alias_and_places_power_symbol(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        aliases={("Device", "R"): {"VIN": (12.0, 20.0)}},
    )

    result = harness.service.add_pin_labels([{"reference": "U1", "pin": "VIN", "net": "3V3"}])

    assert "U1.VIN -> 3V3 (power) @ (17.08, 20.0)" in result
    assert any(name == "place_symbol_block" for name, _payload in harness.calls)
    assert "POWER(3V3,17.08,20.0)" in harness.writes[0][1]


def test_add_pin_labels_accepts_power_symbol_origin_pin_fallback(tmp_path: Path) -> None:
    power = _symbol(reference="#PWR01", lib_id="power:GND")
    power["value"] = "GND"
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [], "power_symbols": [power]},
        pin_positions={("power", "GND"): {}},
    )

    result = harness.service.add_pin_labels([{"reference": "#PWR01", "pin": "1", "net": "GND"}])

    assert "#PWR01.1 -> GND (power) @ (15.08, 20.0)" in result
    assert len(harness.writes) == 1


def test_add_pin_labels_skips_missing_power_library_symbol(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_lib=None,
    )

    result = harness.service.add_pin_labels([{"reference": "U1", "pin": "1", "net": "3V3"}])

    assert result == "No pin labels were added.\nU1.1: power symbol '3V3' was not found"
    assert harness.writes == []


def test_add_pin_labels_merges_stacked_power_pins_into_one_terminal(tmp_path: Path) -> None:
    # A USB-C receptacle stacks its four GND pins on a single coordinate; all four
    # connections must share ONE power symbol and stub, not spawn orphans.
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={
            ("Device", "R"): {
                "A1": (12.0, 20.0),
                "A12": (12.0, 20.0),
                "B1": (12.0, 20.0),
                "B12": (12.0, 20.0),
            }
        },
    )

    result = harness.service.add_pin_labels(
        [
            {"reference": "U1", "pin": "A1", "net": "GND"},
            {"reference": "U1", "pin": "A12", "net": "GND"},
            {"reference": "U1", "pin": "B1", "net": "GND"},
            {"reference": "U1", "pin": "B12", "net": "GND"},
        ]
    )

    # Exactly one wire stub and one power symbol are emitted.
    assert sum(name == "wire_block" for name, _ in harness.calls) == 1
    assert sum(name == "place_symbol_block" for name, _ in harness.calls) == 1
    power_calls = [payload for name, payload in harness.calls if name == "place_symbol_block"]
    assert power_calls[0]["reference"] == "#PWR43981"
    assert str(power_calls[0]["reference"])[4:].isdigit()
    content = harness.writes[0][1]
    assert content.count("POWER(GND,17.08,20.0)") == 1
    assert content.count("WIRE(12.0,20.0->17.08,20.0)") == 1
    # The co-connected pins are still reported, never staggered.
    assert "U1.A1 -> GND (power) @ (17.08, 20.0)" in result
    assert "U1.A12 -> GND (stacked on shared terminal @ (17.08, 20.0))" in result
    assert "U1.B1 -> GND (stacked on shared terminal @ (17.08, 20.0))" in result
    assert "U1.B12 -> GND (stacked on shared terminal @ (17.08, 20.0))" in result
    assert "staggered" not in result
    assert "Added 1 pin terminal(s) with stubs" in result


def test_add_pin_labels_merges_stacked_signal_pins_into_one_label(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0), "2": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    result = harness.service.add_pin_labels(
        [
            {"reference": "U1", "pin": "1", "net": "SIG"},
            {"reference": "U1", "pin": "2", "net": "SIG"},
        ]
    )

    assert sum(name == "wire_block" for name, _ in harness.calls) == 1
    assert sum(name == "label_block" for name, _ in harness.calls) == 1
    assert "U1.1 -> SIG @ (17.08, 20.0)" in result
    assert "U1.2 -> SIG (stacked on shared terminal @ (17.08, 20.0))" in result
    assert "staggered" not in result


def test_add_pin_labels_still_staggers_distinct_colliding_coordinates(tmp_path: Path) -> None:
    # Genuinely distinct pin coordinates whose terminal endpoints collide must
    # still be staggered apart -- unchanged behavior.
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0), "2": (13.0, 20.0)}},
        power_net=lambda name: False,
    )

    result = harness.service.add_pin_labels(
        [
            {"reference": "U1", "pin": "1", "net": "A"},
            {"reference": "U1", "pin": "2", "net": "B"},
        ]
    )

    assert sum(name == "wire_block" for name, _ in harness.calls) == 2
    assert sum(name == "label_block" for name, _ in harness.calls) == 2
    assert "U1.1 -> A @ (17.08, 20.0)" in result
    assert "U1.2 -> B @ (23.16, 20.0); staggered 2 step(s)" in result
    assert "stacked on shared terminal" not in result


def test_route_wire_between_pins_reports_missing_reference_and_pin(tmp_path: Path) -> None:
    missing_ref = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1")], "power_symbols": []},
    )
    # Unknown endpoints are refusals, so they must surface as failed tool
    # results with a stable code rather than as success text (issue #1388).
    with pytest.raises(ToolValidationError) as missing_ref_error:
        missing_ref.service.route_wire_between_pins("U1", "1", "R404", "2")
    assert str(missing_ref_error.value) == "Reference 'R404' was not found in the schematic."
    assert missing_ref_error.value.to_payload()["code"] == "TOOL_VALIDATION_ERROR"
    assert missing_ref.writes == []

    missing_pin = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1"), _symbol("R1")], "power_symbols": []},
        pin_positions={("Device", "R"): {}},
    )
    with pytest.raises(ToolValidationError) as missing_pin_error:
        missing_pin.service.route_wire_between_pins("U1", "1", "R1", "2")
    assert str(missing_pin_error.value) == "Pin 1 was not found on U1."
    assert missing_pin.writes == []


def test_route_wire_between_pins_refuses_empty_route_between_distinct_points(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1"), _symbol("R1")], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (1.0, 1.0), "2": (5.0, 5.0)}},
        route_segments=[],
    )

    with pytest.raises(SchematicRoutingUnsafeError) as error:
        harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert "produced no wire segments" in str(error.value)
    assert harness.writes == []


def test_route_wire_between_pins_reports_overlap_without_write(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1"), _symbol("R1")], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (1.0, 1.0), "2": (1.0, 1.0)}},
        route_segments=[],
    )

    result = harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert result == "U1:1 and R1:2 already overlap."
    assert harness.writes == []


@pytest.mark.parametrize(
    "warning",
    [
        "WARNING: obstacle_bypass_failed",
        "WARNING: net_isolation_failed",
        "WARNING: unexpected_route_failure",
    ],
)
def test_route_wire_between_pins_refuses_unsafe_fallback_without_write(
    tmp_path: Path, warning: str
) -> None:
    harness = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1"), _symbol("R1")], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (1.0, 1.0), "2": (5.0, 5.0)}},
        route_segments=[(1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0)],
        route_warning=warning,
    )

    original = (tmp_path / "board.kicad_sch").read_bytes()
    with pytest.raises(SchematicRoutingUnsafeError) as error:
        harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert str(error.value) == (
        f"Route refused between U1:1 and R1:2: {warning}. No schematic changes were made."
    )
    assert error.value.to_payload()["code"] == "SCHEMATIC_ROUTING_UNSAFE"
    assert error.value.to_payload()["retryable"] is False
    assert harness.writes == []
    assert (tmp_path / "board.kicad_sch").read_bytes() == original


def test_route_wire_between_pins_allows_reviewed_crossing_warning(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"symbols": [_symbol("U1"), _symbol("R1")], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (1.0, 1.0), "2": (5.0, 5.0)}},
        route_segments=[(1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0)],
        route_warning="WARNING: net_crossing_unavoidable",
    )

    result = harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert "Routed 2 wire segment(s)" in result
    assert "WARNING: net_crossing_unavoidable" in result
    assert len(harness.writes) == 1
    assert "WIRE(1.0,1.0->5.0,1.0)" in harness.writes[0][1]


def test_add_missing_junctions_runs_fixer_before_reload(tmp_path: Path) -> None:
    order: list[str] = []
    harness = _harness(tmp_path)
    service = harness.service

    def fix() -> str:
        order.append("fix")
        return "Fixed"

    def reload() -> str:
        order.append("reload")
        return "Reloaded"

    object.__setattr__(service, "run_auto_add_missing_junctions", fix)
    object.__setattr__(service, "reload_schematic", reload)

    result = service.add_missing_junctions()

    assert order == ["fix", "reload"]
    assert result == "Reloaded\nFixed"


def _unit_symbol(reference: str, lib_id: str, unit: int, x: float) -> dict[str, Any]:
    return {
        "reference": reference,
        "lib_id": lib_id,
        "value": "PESD5V0L4UG",
        "x": x,
        "y": 20.0,
        "rotation": 0,
        "unit": unit,
    }


def test_add_pin_labels_resolves_pin_on_non_last_multiunit_block(tmp_path: Path) -> None:
    # Four units share reference D810. Pin 3 lives on unit 2 (a non-last block),
    # so last-unit-wins resolution used to report "pin not found".
    symbols = [
        _unit_symbol("D810", "Power_Protection:PESD5V0L4UG", unit, x)
        for unit, x in ((1, 10.0), (2, 30.0), (3, 50.0), (4, 70.0))
    ]
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": symbols, "power_symbols": []},
        pin_positions={
            ("Power_Protection", "PESD5V0L4UG"): {"3": (32.0, 20.0)},
        },
        power_net=lambda name: False,
    )

    # Only unit 2 exposes pin "3"; wire the getter so every non-matching unit
    # returns nothing, mirroring real per-unit pin geometry.
    original = harness.service.get_pin_positions

    def per_unit(
        library: str, name: str, x: float, y: float, rotation: int, unit: int
    ) -> dict[str, tuple[float, float]]:
        if (library, name) == ("Power_Protection", "PESD5V0L4UG") and unit != 2:
            return {}
        return original(library, name, x, y, rotation, unit)

    object.__setattr__(harness.service, "get_pin_positions", per_unit)

    result = harness.service.add_pin_labels(
        [{"reference": "D810", "pin": "3", "net": "SIG"}],
    )

    assert "pin not found" not in result
    assert "D810.3 -> SIG @ (37.08, 20.0)" in result
    assert len(harness.writes) == 1


def test_add_pin_labels_explicit_unit_disambiguates_shared_pin(tmp_path: Path) -> None:
    # Both units expose pin "1"; an explicit unit selects unit 2's geometry.
    symbols = [
        _unit_symbol("D810", "Power_Protection:PESD5V0L4UG", 1, 10.0),
        _unit_symbol("D810", "Power_Protection:PESD5V0L4UG", 2, 30.0),
    ]
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": symbols, "power_symbols": []},
        pin_positions={("Power_Protection", "PESD5V0L4UG"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    original = harness.service.get_pin_positions

    def per_unit(
        library: str, name: str, x: float, y: float, rotation: int, unit: int
    ) -> dict[str, tuple[float, float]]:
        # Position tracks the block origin x so we can tell the units apart.
        return (
            {"1": (x + 2.0, y)}
            if (library, name)
            == (
                "Power_Protection",
                "PESD5V0L4UG",
            )
            else original(library, name, x, y, rotation, unit)
        )

    object.__setattr__(harness.service, "get_pin_positions", per_unit)

    result = harness.service.add_pin_labels(
        [{"reference": "D810", "pin": "1", "net": "SIG", "unit": 2}],
    )

    # Unit 2 origin x is 30 -> pin at 32 -> stub end at 37.08.
    assert "D810.1 -> SIG @ (37.08, 20.0)" in result


def test_add_pin_labels_rejects_non_integer_unit(tmp_path: Path) -> None:
    symbols = [
        _unit_symbol("D810", "Power_Protection:PESD5V0L4UG", 1, 10.0),
        _unit_symbol("D810", "Power_Protection:PESD5V0L4UG", 2, 30.0),
    ]
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": symbols, "power_symbols": []},
        pin_positions={("Power_Protection", "PESD5V0L4UG"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    with pytest.raises(ValueError, match="unit must be an integer"):
        harness.service.add_pin_labels(
            [{"reference": "D810", "pin": "1", "net": "SIG", "unit": "2a"}],
        )

    assert harness.writes == []


def test_add_pin_labels_single_unit_regression(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    result = harness.service.add_pin_labels([{"reference": "U1", "pin": "1", "net": "SIG"}])

    assert "U1.1 -> SIG @ (17.08, 20.0)" in result
    assert "LABEL(SIG,17.08,20.0,0,global_label,None)" in harness.writes[0][1]


@pytest.mark.parametrize("shape", ["not-a-shape", 123])
def test_add_pin_labels_rejects_invalid_shape_values(tmp_path: Path, shape: object) -> None:
    harness = _harness(
        tmp_path,
        parsed={"uuid": "root", "symbols": [_symbol()], "power_symbols": []},
        pin_positions={("Device", "R"): {"1": (12.0, 20.0)}},
        power_net=lambda name: False,
    )

    with pytest.raises(ValueError, match="shape must be one of"):
        harness.service.add_pin_labels(
            [{"reference": "U1", "pin": "1", "net": "SIG", "shape": shape}],
            label_kind="hierarchical",
        )

    assert harness.writes == []


def test_resolve_pin_point_and_candidate_branches(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        parsed={
            "uuid": "root",
            "symbols": [
                {"reference": "U1", "lib_id": "malformed_no_colon", "x": 0.0, "y": 0.0, "unit": 1}
            ],
            "power_symbols": [],
        },
        pin_positions={},
        power_net=lambda name: False,
    )
    res_lib = harness.service.add_pin_labels([{"reference": "U1", "pin": "1", "net": "SIG"}])
    assert "symbol type not supported: unresolved lib_id 'malformed_no_colon'" in res_lib

    res_unit = harness.service.add_pin_labels(
        [{"reference": "U1", "pin": "1", "net": "SIG", "unit": 99}]
    )
    assert "unit 99 not found" in res_unit

    pt, _pp, err = harness.service._resolve_pin_point(
        point=None,
        pin_positions={},
        sym={"lib_id": "power:GND"},
        pin="2",
        ref="#PWR1",
        ox=0.0,
        oy=0.0,
        is_power_symbol_ref=True,
        unresolved_lib_id=None,
    )
    assert pt is None
    assert err == "#PWR1.2: symbol type not supported for pin '2'"

    pt, pp, err = harness.service._resolve_pin_point(
        point=None,
        pin_positions={},
        sym={"lib_id": "power:GND", "value": "GND"},
        pin="1",
        ref="GND1",
        ox=10.0,
        oy=20.0,
        is_power_symbol_ref=False,
        unresolved_lib_id=None,
    )
    assert pt == (10.0, 20.0)
    assert pp == {"1": (10.0, 20.0)}
    assert err is None

    pt, _pp, err = harness.service._resolve_pin_point(
        point=None,
        pin_positions={},
        sym={"lib_id": "power:CUSTOM", "value": "CUSTOM"},
        pin="2",
        ref="PWR1",
        ox=0.0,
        oy=0.0,
        is_power_symbol_ref=False,
        unresolved_lib_id=None,
    )
    assert pt is None
    assert "has no resolvable pin geometry" in str(err)

    pt, _pp, err = harness.service._resolve_pin_point(
        point=None,
        pin_positions={},
        sym={"lib_id": "Device:R", "value": "R"},
        pin="99",
        ref="R1",
        ox=0.0,
        oy=0.0,
        is_power_symbol_ref=False,
        unresolved_lib_id=None,
    )
    assert pt is None
    assert err == "R1.99: pin not found"


# ---------------------------------------------------------------------------
# Post-route connectivity guard (issue #1388)
# ---------------------------------------------------------------------------

_ROUTE_START = (1.0, 1.0)
_ROUTE_END = (5.0, 5.0)
_ROUTE_SEGMENTS = [(1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0)]


def _group(
    pins: list[tuple[str, str]],
    points: list[tuple[float, float]],
    names: list[str] | None = None,
    *,
    no_connect: bool = False,
) -> dict[str, Any]:
    return {
        "names": sorted(names or []),
        "points": sorted(points),
        "pins": [{"reference": ref, "pin": pin} for ref, pin in pins],
        "no_connect": no_connect,
    }


def _pre_post(
    pre: list[dict[str, Any]], post: list[dict[str, Any]], seen: list[str] | None = None
) -> Callable[[str], list[dict[str, Any]]]:
    """Return a fake graph builder: ``post`` once the route's wires are in the text."""

    def groups(text: str) -> list[dict[str, Any]]:
        if seen is not None:
            seen.append(text)
        return post if "WIRE(" in text else pre

    return groups


def _guarded_harness(
    tmp_path: Path,
    groups: Callable[[str], list[dict[str, Any]]],
    *,
    route_warning: str | None = None,
    power_symbols: list[dict[str, Any]] | None = None,
) -> ServiceHarness:
    symbols = [_symbol("R1")]
    if power_symbols is None:
        symbols.insert(0, _symbol("U1"))
    return _harness(
        tmp_path,
        parsed={"symbols": symbols, "power_symbols": power_symbols or []},
        pin_positions={
            ("Device", "R"): {"1": _ROUTE_START, "2": _ROUTE_END},
            ("power", "GND"): {"1": _ROUTE_START},
        },
        route_segments=list(_ROUTE_SEGMENTS),
        route_warning=route_warning,
        connectivity_groups=groups,
    )


def _assert_refused_without_write(
    harness: ServiceHarness, tmp_path: Path, ref1: str = "U1", pin1: str = "1"
) -> str:
    original = (tmp_path / "board.kicad_sch").read_bytes()
    with pytest.raises(SchematicRoutingUnsafeError) as error:
        harness.service.route_wire_between_pins(ref1, pin1, "R1", "2")
    message = str(error.value)
    assert message.startswith(f"Route refused between {ref1}:{pin1} and R1:2: ")
    assert message.endswith("No schematic changes were made.")
    assert error.value.to_payload()["code"] == "SCHEMATIC_ROUTING_UNSAFE"
    assert harness.writes == []
    assert (tmp_path / "board.kicad_sch").read_bytes() == original
    return message


def test_route_refuses_when_post_route_merges_a_third_net(tmp_path: Path) -> None:
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [_ROUTE_END])
    third = _group([("J1", "A7"), ("U2", "3")], [(9.0, 9.0)], ["USB_DM"])
    merged = _group(
        [("U1", "1"), ("R1", "2"), ("J1", "A7"), ("U2", "3")],
        [_ROUTE_START, _ROUTE_END, (5.0, 1.0), (9.0, 9.0)],
        ["USB_DM"],
    )
    harness = _guarded_harness(tmp_path, _pre_post([start, end, third], [merged]))

    message = _assert_refused_without_write(harness, tmp_path)

    assert "would merge another net" in message
    assert "USB_DM" in message
    assert "J1:A7" in message
    assert "U2:3" in message


def test_route_refuses_to_merge_two_distinct_named_nets(tmp_path: Path) -> None:
    start = _group([("U1", "1"), ("J1", "A6")], [_ROUTE_START, (0.0, 1.0)], ["USB_DP"])
    end = _group([("R1", "2"), ("U2", "3")], [_ROUTE_END, (6.0, 5.0)], ["USB_DM"])
    merged = _group(
        [("U1", "1"), ("J1", "A6"), ("R1", "2"), ("U2", "3")],
        [_ROUTE_START, (0.0, 1.0), _ROUTE_END, (6.0, 5.0), (5.0, 1.0)],
        ["USB_DM", "USB_DP"],
    )
    harness = _guarded_harness(tmp_path, _pre_post([start, end], [merged]))

    message = _assert_refused_without_write(harness, tmp_path)

    assert "distinct named nets" in message
    assert "USB_DP" in message
    assert "USB_DM" in message


def test_route_refuses_to_merge_two_power_nets(tmp_path: Path) -> None:
    gnd = _group([], [_ROUTE_START], ["GND"])
    vcc = _group([("R1", "2")], [_ROUTE_END], ["VCC"])
    merged = _group([("R1", "2")], [_ROUTE_START, _ROUTE_END, (5.0, 1.0)], ["GND", "VCC"])
    power_symbol = {
        "reference": "#PWR01",
        "lib_id": "power:GND",
        "value": "GND",
        "x": 1.0,
        "y": 1.0,
        "rotation": 0,
        "unit": 1,
    }
    harness = _guarded_harness(
        tmp_path, _pre_post([gnd, vcc], [merged]), power_symbols=[power_symbol]
    )

    message = _assert_refused_without_write(harness, tmp_path, ref1="#PWR01")

    assert "'GND'" in message
    assert "'VCC'" in message


def test_route_refuses_when_route_does_not_join_requested_endpoints(tmp_path: Path) -> None:
    # e.g. a mirrored symbol whose real pin is not where the route ended.
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [(7.0, 5.0)])
    post = [_group([("U1", "1")], [_ROUTE_START, (5.0, 1.0), _ROUTE_END]), end]
    harness = _guarded_harness(tmp_path, _pre_post([start, end], post))

    message = _assert_refused_without_write(harness, tmp_path)

    assert "does not connect U1:1 to R1:2" in message


def test_route_allows_extending_partially_drawn_named_net(tmp_path: Path) -> None:
    start = _group([("U1", "1"), ("J1", "A6")], [_ROUTE_START, (0.0, 1.0)], ["USB_DP"])
    end = _group([("R1", "2")], [_ROUTE_END])
    merged = _group(
        [("U1", "1"), ("J1", "A6"), ("R1", "2")],
        [_ROUTE_START, (0.0, 1.0), _ROUTE_END, (5.0, 1.0)],
        ["USB_DP"],
    )
    harness = _guarded_harness(tmp_path, _pre_post([start, end], [merged]))

    result = harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert "Routed 2 wire segment(s)" in result
    assert len(harness.writes) == 1


def test_route_allows_endpoints_already_on_same_net(tmp_path: Path) -> None:
    same = _group([("U1", "1"), ("R1", "2")], [_ROUTE_START, _ROUTE_END], ["USB_DP"])
    post = _group([("U1", "1"), ("R1", "2")], [_ROUTE_START, _ROUTE_END, (5.0, 1.0)], ["USB_DP"])
    harness = _guarded_harness(tmp_path, _pre_post([same], [post]))

    harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert len(harness.writes) == 1


def test_route_allows_absorbing_dangling_unnamed_wire(tmp_path: Path) -> None:
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [_ROUTE_END])
    dangling = _group([], [(5.0, 2.0), (5.0, 3.0)])
    merged = _group(
        [("U1", "1"), ("R1", "2")],
        [_ROUTE_START, _ROUTE_END, (5.0, 1.0), (5.0, 2.0), (5.0, 3.0)],
    )
    harness = _guarded_harness(tmp_path, _pre_post([start, end, dangling], [merged]))

    harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert len(harness.writes) == 1


def test_route_refuses_absorbing_no_connect_marker(tmp_path: Path) -> None:
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [_ROUTE_END])
    marker = _group([], [(5.0, 2.0)], no_connect=True)
    merged = _group(
        [("U1", "1"), ("R1", "2")],
        [_ROUTE_START, _ROUTE_END, (5.0, 1.0), (5.0, 2.0)],
        no_connect=True,
    )
    harness = _guarded_harness(tmp_path, _pre_post([start, end, marker], [merged]))

    message = _assert_refused_without_write(harness, tmp_path)

    assert "no-connect" in message


def test_route_guard_still_allows_reviewed_crossing_warning(tmp_path: Path) -> None:
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [_ROUTE_END])
    crossed = _group([("J1", "A7")], [(3.0, 0.0), (3.0, 2.0)], ["USB_DM"])
    merged = _group([("U1", "1"), ("R1", "2")], [_ROUTE_START, _ROUTE_END, (5.0, 1.0)])
    harness = _guarded_harness(
        tmp_path,
        _pre_post([start, end, crossed], [merged, crossed]),
        route_warning="WARNING: net_crossing_unavoidable",
    )

    result = harness.service.route_wire_between_pins("U1", "1", "R1", "2")

    assert "WARNING: net_crossing_unavoidable" in result
    assert len(harness.writes) == 1


def test_merge_guard_accepts_route_to_common_pin_of_second_unit() -> None:
    # A unit-0 pin is repeated on every placed unit at a different position;
    # the route lands on the copy at its end point, not the first one listed.
    r1 = _group([("R1", "2")], [(101.6, 25.4)])
    unit_a = _group([("U3", "8")], [(50.8, 43.18)])
    unit_b = _group([("U3", "8")], [(101.6, 43.18)])
    merged = _group([("R1", "2"), ("U3", "8")], [(101.6, 25.4), (101.6, 43.18)])

    reason = route_merge_violation(
        [r1, unit_a, unit_b],
        [unit_a, merged],
        ("R1", "2", (101.6, 25.4)),
        ("U3", "8", (101.6, 43.18)),
    )

    assert reason is None


def test_merge_guard_refuses_when_route_misses_every_copy_of_a_repeated_pin() -> None:
    r1 = _group([("R1", "2")], [(101.6, 25.4)])
    unit_a = _group([("U3", "8")], [(50.8, 43.18)])
    unit_b = _group([("U3", "8")], [(101.6, 43.18)])
    stray = _group([("R1", "2")], [(101.6, 25.4), (90.0, 43.18)])

    reason = route_merge_violation(
        [r1, unit_a, unit_b],
        [unit_a, unit_b, stray],
        ("R1", "2", (101.6, 25.4)),
        ("U3", "8", (90.0, 43.18)),
    )

    assert reason is not None
    assert "does not connect R1:2 to U3:8" in reason


def test_merge_guard_refuses_disjoint_segment_bridging_two_foreign_nets() -> None:
    # A faulty router emits a stray segment (5,20)->(10,20) away from the path.
    r1 = _group([("R1", "1")], [(0.0, 0.0)])
    r2 = _group([("R2", "1")], [(10.0, 0.0)])
    dp = _group([("U1", "33")], [(0.0, 20.0), (5.0, 20.0)], ["USB_DP"])
    dm = _group([("U1", "32")], [(10.0, 20.0), (15.0, 20.0)], ["USB_DM"])
    joined = _group([("R1", "1"), ("R2", "1")], [(0.0, 0.0), (10.0, 0.0)])
    shorted = _group(
        [("U1", "33"), ("U1", "32")],
        [(0.0, 20.0), (5.0, 20.0), (10.0, 20.0), (15.0, 20.0)],
        ["USB_DM", "USB_DP"],
    )

    reason = route_merge_violation(
        [r1, r2, dp, dm],
        [joined, shorted],
        ("R1", "1", (0.0, 0.0)),
        ("R2", "1", (10.0, 0.0)),
    )

    assert reason is not None
    assert "would merge other nets" in reason
    assert "USB_DP" in reason
    assert "USB_DM" in reason


def test_merge_guard_allows_disjoint_segment_touching_one_foreign_net() -> None:
    # Extending a foreign net with a dangling stub joins nothing to anything.
    r1 = _group([("R1", "1")], [(0.0, 0.0)])
    r2 = _group([("R2", "1")], [(10.0, 0.0)])
    dp = _group([("U1", "33")], [(0.0, 20.0), (5.0, 20.0)], ["USB_DP"])
    joined = _group([("R1", "1"), ("R2", "1")], [(0.0, 0.0), (10.0, 0.0)])
    extended = _group([("U1", "33")], [(0.0, 20.0), (5.0, 20.0), (8.0, 20.0)], ["USB_DP"])

    reason = route_merge_violation(
        [r1, r2, dp],
        [joined, extended],
        ("R1", "1", (0.0, 0.0)),
        ("R2", "1", (10.0, 0.0)),
    )

    assert reason is None


def test_route_guard_fails_closed_when_connectivity_cannot_be_built(tmp_path: Path) -> None:
    def broken(_text: str) -> list[dict[str, Any]]:
        raise ValueError("unparseable")

    harness = _guarded_harness(tmp_path, broken)

    message = _assert_refused_without_write(harness, tmp_path)

    assert "connectivity check failed" in message


def test_route_guard_judges_the_text_seen_inside_the_write(tmp_path: Path) -> None:
    start = _group([("U1", "1")], [_ROUTE_START])
    end = _group([("R1", "2")], [_ROUTE_END])
    merged = _group([("U1", "1"), ("R1", "2")], [_ROUTE_START, _ROUTE_END, (5.0, 1.0)])
    seen: list[str] = []
    harness = _guarded_harness(tmp_path, _pre_post([start, end], [merged], seen))
    in_write: list[bool] = []
    inner = harness.service.transactional_write

    def tracking_write(mutator: Callable[[str], str], path: Path | None) -> str:
        in_write.append(True)
        # Simulate a concurrent edit that lands before the write lock is taken.
        (tmp_path / "board.kicad_sch").write_text(
            "(kicad_sch\n\t(junction concurrent)\n\t(sheet_instances)\n)", encoding="utf-8"
        )
        return inner(mutator, path)

    guarded = replace(harness.service, transactional_write=tracking_write)
    guarded.route_wire_between_pins("U1", "1", "R1", "2")

    assert in_write == [True]
    assert len(seen) == 2
    assert all("junction concurrent" in text for text in seen)
    assert "WIRE(" not in seen[0]
    assert seen[1] == harness.writes[0][1]
