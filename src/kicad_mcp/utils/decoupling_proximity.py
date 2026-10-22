"""Pad-to-pad decoupling proximity measurement (pure domain, KiCad-free).

The placement gate used to judge decoupling by the distance between the IC and
capacitor footprint origins. That centroid proxy is not a physical measurement:
large packages and origin-offset footprints can put a capacitor's supply pad
millimetres from the IC supply pin while the origins stay far apart.

This module measures what the rule means instead: for each IC supply pin, the
distance from its pad centre to the nearest capacitor pad on the *same* net. The
result is pad-centre Euclidean proximity only. It says nothing about copper
path length, loop area or EMC performance, so callers must not present a pass as
a decoupling or EMC clearance.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .schematic_rules import is_ground_name, is_supply_rail_name

PadView = Mapping[str, Any]

# KiCad pin electrical types that are never a supply input.
_SIGNAL_PINTYPES = frozenset(
    {
        "input",
        "output",
        "bidirectional",
        "tri_state",
        "open_collector",
        "open_emitter",
        "no_connect",
        "free",
    }
)

# Deliberately narrower than the schematic protection heuristics: generic
# ``SP[0-9]`` / ``SS[0-9]`` / ``FUSE`` patterns would also catch transceivers
# such as SP3485 or ordinary diodes. Only dedicated ESD/TVS parts match here.
_ESD_VALUE_RE = re.compile(
    r"(?:USBLC|TPD\d|PESD|ESDA|ESD\d|PRTR|SRV05|RCLAMP|TVS|SP05\d\d|SMAJ|SMBJ|SMCJ|"
    r"SMF\d|SM712|NUP\d|IP42\d\d)",
    re.IGNORECASE,
)
_ESD_REF_RE = re.compile(r"^(?:ESD|TVS)\d", re.IGNORECASE)
_ESD_FOOTPRINT_RE = re.compile(r"(?:^|[^A-Z])(?:ESD|TVS)(?:[^A-Z]|$)", re.IGNORECASE)


def is_esd_protection_device(reference: str, value: str, footprint_name: str) -> bool:
    """Return whether a part is a dedicated ESD/TVS protector, not a decoupled IC."""
    if _ESD_REF_RE.match(reference.strip()):
        return True
    if _ESD_VALUE_RE.search(value):
        return True
    return bool(_ESD_FOOTPRINT_RE.search(footprint_name))


def normalize_net_name(net: str) -> str:
    """Strip a hierarchical sheet path (``/power/+3V3`` -> ``+3V3``)."""
    stripped = net.strip()
    if stripped.startswith("/"):
        return stripped.rsplit("/", 1)[-1]
    return stripped


def is_supply_pin(pad: PadView) -> bool:
    """Return whether an IC pad is a positive supply input that needs local decoupling."""
    net = str(pad.get("net", "") or "").strip()
    if not net or net.lower().startswith("unconnected-"):
        return False
    name = normalize_net_name(net)
    if is_ground_name(name):
        return False
    pintype = str(pad.get("pintype", "") or "").strip().lower()
    if pintype.startswith("power_in"):
        return True
    if pintype in _SIGNAL_PINTYPES:
        return False
    return is_supply_rail_name(name)


@dataclass(frozen=True)
class PinDistance:
    """Nearest same-net capacitor pad for one IC supply pin."""

    pin: str
    net: str
    cap_ref: str
    cap_pad: str
    distance_mm: float


@dataclass(frozen=True)
class DecouplingMeasurement:
    """Pad-centre decoupling proximity for one IC.

    ``verdict`` is ``"heuristic"`` only when pad data cannot support a
    measurement: the IC has no supply pins with a net and a position, or no
    capacitor pad on the board carries net data. Callers should then fall back
    to a clearly labelled, non-failing heuristic.

    ``"fail"`` is a measured violation: a supply pin whose nearest same-net
    capacitor pad is beyond the limit (``failing_pins``), or a supply net with
    no same-net capacitor pad anywhere on the board (``uncapped_nets``).
    """

    verdict: Literal["pass", "fail", "heuristic"]
    limit_mm: float
    per_pin: tuple[PinDistance, ...] = ()
    failing_pins: tuple[PinDistance, ...] = ()
    uncapped_nets: tuple[str, ...] = ()
    uncapped_pins: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _coords(pad: PadView) -> tuple[float, float] | None:
    x_mm, y_mm = pad.get("x_mm"), pad.get("y_mm")
    if x_mm is None or y_mm is None:
        return None
    return float(x_mm), float(y_mm)


def _cap_pads_on_net(
    cap_pads_by_ref: Mapping[str, Sequence[PadView]],
    net: str,
) -> list[tuple[str, str, tuple[float, float]]]:
    found: list[tuple[str, str, tuple[float, float]]] = []
    for cap_ref, pads in sorted(cap_pads_by_ref.items()):
        for pad in pads:
            point = _coords(pad)
            if point is not None and str(pad.get("net", "") or "").strip() == net:
                found.append((cap_ref, str(pad.get("number", "")), point))
    return found


def _nearest(
    pin: str,
    net: str,
    point: tuple[float, float],
    candidates: list[tuple[str, str, tuple[float, float]]],
) -> PinDistance:
    cap_ref, cap_pad, cap_point = min(
        candidates,
        key=lambda item: math.hypot(item[2][0] - point[0], item[2][1] - point[1]),
    )
    distance = math.hypot(cap_point[0] - point[0], cap_point[1] - point[1])
    return PinDistance(pin, net, cap_ref, cap_pad, round(distance, 4))


def _has_cap_net_data(cap_pads_by_ref: Mapping[str, Sequence[PadView]]) -> bool:
    return any(
        _coords(pad) is not None and str(pad.get("net", "") or "").strip()
        for pads in cap_pads_by_ref.values()
        for pad in pads
    )


def measure_decoupling(
    ic_pads: Sequence[PadView],
    cap_pads_by_ref: Mapping[str, Sequence[PadView]],
    limit_mm: float,
) -> DecouplingMeasurement:
    """Measure each IC supply pin against the nearest same-net capacitor pad.

    Every supply pin is judged on its own: a pin fails when its nearest
    same-net capacitor pad is farther than ``limit_mm``, even if another pin on
    the same net is decoupled. A supply net with no same-net capacitor pad at
    all fails as ``uncapped_nets`` whenever capacitor net data is available.
    """
    pins_by_net: dict[str, list[tuple[str, tuple[float, float]]]] = {}
    for pad in ic_pads:
        point = _coords(pad)
        if point is None or not is_supply_pin(pad):
            continue
        net = str(pad.get("net", "")).strip()
        pins_by_net.setdefault(net, []).append((str(pad.get("number", "")), point))

    if not pins_by_net or not _has_cap_net_data(cap_pads_by_ref):
        return DecouplingMeasurement(verdict="heuristic", limit_mm=limit_mm)

    per_pin: list[PinDistance] = []
    failing: list[PinDistance] = []
    uncapped: list[str] = []
    uncapped_pins: list[tuple[str, tuple[str, ...]]] = []
    for net in sorted(pins_by_net):
        candidates = _cap_pads_on_net(cap_pads_by_ref, net)
        if not candidates:
            uncapped.append(net)
            uncapped_pins.append((net, tuple(pin for pin, _ in pins_by_net[net])))
            continue
        measured = [_nearest(pin, net, point, candidates) for pin, point in pins_by_net[net]]
        per_pin.extend(measured)
        failing.extend(item for item in measured if item.distance_mm > limit_mm)

    return DecouplingMeasurement(
        verdict="fail" if failing or uncapped else "pass",
        limit_mm=limit_mm,
        per_pin=tuple(per_pin),
        failing_pins=tuple(failing),
        uncapped_nets=tuple(uncapped),
        uncapped_pins=tuple(uncapped_pins),
    )
