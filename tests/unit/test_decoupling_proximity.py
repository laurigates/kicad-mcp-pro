"""Pure-domain pad-to-pad decoupling measurement (issue #1380)."""

from __future__ import annotations

import pytest

from kicad_mcp.utils.decoupling_proximity import (
    is_esd_protection_device,
    is_supply_pin,
    measure_decoupling,
)


def _pad(number: str, net: str, x: float, y: float, pintype: str = "") -> dict[str, object]:
    return {"number": number, "net": net, "pintype": pintype, "x_mm": x, "y_mm": y}


def test_centroid_far_but_pad_near_cap_passes() -> None:
    # IC origin (0, 0) and cap origin (9, 0) are 9 mm apart, but the VDD pad at
    # (4, 0) is 2.5 mm from the cap's +3V3 pad at (6.5, 0).
    ic_pads = [_pad("1", "+3V3", 4.0, 0.0, "power_in"), _pad("2", "GND", -4.0, 0.0)]
    caps = {"C1": [_pad("1", "+3V3", 6.5, 0.0), _pad("2", "GND", 11.5, 0.0)]}

    result = measure_decoupling(ic_pads, caps, limit_mm=6.0)

    assert result.verdict == "pass"
    assert len(result.per_pin) == 1
    pin = result.per_pin[0]
    assert (pin.pin, pin.net, pin.cap_ref, pin.cap_pad) == ("1", "+3V3", "C1", "1")
    assert pin.distance_mm == pytest.approx(2.5)


def test_pad_far_same_net_cap_fails_with_per_pin_distances() -> None:
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in"), _pad("7", "+3V3", 0.0, 2.0, "power_in")]
    caps = {"C1": [_pad("1", "+3V3", 20.0, 0.0), _pad("2", "GND", 21.0, 0.0)]}

    result = measure_decoupling(ic_pads, caps, limit_mm=6.0)

    assert result.verdict == "fail"
    assert [p.pin for p in result.per_pin] == ["1", "7"]
    assert [p.pin for p in result.failing_pins] == ["1", "7"]
    assert result.failing_pins[0].net == "+3V3"
    assert result.failing_pins[0].distance_mm == pytest.approx(20.0)


def test_cap_on_different_net_is_ignored() -> None:
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in")]
    near_wrong_net = {"C1": [_pad("1", "+5V", 0.5, 0.0), _pad("2", "GND", 1.5, 0.0)]}
    # Capacitor net data exists, so a supply net without any same-net cap is a
    # measured violation, not a "pad data unavailable" heuristic.
    only_wrong_net = measure_decoupling(ic_pads, near_wrong_net, limit_mm=6.0)
    assert only_wrong_net.verdict == "fail"
    assert only_wrong_net.uncapped_nets == ("+3V3",)
    assert only_wrong_net.per_pin == ()

    with_far_right_net = {
        **near_wrong_net,
        "C2": [_pad("1", "+3V3", 30.0, 0.0), _pad("2", "GND", 31.0, 0.0)],
    }
    assert measure_decoupling(ic_pads, with_far_right_net, limit_mm=6.0).verdict == "fail"


def test_ground_pads_are_never_supply_pins() -> None:
    assert not is_supply_pin(_pad("2", "GND", 0, 0))
    assert not is_supply_pin(_pad("2", "GND", 0, 0, "power_in"))
    assert not is_supply_pin(_pad("2", "/sub/VSS", 0, 0))
    ic_pads = [_pad("2", "GND", 0.0, 0.0, "power_in")]
    caps = {"C1": [_pad("2", "GND", 30.0, 0.0)]}
    assert measure_decoupling(ic_pads, caps, limit_mm=6.0).verdict == "heuristic"


def test_supply_pin_detection_rules() -> None:
    # pintype power_in marks a supply pin even on an unnamed net.
    assert is_supply_pin(_pad("3", "Net-(U1-VCAP)", 0, 0, "power_in"))
    # Without pintype the net name decides; hierarchical prefixes are normalised.
    assert is_supply_pin(_pad("3", "/+3V3", 0, 0))
    assert is_supply_pin(_pad("3", "/power/VDD_MCU", 0, 0))
    assert not is_supply_pin(_pad("3", "/USB_DP", 0, 0))
    # Signal pin types are never supply pins, even on supply-like names.
    assert not is_supply_pin(_pad("3", "VBUS_DET", 0, 0, "input"))
    assert not is_supply_pin(_pad("3", "+3V3", 0, 0, "no_connect"))
    assert not is_supply_pin(_pad("3", "unconnected-(U1-VDD-Pad3)", 0, 0))
    assert not is_supply_pin(_pad("3", "", 0, 0, "power_in"))


def test_missing_pad_data_is_heuristic() -> None:
    assert measure_decoupling([], {"C1": []}, limit_mm=6.0).verdict == "heuristic"
    # Supply pins exist but no capacitor pad carries net data: nothing measurable.
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in")]
    assert measure_decoupling(ic_pads, {"C1": []}, limit_mm=6.0).verdict == "heuristic"
    unnetted = {"C1": [_pad("1", "", 0.5, 0.0), _pad("2", "", 1.5, 0.0)]}
    assert measure_decoupling(ic_pads, unnetted, limit_mm=6.0).verdict == "heuristic"


def test_far_pin_fails_even_when_another_pin_on_the_net_is_decoupled() -> None:
    # A near cap at one VDD pin must not mask a VDD pin on the far side of the IC.
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in"), _pad("24", "+3V3", 0.0, 12.0, "power_in")]
    caps = {"C1": [_pad("1", "+3V3", 1.0, 0.0), _pad("2", "GND", 2.0, 0.0)]}

    result = measure_decoupling(ic_pads, caps, limit_mm=6.0)

    assert result.verdict == "fail"
    assert [p.pin for p in result.per_pin] == ["1", "24"]
    assert [p.pin for p in result.failing_pins] == ["24"]


def test_supply_net_without_any_cap_fails_as_uncapped() -> None:
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in"), _pad("2", "VBAT", 0.0, 1.0, "power_in")]
    caps = {"C1": [_pad("1", "+3V3", 1.0, 0.0), _pad("2", "GND", 2.0, 0.0)]}

    result = measure_decoupling(ic_pads, caps, limit_mm=6.0)

    assert result.verdict == "fail"
    assert result.failing_pins == ()
    assert result.uncapped_nets == ("VBAT",)
    assert result.uncapped_pins == (("VBAT", ("2",)),)


def test_all_pins_within_limit_pass() -> None:
    ic_pads = [_pad("1", "+3V3", 0.0, 0.0, "power_in"), _pad("24", "+3V3", 0.0, 12.0, "power_in")]
    caps = {
        "C1": [_pad("1", "+3V3", 1.0, 0.0), _pad("2", "GND", 2.0, 0.0)],
        "C2": [_pad("1", "+3V3", 1.0, 12.0), _pad("2", "GND", 2.0, 12.0)],
    }

    result = measure_decoupling(ic_pads, caps, limit_mm=6.0)

    assert result.verdict == "pass"
    assert result.failing_pins == ()
    assert result.uncapped_nets == ()


@pytest.mark.parametrize(
    ("reference", "value", "footprint"),
    [
        ("U4", "USBLC6-2SC6", "Package_TO_SOT_SMD:SOT-23-6"),
        ("U5", "TPD4E05U06", "Package_SON:USON-10"),
        ("D3", "PESD5V0S1BL", "Diode_SMD:D_SOD-882"),
        ("U6", "SRV05-4", "Package_TO_SOT_SMD:SOT-23-6"),
        ("U7", "PROT", "Package_TO_SOT_SMD:SOT-23-6_ESD_Array"),
        ("TVS1", "SMBJ5.0A", "Diode_SMD:D_SMB"),
    ],
)
def test_esd_protection_devices_are_classified(reference: str, value: str, footprint: str) -> None:
    assert is_esd_protection_device(reference, value, footprint)


@pytest.mark.parametrize(
    ("reference", "value", "footprint"),
    [
        ("U1", "STM32F072C8T6", "Package_QFP:LQFP-48_7x7mm_P0.5mm"),
        ("U2", "AMS1117-3.3", "Package_TO_SOT_SMD:SOT-223-3_TabPin2"),
        ("U3", "SP3485EN", "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"),
    ],
)
def test_regular_ics_are_not_esd_devices(reference: str, value: str, footprint: str) -> None:
    assert not is_esd_protection_device(reference, value, footprint)
