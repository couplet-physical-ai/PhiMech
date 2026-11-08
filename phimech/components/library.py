"""Parametric envelopes for common mechanical and electronic parts.

The library deliberately emits the same small CSG vocabulary as the rest of PhiMech.  The
models are envelopes rather than vendor meshes: keeping the dimensions, mounting holes, and
outside-world access features here lets synthesis stay deterministic even when a detailed
vendor model is not installed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from phimech.csg.ast import Color, Cube, Cylinder, Group, Node, Rotate, Translate
from phimech.components.envelope import Access, Envelope, Hole

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Build = Callable[[str, Dict[str, Any]], Envelope]




@dataclass(frozen=True)
class PartSpec:
    """Description and factory for one library key.

    Keeping defaults beside the factory makes the registry useful to agents for discovery,
    while the factory remains the single source of truth for derived dimensions.
    """

    key: str
    name: str
    category: str
    params: Dict[str, Any]
    build: Build


def _rgb(body: Node, rgba: Sequence[float], name: str) -> Node:
    """Colour a placeholder without changing its dimensions in either backend."""
    return Color(tuple(float(v) for v in rgba), [body], name)  # type: ignore[arg-type]


def _box(size: Vec3, rgba: Sequence[float] = (0.55, 0.58, 0.62, 1.0)) -> Node:
    return _rgb(Cube(tuple(float(v) for v in size)), rgba, "envelope")


def _cylinder_body(d: float, h: float, axis: str) -> Node:
    """Make a cylinder whose local bounding box starts at (0, 0, 0).

    The AST cylinder is +Z-only.  Translating the rotated primitive by its radius preserves
    the component-local origin convention used by :class:`Envelope`.
    """
    r = float(d) / 2.0
    primitive = Cylinder(h=float(h), r=r, segments=48)
    if axis == "x":
        return Translate((0.0, r, r), [Rotate((0.0, 90.0, 0.0), [primitive])])
    if axis == "y":
        return Translate((r, 0.0, r), [Rotate((-90.0, 0.0, 0.0), [primitive])])
    if axis == "z":
        return Translate((r, r, 0.0), [primitive])
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def _cylinder_size(d: float, h: float, axis: str) -> Vec3:
    if axis == "x":
        return (float(h), float(d), float(d))
    if axis == "y":
        return (float(d), float(h), float(d))
    if axis == "z":
        return (float(d), float(d), float(h))
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def _env(
    key: str,
    ref: str,
    name: str,
    size: Vec3,
    body: Node,
    *,
    holes: Sequence[Hole] = (),
    access: Sequence[Access] = (),
    thermal_w: float = 0.0,
    mount_plane_z: float = 0.0,
    mount_thickness: float = 0.0,
    footprint: Sequence[Vec2] = (),
) -> Envelope:
    return Envelope(
        ref=ref,
        size=tuple(float(v) for v in size),
        footprint=tuple((float(x), float(y)) for x, y in footprint),
        holes=tuple(holes),
        access=tuple(access),
        mount_plane_z=float(mount_plane_z),
        mount_thickness=float(mount_thickness),
        body=body,
        name=name,
        source="library:" + key,
        thermal_w=float(thermal_w),
    )


def _box_spec(
    key: str,
    name: str,
    category: str,
    size: Vec3,
    *,
    holes: Sequence[Hole] = (),
    access: Sequence[Access] = (),
    thermal_w: float = 0.0,
) -> PartSpec:
    def build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        return _env(
            key,
            ref,
            name,
            size,
            _box(size),
            holes=holes,
            access=access,
            thermal_w=thermal_w,
        )

    return PartSpec(key, name, category, {}, build)


def _meta_box_spec(
    key: str,
    name: str,
    category: str,
    size: Vec3,
    *,
    meta: Dict[str, Any],
    holes: Sequence[Hole] = (),
    access: Sequence[Access] = (),
    thermal_w: float = 0.0,
) -> PartSpec:
    """Add metadata to a box without changing the existing box-spec helper contract."""
    base = _box_spec(
        key,
        name,
        category,
        size,
        holes=holes,
        access=access,
        thermal_w=thermal_w,
    )

    def build(ref: str, params: Dict[str, Any]) -> Envelope:
        return replace(base.build(ref, params), meta=deepcopy(meta))

    return PartSpec(key, name, category, {}, build)


def _cylinder_spec(
    key: str,
    name: str,
    category: str,
    d: float,
    h: float,
    *,
    axis: str = "z",
    params: Optional[Dict[str, Any]] = None,
    holes: Sequence[Hole] = (),
    access: Sequence[Access] = (),
) -> PartSpec:
    defaults = dict(params or {})
    parameterized_orientation = "orientation" in defaults

    def build(ref: str, values: Dict[str, Any]) -> Envelope:
        orientation = str(values.get("orientation", axis)) if parameterized_orientation else axis
        size = _cylinder_size(d, h, orientation)
        return _env(
            key,
            ref,
            name,
            size,
            _rgb(_cylinder_body(d, h, orientation), (0.32, 0.34, 0.37, 1.0), "cylinder"),
            holes=holes,
            access=access,
        )

    return PartSpec(key, name, category, defaults, build)


def _battery_spec(
    key: str,
    name: str,
    d: float,
    h: float,
    *,
    orientation: str = "x",
) -> PartSpec:
    defaults = {"orientation": orientation}

    def build(ref: str, values: Dict[str, Any]) -> Envelope:
        axis = str(values.get("orientation", orientation))
        size = _cylinder_size(d, h, axis)
        return _env(
            key,
            ref,
            name,
            size,
            _rgb(_cylinder_body(d, h, axis), (0.15, 0.16, 0.18, 1.0), "cell"),
        )

    return PartSpec(key, name, "battery", defaults, build)


def _variable_box_spec(
    key: str,
    name: str,
    category: str,
    defaults: Dict[str, Any],
    size_fn: Callable[[Dict[str, Any]], Vec3],
    *,
    holes_fn: Optional[Callable[[Vec3, Dict[str, Any]], Sequence[Hole]]] = None,
    access_fn: Optional[Callable[[Vec3, Dict[str, Any]], Sequence[Access]]] = None,
) -> PartSpec:
    def build(ref: str, values: Dict[str, Any]) -> Envelope:
        size = size_fn(values)
        holes = holes_fn(size, values) if holes_fn else ()
        access = access_fn(size, values) if access_fn else ()
        return _env(key, ref, name, size, _box(size), holes=holes, access=access)

    return PartSpec(key, name, category, dict(defaults), build)


def _meta_variable_box_spec(
    key: str,
    name: str,
    category: str,
    defaults: Dict[str, Any],
    size_fn: Callable[[Dict[str, Any]], Vec3],
    meta_fn: Callable[[Vec3, Dict[str, Any]], Dict[str, Any]],
    *,
    holes_fn: Optional[Callable[[Vec3, Dict[str, Any]], Sequence[Hole]]] = None,
    access_fn: Optional[Callable[[Vec3, Dict[str, Any]], Sequence[Access]]] = None,
) -> PartSpec:
    """Attach metadata computed from the resolved dimensions."""
    base = _variable_box_spec(
        key, name, category, defaults, size_fn, holes_fn=holes_fn, access_fn=access_fn
    )

    def build(ref: str, values: Dict[str, Any]) -> Envelope:
        envelope = base.build(ref, values)
        return replace(envelope, meta=deepcopy(meta_fn(envelope.size, values)))

    return PartSpec(key, name, category, dict(defaults), build)




def _display_spec(
    key: str,
    name: str,
    size: Vec3,
    window: Vec2,
    *,
    connector: Access,
    holes: Sequence[Hole] = (),
) -> PartSpec:
    w, d, _ = size
    access = (
        Access(
            "WINDOW",
            "top",
            (w / 2.0, d / 2.0),
            window,
            through_lid=True,
        ),
        connector,
    )
    return _box_spec(key, name, "display", size, holes=holes, access=access)


def _module_body(size: Vec3, components: Sequence[Tuple[Vec3, Vec3]]) -> Node:
    """Add simple raised placeholders while retaining a board-sized base."""
    children: List[Node] = [_box(size)]
    for position, block_size in components:
        children.append(Translate(position, [_box(block_size, (0.12, 0.13, 0.15, 1.0))]))
    return Group(children, name="module")


def _make_specs() -> Dict[str, PartSpec]:
    parts: Dict[str, PartSpec] = {}

    # Batteries.  Cylindrical cells default to a horizontal axis so a tray can stack them.
    parts["18650"] = _battery_spec("18650", "18650 Li-ion cell", 18.6, 65.2)
    parts["21700"] = _battery_spec("21700", "21700 Li-ion cell", 21.7, 70.0)
    parts["aa"] = _battery_spec("aa", "AA alkaline/NiMH cell", 14.5, 50.5)
    parts["aaa"] = _battery_spec("aaa", "AAA alkaline/NiMH cell", 10.5, 44.5)
    parts["cr2032"] = _battery_spec(
        "cr2032", "CR2032 coin cell", 20.0, 3.2, orientation="z"
    )

    def lipo_size(p: Dict[str, Any]) -> Vec3:
        return (float(p["w"]), float(p["d"]), float(p["h"]))

    parts["lipo-pouch"] = _variable_box_spec(
        "lipo-pouch",
        "LiPo pouch cell",
        "battery",
        {"w": 50.0, "d": 34.0, "h": 6.0},
        lipo_size,
    )
    parts["9v"] = _box_spec("9v", "9 V rectangular battery", "battery", (48.5, 26.5, 17.5))
    parts["cr123a"] = _battery_spec("cr123a", "CR123A Li-ion cell", 17.0, 34.5)
    # This holder mass is an estimate, not a datasheet measurement.
    parts["cr2032-holder"] = _meta_box_spec(
        "cr2032-holder",
        "CR2032 through-hole holder",
        "power",
        (20.0, 25.0, 6.0),
        meta={"mass_g_estimate": 4.0},
    )

    # Displays: top windows are through-lid features; connectors are on a board edge.
    parts["ssd1306-0.96"] = _display_spec(
        "ssd1306-0.96",
        "0.96 inch SSD1306 OLED module",
        (27.3, 27.8, 4.1),
        (21.7, 11.2),
        connector=Access("I2C", "back", (13.65, 2.0), (8.0, 3.0), protrusion=1.0),
    )
    parts["ssd1306-1.3"] = _display_spec(
        "ssd1306-1.3",
        "1.3 inch SSD1306 OLED module",
        (35.0, 34.0, 4.5),
        (29.0, 22.0),
        connector=Access("I2C", "back", (17.5, 2.2), (8.0, 3.0), protrusion=1.0),
    )
    parts["ili9341-2.4"] = _display_spec(
        "ili9341-2.4",
        "2.4 inch ILI9341 TFT module",
        (60.0, 42.7, 4.5),
        (48.6, 36.9),
        connector=Access("SPI", "back", (30.0, 2.2), (14.0, 3.0), protrusion=1.0),
        holes=(
            Hole((3.0, 3.0), 3.0),
            Hole((57.0, 3.0), 3.0),
            Hole((3.0, 39.7), 3.0),
            Hole((57.0, 39.7), 3.0),
        ),
    )
    parts["lcd1602"] = _display_spec(
        "lcd1602",
        "1602 character LCD module",
        (80.0, 36.0, 12.5),
        (71.0, 24.0),
        connector=Access("HEADER", "back", (40.0, 6.5), (16.0, 3.0), protrusion=1.0),
        holes=(
            Hole((2.5, 2.5), 2.5),
            Hole((77.5, 2.5), 2.5),
            Hole((2.5, 33.5), 2.5),
            Hole((77.5, 33.5), 2.5),
        ),
    )
    parts["epaper-2.13"] = _display_spec(
        "epaper-2.13",
        "2.13 inch e-paper module",
        (59.2, 29.2, 1.2),
        (48.5, 23.7),
        connector=Access("SPI", "back", (29.6, 1.0), (14.0, 3.0), protrusion=1.0),
    )

    # Boards and modules.  Coordinates are the published board-local mounting patterns.
    rpi_zero_holes = (
        Hole((3.5, 3.5), 2.75),
        Hole((61.5, 3.5), 2.75),
        Hole((3.5, 26.5), 2.75),
        Hole((61.5, 26.5), 2.75),
    )
    rpi_zero_access = (
        Access("USB1", "front", (41.4, 2.5), (8.0, 3.0), protrusion=1.0),
        Access("USB2", "front", (54.0, 2.5), (8.0, 3.0), protrusion=1.0),
        Access("HDMI", "front", (12.4, 2.5), (11.2, 3.2), protrusion=1.0),
        Access("SD", "left", (15.0, 2.5), (12.0, 3.0), protrusion=0.5),
    )

    def rpi_zero_build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        size = (65.0, 30.0, 5.0)
        body = _module_body(size, [((18.0, 5.0, 3.0), (16.0, 20.0, 2.0))])
        return _env(
            "raspberry-pi-zero-2w",
            ref,
            "Raspberry Pi Zero 2 W",
            size,
            body,
            holes=rpi_zero_holes,
            access=rpi_zero_access,
            thermal_w=2.5,
        )

    parts["raspberry-pi-zero-2w"] = PartSpec(
        "raspberry-pi-zero-2w",
        "Raspberry Pi Zero 2 W",
        "sbc",
        {},
        rpi_zero_build,
    )

    rpi4_holes = (
        Hole((3.5, 3.5), 2.7),
        Hole((61.5, 3.5), 2.7),
        Hole((3.5, 52.5), 2.7),
        Hole((61.5, 52.5), 2.7),
    )
    rpi4_access = (
        Access("USB1", "right", (15.0, 8.0), (14.0, 11.0), protrusion=1.0),
        Access("USB2", "right", (32.0, 8.0), (14.0, 11.0), protrusion=1.0),
        Access("ETHERNET", "right", (47.0, 8.0), (16.0, 14.0), protrusion=1.0),
        Access("USB-C", "front", (12.0, 8.0), (10.0, 4.0), protrusion=1.0),
        Access("HDMI1", "front", (30.0, 8.0), (8.0, 3.0), protrusion=1.0),
        Access("HDMI2", "front", (43.0, 8.0), (8.0, 3.0), protrusion=1.0),
        Access("AUDIO", "front", (70.0, 8.0), (8.0, 6.0), protrusion=1.0),
    )

    def rpi4_build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        size = (85.0, 56.0, 16.0)
        body = _module_body(size, [((5.0, 8.0, 3.0), (55.0, 40.0, 8.0))])
        return _env(
            "raspberry-pi-4b",
            ref,
            "Raspberry Pi 4 Model B",
            size,
            body,
            holes=rpi4_holes,
            access=rpi4_access,
            thermal_w=6.0,
        )

    parts["raspberry-pi-4b"] = PartSpec(
        "raspberry-pi-4b", "Raspberry Pi 4 Model B", "sbc", {}, rpi4_build
    )

    # NVIDIA Jetson developer kits. Overall envelopes are NVIDIA's published kit dimensions
    # (Nano dev kit 100 x 80 x 29 mm; Orin Nano dev kit 103 x 90.5 x 34.6 mm, module, heatsink
    # and fan included). The Nano carrier (P3449) exposes four mounting holes on the commonly
    # documented 86 x 58 mm pattern; the hole centres below are that pattern and are flagged in
    # `meta` as to be verified against the P3449 CAD before anything is drilled. The port row
    # is one window on the +X edge: individual connector positions are NOT taken from the
    # drawing, so a case that needs per-port holes has to measure the real board.
    jetson_nano_holes = (
        Hole((4.0, 17.0), 3.0, "H1"),
        Hole((90.0, 17.0), 3.0, "H2"),
        Hole((4.0, 75.0), 3.0, "H3"),
        Hole((90.0, 75.0), 3.0, "H4"),
    )

    def jetson_nano_build(ref: str, params: Dict[str, Any]) -> Envelope:
        """Board-level placeholder: carrier, module-plus-heatsink block, port row."""
        del params
        size = (100.0, 80.0, 29.0)
        board = _rgb(Cube((100.0, 80.0, 1.6)), (0.08, 0.34, 0.20, 1.0), "carrier_pcb")
        module = Translate(
            (10.0, 22.0, 1.6),
            [_rgb(Cube((70.0, 45.0, 27.4)), (0.55, 0.58, 0.62, 1.0), "module_heatsink")],
        )
        ports = Translate(
            (84.0, 4.0, 1.6),
            [_rgb(Cube((16.0, 72.0, 16.0)), (0.70, 0.72, 0.74, 1.0), "port_row")],
        )
        header = Translate(
            (10.0, 76.0, 1.6),
            [_rgb(Cube((51.0, 5.0, 8.5)), (0.09, 0.10, 0.11, 1.0), "gpio_header")],
        )
        body = Group([board, module, ports, header], name="jetson_nano_devkit")
        envelope = _env(
            "jetson-nano-devkit",
            ref,
            "NVIDIA Jetson Nano Developer Kit (B01)",
            size,
            body,
            holes=jetson_nano_holes,
            access=(
                # One window over the whole port row (DC jack, micro-USB, HDMI, DP, 4x USB-A,
                # RJ45): the row is ~16 mm tall above the carrier.
                Access("IO", "right", (40.0, 9.8), (74.0, 18.0), protrusion=1.0),
            ),
            thermal_w=10.0,
            mount_thickness=1.6,
        )
        return replace(
            envelope,
            meta={
                "mass_g_estimate": 140.0,
                "holes_source": "86 x 58 mm pattern as commonly documented for the P3449 "
                "carrier; verify against NVIDIA's dev-kit CAD before drilling",
                "power_w_modes": (5.0, 10.0),
                "power_input": "5 V barrel jack (5.5 mm) or micro-USB",
            },
        )

    parts["jetson-nano-devkit"] = PartSpec(
        "jetson-nano-devkit", "NVIDIA Jetson Nano Developer Kit (B01)", "sbc", {}, jetson_nano_build
    )

    def jetson_orin_nano_build(ref: str, params: Dict[str, Any]) -> Envelope:
        """Envelope of the Orin Nano / Orin NX developer kit (P3768 carrier + reference fan).

        NVIDIA's dev-kit documentation does not publish carrier mounting-hole positions (the
        kit ships screwed to a plastic base), so no holes are declared: mount it with a
        `pocket` or `clip`, or capture its edges, until the P3768 design files are measured.
        """
        del params
        size = (103.0, 90.5, 34.6)
        board = _rgb(Cube((103.0, 90.5, 1.6)), (0.08, 0.34, 0.20, 1.0), "carrier_pcb")
        module = Translate(
            (12.0, 24.0, 1.6),
            [_rgb(Cube((70.0, 45.0, 33.0)), (0.55, 0.58, 0.62, 1.0), "module_heatsink_fan")],
        )
        ports = Translate(
            (87.0, 5.0, 1.6),
            [_rgb(Cube((16.0, 80.0, 16.0)), (0.70, 0.72, 0.74, 1.0), "port_row")],
        )
        body = Group([board, module, ports], name="jetson_orin_nano_devkit")
        envelope = _env(
            "jetson-orin-nano-devkit",
            ref,
            "NVIDIA Jetson Orin Nano / Orin NX Developer Kit",
            size,
            body,
            access=(
                Access("IO", "right", (45.25, 9.8), (82.0, 18.0), protrusion=1.0),
            ),
            thermal_w=15.0,
            mount_thickness=1.6,
        )
        return replace(
            envelope,
            meta={
                "mass_g_estimate": 250.0,
                "holes_source": "none declared: the dev kit ships on a plastic base and NVIDIA "
                "publishes no carrier hole positions; pocket/clip mount or measure P3768",
                "power_w_modes": (7.0, 15.0, 25.0),
                "power_input": "9-19 V DC barrel jack (5.5 x 2.5 mm)",
            },
        )

    parts["jetson-orin-nano-devkit"] = PartSpec(
        "jetson-orin-nano-devkit",
        "NVIDIA Jetson Orin Nano / Orin NX Developer Kit",
        "sbc",
        {},
        jetson_orin_nano_build,
    )

    # Intel RealSense D435i: 90 x 25 x 25 mm, one 1/4-20 UNC tripod socket centred on the
    # bottom and two M3 threaded points on the back, 45 mm apart (Intel D400 series datasheet).
    # The mounting face is the bottom, so the single hole is the tripod socket; its `d` is the
    # 6.6 mm clearance a through-bolt needs, not the thread, because the hole is what the boss
    # is bored to. The USB-C port is on an end face; which end is an envelope estimate.
    def d435i_build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        size = (90.0, 25.0, 25.0)
        shell = _rgb(Cube(size), (0.62, 0.65, 0.68, 1.0), "camera_body")
        glass = Translate(
            (2.0, -0.4, 2.0),
            [_rgb(Cube((86.0, 0.6, 21.0)), (0.05, 0.05, 0.06, 1.0), "front_glass")],
        )
        body = Group([shell, glass], name="realsense_d435i")
        envelope = _env(
            "realsense-d435i",
            ref,
            "Intel RealSense D435i depth camera",
            size,
            body,
            holes=(Hole((45.0, 12.5), 6.6, "TRIPOD"),),
            access=(
                Access("LENS", "front", (45.0, 12.5), (86.0, 21.0), tolerance=0.5),
                Access("USB-C", "right", (12.5, 12.5), (9.0, 4.0), protrusion=1.0),
            ),
            thermal_w=1.5,
        )
        return replace(
            envelope,
            meta={
                "mass_g_estimate": 72.0,
                "back_m3_holes": ((22.5, 12.5), (67.5, 12.5)),
                "back_m3_spacing_mm": 45.0,
                "tripod_thread": "1/4-20 UNC",
                "fov_deg": {"depth": (87.0, 58.0), "rgb": (69.0, 42.0)},
            },
        )

    parts["realsense-d435i"] = PartSpec(
        "realsense-d435i", "Intel RealSense D435i depth camera", "camera", {}, d435i_build
    )

    def esp32_build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        size = (51.5, 28.0, 13.0)
        body = _module_body(size, [((15.0, 3.0, 3.0), (18.0, 22.0, 8.0))])
        return _env(
            "esp32-devkit-v1",
            ref,
            "ESP32 DevKit V1",
            size,
            body,
            access=(Access("USB", "left", (14.0, 6.5), (8.0, 5.0), protrusion=1.0),),
            thermal_w=1.0,
        )

    parts["esp32-devkit-v1"] = PartSpec(
        "esp32-devkit-v1", "ESP32 DevKit V1", "module", {}, esp32_build
    )

    uno_holes = (
        Hole((14.0, 2.5), 3.2),
        Hole((15.3, 50.7), 3.2),
        Hole((66.1, 7.6), 3.2),
        Hole((66.1, 35.5), 3.2),
    )
    parts["arduino-uno-r3"] = _box_spec(
        "arduino-uno-r3",
        "Arduino Uno Rev3",
        "module",
        (68.6, 53.3, 15.0),
        holes=uno_holes,
        access=(
            Access("USB-B", "left", (38.0, 8.5), (12.0, 11.0), protrusion=1.0),
            Access("BARREL", "left", (13.0, 8.0), (10.0, 10.0), shape="round", protrusion=1.0),
        ),
        thermal_w=0.8,
    )
    parts["arduino-nano"] = _box_spec(
        "arduino-nano",
        "Arduino Nano",
        "module",
        (45.0, 18.0, 8.0),
        access=(Access("USB", "front", (22.5, 4.0), (8.0, 5.0), protrusion=1.0),),
    )
    parts["teensy-4.1"] = _box_spec(
        "teensy-4.1",
        "Teensy 4.1",
        "module",
        (61.0, 18.0, 8.5),
        access=(Access("USB", "front", (30.5, 4.0), (8.0, 5.0), protrusion=1.0),),
        thermal_w=1.0,
    )
    # This mass is an estimate, not a datasheet measurement.
    def teensy_40_build(ref: str, params: Dict[str, Any]) -> Envelope:
        """Board-level placeholder so an assembly view shows the MCU and connectors."""
        del params
        size = (35.6, 17.8, 4.5)
        board = _rgb(Cube((35.6, 17.8, 1.6)), (0.08, 0.34, 0.20, 1.0), "teensy_pcb")
        mcu = Translate(
            (14.0, 5.1, 1.6),
            [_rgb(Cube((8.0, 8.0, 1.2)), (0.08, 0.09, 0.10, 1.0), "imxrt1062")],
        )
        usb = Translate(
            (13.8, 0.0, 1.6),
            [_rgb(Cube((8.0, 5.0, 2.8)), (0.70, 0.72, 0.74, 1.0), "micro_usb")],
        )
        flash = Translate(
            (24.5, 6.5, 1.6),
            [_rgb(Cube((4.8, 6.0, 1.0)), (0.10, 0.11, 0.12, 1.0), "flash")],
        )
        crystal = Translate(
            (8.0, 6.8, 1.6),
            [_rgb(Cube((3.2, 2.5, 0.9)), (0.68, 0.70, 0.72, 1.0), "crystal")],
        )
        headers = []
        for y in (0.5, 15.3):
            headers.append(
                Translate(
                    (1.0, y, 1.6),
                    [_rgb(Cube((33.6, 2.0, 2.4)), (0.09, 0.10, 0.11, 1.0), "header")],
                )
            )
        body = Group([board, mcu, usb, flash, crystal] + headers, name="teensy_4_0_board")
        envelope = _env(
            "teensy-4.0",
            ref,
            "Teensy 4.0",
            size,
            body,
            access=(Access("USB", "front", (17.8, 2.25), (8.0, 5.0), protrusion=1.0),),
            thermal_w=0.5,
        )
        return replace(envelope, meta={"mass_g_estimate": 3.0})

    parts["teensy-4.0"] = PartSpec(
        "teensy-4.0", "Teensy 4.0", "module", {}, teensy_40_build
    )
    parts["esp32-c3-supermini"] = _box_spec(
        "esp32-c3-supermini",
        "ESP32-C3 SuperMini",
        "module",
        (24.5, 18.0, 4.5),
        access=(Access("USB-C", "front", (12.25, 2.5), (8.0, 4.0), protrusion=1.0),),
        thermal_w=0.5,
    )
    parts["raspberry-pi-pico"] = _box_spec(
        "raspberry-pi-pico",
        "Raspberry Pi Pico",
        "module",
        (51.0, 21.0, 3.9),
        access=(Access("USB", "front", (25.5, 2.0), (8.0, 4.0), protrusion=1.0),),
    )

    # Power and panel hardware.
    parts["tp4056-usbc"] = _box_spec(
        "tp4056-usbc",
        "TP4056 USB-C Li-ion charger",
        "power",
        (26.0, 17.5, 5.0),
        access=(Access("USB-C", "front", (13.0, 2.5), (8.0, 4.0), protrusion=1.0),),
    )

    def buck_build(ref: str, params: Dict[str, Any]) -> Envelope:
        del params
        size = (43.0, 21.0, 14.0)
        children: List[Node] = [_box(size)]
        for x in (11.0, 32.0):
            children.append(
                Translate((x, 10.5, 10.0), [_cylinder_body(9.0, 4.0, "z")])
            )
        return _env("buck-lm2596", ref, "LM2596 buck converter", size, Group(children, "module"))

    parts["buck-lm2596"] = PartSpec("buck-lm2596", "LM2596 buck converter", "power", {}, buck_build)

    # A generic high-current buck module (28.8 V battery -> 19 V, XL4016/LM5116 300 W class
    # boards with bolt-on heatsinks). The default envelope is that board class, not a specific
    # vendor drawing: size it with `w`/`d`/`h` once the real module is in hand. Screw terminals
    # sit at the two X ends; they are wired internally, so no access cutout is declared.
    def buck_module_build(ref: str, values: Dict[str, Any]) -> Envelope:
        size = (float(values["w"]), float(values["d"]), float(values["h"]))
        children: List[Node] = [_box(size)]
        # Two bolt-on heatsink fins so the render reads as a power board, not a blank block.
        fin_w = max(4.0, size[0] * 0.18)
        for x in (2.0, size[0] - fin_w - 2.0):
            children.append(
                Translate((x, 2.0, 0.0), [_box((fin_w, size[1] - 4.0, size[2]), (0.3, 0.32, 0.35, 1.0))])
            )
        envelope = _env(
            "buck-module",
            ref,
            "DC-DC buck converter module (28.8 V -> 19 V class)",
            size,
            Group(children, "module"),
            thermal_w=3.0,
        )
        return replace(
            envelope,
            meta={
                "mass_g_estimate": 60.0,
                "electrical": {"vin_v": (12.0, 40.0), "vout_v": 19.0, "iout_a_max": 10.0},
                "envelope_source": "XL4016-class 300 W module with heatsinks; verify against "
                "the purchased unit",
            },
        )

    parts["buck-module"] = PartSpec(
        "buck-module",
        "DC-DC buck converter module (28.8 V -> 19 V class)",
        "power",
        {"w": 65.0, "d": 47.0, "h": 24.0},
        buck_module_build,
    )

    # Amass XT30U in a printed panel pass-through: the housing is a 10 x 8 mm rectangle, 17.5 mm
    # long, mating along local -Y (`front`). Rotate it 90/270 about Z to mate through a side
    # wall. A printed wall holds it only by the opening, so a design should note the retention.
    parts["xt30-panel"] = _meta_box_spec(
        "xt30-panel",
        "XT30 connector, panel pass-through",
        "panel",
        (10.0, 17.5, 8.0),
        access=(Access("MATE", "front", (5.0, 4.0), (10.0, 8.0), tolerance=0.3),),
        meta={
            "mass_g_estimate": 3.0,
            "electrical": {"i_cont_a": 30.0, "pitch_mm": 5.0},
            "envelope_source": "Amass XT30U housing, nominal; verify the panel cutout against "
            "the connector in hand",
        },
    )
    parts["relay-srd-05vdc"] = _box_spec(
        "relay-srd-05vdc",
        "Songle SRD-05VDC relay",
        "power",
        (19.0, 15.5, 15.5),
        access=(Access("TERMINALS", "front", (9.5, 7.0), (13.0, 8.0)),),
    )
    parts["dc-barrel-jack-5.5"] = _cylinder_spec(
        "dc-barrel-jack-5.5",
        "5.5 mm DC barrel panel jack",
        "power",
        11.0,
        14.0,
        axis="x",
        access=(Access("THROUGH", "front", (7.0, 5.5), (11.0, 11.0), shape="round"),),
    )
    parts["rocker-switch-kcd1"] = _box_spec(
        "rocker-switch-kcd1",
        "KCD1 rocker switch",
        "panel",
        (21.0, 15.0, 18.0),
        access=(
            Access("CUTOUT", "top", (10.5, 7.5), (19.0, 13.0), through_lid=True),
        ),
    )
    parts["momentary-12mm"] = _cylinder_spec(
        "momentary-12mm",
        "12 mm momentary push button",
        "panel",
        12.0,
        18.0,
        access=(Access("PANEL_HOLE", "top", (6.0, 6.0), (12.0, 12.0), shape="round"),),
    )
    parts["usb-c-panel-mount"] = _box_spec(
        "usb-c-panel-mount",
        "USB-C panel-mount extension",
        "panel",
        (27.0, 15.0, 8.0),
        access=(Access("USB-C", "front", (13.5, 4.0), (9.0, 4.5), protrusion=1.0),),
    )
    parts["sma-connector"] = _cylinder_spec(
        "sma-connector",
        "SMA bulkhead connector",
        "panel",
        6.35,
        16.0,
        access=(Access("THROUGH", "top", (3.175, 3.175), (6.35, 6.35), shape="round"),),
    )

    # Small hardware is represented as a solid placeholder plus a hole where appropriate.
    parts["m3-heat-set-insert"] = _cylinder_spec(
        "m3-heat-set-insert",
        "M3 brass heat-set insert",
        "fastener",
        4.6,
        5.0,
        access=(Access("BORE", "top", (2.3, 2.3), (3.2, 3.2), shape="round"),),
    )

    def screw_size(p: Dict[str, Any]) -> Vec3:
        return (5.5, 5.5, float(p["l"]) + 2.4)

    parts["m3-screw"] = _variable_box_spec(
        "m3-screw", "M3 pan-head screw", "fastener", {"l": 12.0}, screw_size
    )

    def standoff_size(p: Dict[str, Any]) -> Vec3:
        return (5.0, 5.0, float(p["h"]))

    def standoff_holes(size: Vec3, p: Dict[str, Any]) -> Sequence[Hole]:
        del p
        return (Hole((size[0] / 2.0, size[1] / 2.0), 2.7),)

    parts["m2.5-standoff"] = _variable_box_spec(
        "m2.5-standoff",
        "M2.5 threaded standoff",
        "fastener",
        {"h": 6.0},
        standoff_size,
        holes_fn=standoff_holes,
    )
    parts["rubber-foot-8mm"] = _cylinder_spec(
        "rubber-foot-8mm", "8 mm rubber foot", "hardware", 8.0, 3.0
    )

    def fan_spec(key: str, name: str, size: float, thickness: float, hole_d: float) -> PartSpec:
        holes = (
            Hole((4.0, 4.0), hole_d),
            Hole((size - 4.0, 4.0), hole_d),
            Hole((4.0, size - 4.0), hole_d),
            Hole((size - 4.0, size - 4.0), hole_d),
        )
        access = (
            Access("FAN", "top", (size / 2.0, size / 2.0), (size - 2.0, size - 2.0), shape="round"),
        )
        return _box_spec(key, name, "fan", (size, size, thickness), holes=holes, access=access)

    parts["40mm-fan"] = fan_spec("40mm-fan", "40 mm axial fan", 40.0, 10.0, 3.5)
    parts["30mm-fan"] = fan_spec("30mm-fan", "30 mm axial fan", 30.0, 10.0, 3.0)
    parts["5015-blower"] = _box_spec(
        "5015-blower",
        "5015 radial blower fan",
        "fan",
        (50.0, 50.0, 15.0),
        holes=(
            Hole((4.0, 4.0), 3.0),
            Hole((46.0, 4.0), 3.0),
            Hole((4.0, 46.0), 3.0),
            Hole((46.0, 46.0), 3.0),
        ),
        access=(Access("OUTLET", "right", (25.0, 9.0), (30.0, 12.0), protrusion=1.0),),
    )

    return parts


PARTS: Dict[str, PartSpec] = _make_specs()


def _unknown_part(part: str) -> KeyError:
    matches = get_close_matches(part, sorted(PARTS), n=4, cutoff=0.25)
    hint = "; close matches: " + ", ".join(matches) if matches else ""
    return KeyError("unknown library part {!r}{}".format(part, hint))


def get(part: str) -> PartSpec:
    """Return a registry entry, with useful suggestions for a misspelled key."""
    try:
        return PARTS[part]
    except KeyError:
        raise _unknown_part(part) from None


def _merged_params(spec: PartSpec, supplied: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(supplied) - set(spec.params))
    if unknown:
        raise TypeError(
            "{} does not accept parameter(s): {}; accepted: {}".format(
                spec.key, ", ".join(unknown), ", ".join(sorted(spec.params)) or "none"
            )
        )
    values = dict(spec.params)
    values.update(supplied)
    return values


def make(part: str, ref: str = "U1", **params: Any) -> Envelope:
    """Build a fresh envelope for ``part`` and ``ref``.

    Unknown keys fail early because a typo in a component source is much safer to diagnose
    than silently producing a nominally valid but dimensionally wrong enclosure.
    """
    spec = get(part)
    return spec.build(ref, _merged_params(spec, params))


def list_parts() -> List[Dict[str, Any]]:
    """List discoverable parts, including default dimensions for agent/UI browsing."""
    out: List[Dict[str, Any]] = []
    for spec in PARTS.values():
        envelope = spec.build("__preview__", dict(spec.params))
        out.append(
            {
                "key": spec.key,
                "name": spec.name,
                "category": spec.category,
                "params": dict(spec.params),
                "size": list(envelope.size),
            }
        )
    return out


__all__ = ["PARTS", "PartSpec", "get", "list_parts", "make"]
