"""GUI utilities: colours, layout paths, config I/O, theme building."""
import json
import os

import dearpygui.dearpygui as dpg

import funautoscripter as fl


OVERLAYS = ("ly_roi", "ly_cand", "ly_heat", "ly_mag", "ly_per", "ly_axis")

# Intensity colour ramp (blue = slow, red = fast).
HEAT = [(0.0, (28, 38, 92)), (0.2, (36, 96, 200)), (0.45, (36, 190, 140)), (0.7, (240, 210, 60)), (1.0, (235, 60, 50))]


def crop_rect(crop, zoom, sw, sh, use_zoom=True):
    """Rectangle (x, y, w, h) of the original video that the engine analyses. Mirrors decode_cmd."""
    x, y, w, h = 0, 0, sw, sh
    spec = crop
    if spec == "left":
        w = sw // 2
    elif spec == "right":
        x, w = sw // 2, sw // 2
    elif spec == "top":
        h = sh // 2
    elif spec == "bottom":
        y, h = sh // 2, sh // 2
    elif spec:
        w, h, x, y = (int(v) for v in spec.split(":"))
    if use_zoom and zoom:
        zx0, zy0, zx1, zy1 = fl.parse_zoom(zoom)
        x, y = x + int(w * zx0), y + int(h * zy0)
        w, h = max(2, int(w * (zx1 - zx0))), max(2, int(h * (zy1 - zy0)))
    return x, y, w, h


def fmt_time(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"


def mix(a, b, t):
    """Blend two RGB colours: t = 0 gives a, t = 1 gives b."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def line_theme(rgb, weight=1.5):
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, rgb, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight, category=dpg.mvThemeCat_Plots)
    return th


def make_theme(sp):
    """Global DearPyGui theme (windows, buttons, sliders, plots) built from a THEMES entry."""
    bg, tx, ac, dk = sp["bg"], sp["text"], sp["accent"], sp["dark"]
    black_white = (0, 0, 0) if dk else (255, 255, 255)
    plot_bg = mix(bg, black_white, 0.35 if dk else 0.7)
    core = {
        dpg.mvThemeCol_WindowBg: bg, dpg.mvThemeCol_ChildBg: mix(bg, tx, 0.03),
        dpg.mvThemeCol_PopupBg: mix(bg, tx, 0.06), dpg.mvThemeCol_Border: mix(bg, tx, 0.22),
        dpg.mvThemeCol_FrameBg: mix(bg, tx, 0.10), dpg.mvThemeCol_FrameBgHovered: mix(bg, tx, 0.17),
        dpg.mvThemeCol_FrameBgActive: mix(bg, tx, 0.24),
        dpg.mvThemeCol_TitleBg: mix(bg, tx, 0.07), dpg.mvThemeCol_TitleBgActive: mix(bg, ac, 0.4),
        dpg.mvThemeCol_TitleBgCollapsed: mix(bg, tx, 0.05), dpg.mvThemeCol_MenuBarBg: mix(bg, tx, 0.07),
        dpg.mvThemeCol_ScrollbarBg: mix(bg, tx, 0.03), dpg.mvThemeCol_ScrollbarGrab: mix(bg, tx, 0.3),
        dpg.mvThemeCol_ScrollbarGrabHovered: mix(bg, tx, 0.42), dpg.mvThemeCol_ScrollbarGrabActive: mix(bg, tx, 0.55),
        dpg.mvThemeCol_CheckMark: ac, dpg.mvThemeCol_SliderGrab: ac, dpg.mvThemeCol_SliderGrabActive: mix(ac, tx, 0.3),
        dpg.mvThemeCol_NavHighlight: ac, dpg.mvThemeCol_NavCursor: ac,
        dpg.mvThemeCol_Button: mix(bg, ac, 0.55 if dk else 0.35), dpg.mvThemeCol_ButtonHovered: mix(bg, ac, 0.78 if dk else 0.55),
        dpg.mvThemeCol_ButtonActive: ac,
        dpg.mvThemeCol_Header: mix(bg, ac, 0.4), dpg.mvThemeCol_HeaderHovered: mix(bg, ac, 0.6),
        dpg.mvThemeCol_HeaderActive: mix(bg, ac, 0.8),
        dpg.mvThemeCol_Separator: mix(bg, tx, 0.25), dpg.mvThemeCol_SeparatorHovered: ac, dpg.mvThemeCol_SeparatorActive: ac,
        dpg.mvThemeCol_Text: tx, dpg.mvThemeCol_TextDisabled: sp["muted"],
        dpg.mvThemeCol_Tab: mix(bg, ac, 0.25), dpg.mvThemeCol_TabHovered: mix(bg, ac, 0.7),
        dpg.mvThemeCol_TabSelected: mix(bg, ac, 0.5), dpg.mvThemeCol_TabDimmed: mix(bg, tx, 0.08),
        dpg.mvThemeCol_TabDimmedSelected: mix(bg, ac, 0.3), dpg.mvThemeCol_DockingEmptyBg: mix(bg, black_white, 0.3),
        dpg.mvThemeCol_ResizeGrip: mix(bg, tx, 0.2), dpg.mvThemeCol_ResizeGripHovered: mix(bg, ac, 0.7),
        dpg.mvThemeCol_ResizeGripActive: ac, dpg.mvThemeCol_PlotHistogram: ac,
    }
    plots = {
        dpg.mvPlotCol_FrameBg: mix(bg, tx, 0.03), dpg.mvPlotCol_PlotBg: plot_bg,
        dpg.mvPlotCol_PlotBorder: mix(bg, tx, 0.25), dpg.mvPlotCol_LegendBg: mix(bg, tx, 0.06),
        dpg.mvPlotCol_LegendBorder: mix(bg, tx, 0.25), dpg.mvPlotCol_LegendText: tx,
        dpg.mvPlotCol_TitleText: tx, dpg.mvPlotCol_InlayText: tx, dpg.mvPlotCol_AxisText: tx,
        dpg.mvPlotCol_AxisGrid: mix(plot_bg, tx, 0.18), dpg.mvPlotCol_AxisTick: mix(plot_bg, tx, 0.4),
        dpg.mvPlotCol_Crosshairs: tx,
    }
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvAll):
            for what, col in core.items():
                dpg.add_theme_color(what, col + (255,), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_DockingPreview, ac + (170,), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, ac + (110,), category=dpg.mvThemeCat_Core)
            for what, col in plots.items():
                dpg.add_theme_color(what, col + (255,), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Selection, ac + (255,), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 3, category=dpg.mvThemeCat_Core)
    return th


def heat_rgb(x):
    x = min(max(float(x), 0.0), 1.0)
    for (x0, c0), (x1, c1) in zip(HEAT, HEAT[1:]):
        if x <= x1:
            return mix(c0, c1, (x - x0) / (x1 - x0))
    return HEAT[-1][1]


def hue_rgb(x):
    """Colour for a normalised angle (0 to 1, wraps around) used in the "direction" mode."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(float(x) % 1.0, 0.62, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def nice_step(span, width):
    """Time marker step (s) so that markers are roughly 70 px apart."""
    for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200):
        if span / s <= max(1, width / 70.0):
            return s
    return 7200


def layout_path():
    """Window layout (docking), stored next to gui.json."""
    return os.path.join(os.path.dirname(config_path()), "layout.ini")


def config_path():
    """GUI settings file (FUNAUTOSCRIPTER_CONFIG overrides the path, used in tests)."""
    p = os.environ.get("FUNAUTOSCRIPTER_CONFIG")
    if p:
        return p
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "funautoscripter", "gui.json")


def load_config():
    try:
        with open(config_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}
