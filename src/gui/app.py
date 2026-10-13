"""GUI App class and main() for FunAutoScripter."""
import argparse
import copy
import json
import os
import sys
import threading
import time
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
import funautoscripter as fl
from src.gui.utils import (crop_rect, fmt_time, mix, line_theme, make_theme, heat_rgb, hue_rgb,
                            nice_step, layout_path, config_path, load_config, OVERLAYS, HEAT)
from src.gui.config import (CW, CH, MAXC, PALETTE_RGB, PARAMS, TIPS, TIPS_UI, AUTO_HELP, AXMODES,
                             CROPS, OV_H, OV_BAND, CH_COLORS, WINDOWS, THEMES, DEFAULT_THEME,
                             PRESET_ATTRS, PARAM_BY_ATTR, BUILTIN_PRESETS, presets_path,
                             clean_param, load_user_presets)
from src.gui.frame_source import FrameSource


def _fmt_eta(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


class App:
    def __init__(self, video=None, captura=None, tempo=None):
        self.alive = True
        self.a = fl.build_parser().parse_args(["_"])        # default values only
        self.a.video = None
        self.defaults = {k: getattr(self.a, k) for k in PRESET_ATTRS}
        self.user_presets = load_user_presets()
        self.cfg = load_config()
        self.load_params()
        self.cfg_last = None
        self.cfg_time = 0.0
        self.src = None
        self.flow = None            # (dx, dy, mag, W, H)
        self.pending_flow = None
        self.pending = None
        self.worker_flow = None     # (dx, dy, mag) for the background thread only
        self.worker_fid = 0
        self.fid = 0                # identifier of the currently applied flow
        self.chapters = [{"start": 0.0, "zoom": None, "region": None, "invert": False, "axis_ref": None, "why": None}]
        self.active = 0
        self.chap_items = []
        self.chap_done = False
        self.bulk = False
        self._fit_memo = {}
        self.warn_text = ""
        self.bulk_msg = ""
        self.auto_flow = bool(captura)   # only tests trigger flow automatically
        self.sig = None              # crop, zoom, width, cell and fps with which the applied flow was computed
        self.worker_sig = None
        self.working = False
        self.cancel_flag = False
        self.flowing = False
        self.res = None
        self.dbg = []
        self.status = "Open a video or a folder."
        self.error = None
        self.progress = 0.0
        self.need_flow = False
        self.need_compute = False
        self.last_change = 0.0
        self.compute_ms = 0.0
        self.t = 0.0
        self.playing = False
        self.speed = 1.0
        self.last_tick = time.perf_counter()
        self.mode = "none"
        self.view = (0, 0, CW, CH)
        self.cur_win = -1
        self.dirty_view = True
        self.fresh = False
        self.drag_until = 0.0
        self.rect_reset = False
        self.sel = None
        self.drag = None
        self.folder_items = {}
        self.rows = []
        self.folder = None
        self.bulk_done = False
        self.captura, self.tempo_ini = captura, tempo
        self.cap_state = 0
        self.settle = 0
        tn = self.cfg.get("theme")
        self.theme_name = tn if tn in THEMES else DEFAULT_THEME
        self.pal = THEMES[self.theme_name]
        self.styled = []             # (item, tipo) dos textos que mudam de cor com o tema
        self.series_rgb = {}         # series tag -> (colour or "fg", thickness)
        self.theme_items = {}        # temas criados, para os apagar ao trocar
        self.layout_path = layout_path()
        self.have_layout = False
        self.layout_reset = False
        self.layout_t = time.time()
        self.frame = 0
        self.ov_other = {}
        self.ov_res_key = None
        self.prog_prefix, self.prog_t0 = "", time.time()
        self.accel_lines = None
        self.preset_ref = self.cfg.get("preset_ref", "Default")
        if self.preset_ref not in BUILTIN_PRESETS and self.preset_ref not in self.user_presets:
            self.preset_ref = "Default"
        self.pending_apply = False
        self.ov_memo = {}
        self.axis_drag = None
        self.axinfo = ""
        self.suggested = None
        self.suggest_armed = 0.0
        self.ov_res_val = None           # chapter -> (t, fs) for chapters other than the active one
        self.ov_cache = []
        self.ov_dirty = True
        self.need_ov = False
        self.ov_w, self.ov_D = 0, 0.0
        self.ov_drag = None
        self.ov_hover = None
        self.ov_dyn_key = None
        self.ov_redraw_t = 0.0
        self.vid_progress = []   # [(fraction, overlay_text)] indexed by row, updated by bulk thread
        self.build_ui()
        self.apply_ui_config()
        threading.Thread(target=self.worker, daemon=True).start()
        if video:
            self.open_path(video)

    # ------------------------------------------------------------ settings
    def load_params(self):
        """Applies saved values from the last session to the namespace (ignores invalid entries)."""
        saved = self.cfg.get("params", {})
        for grp, attr, label, kind, lo, hi, fmt, stage in PARAMS:
            if attr not in saved:
                continue
            v = saved[attr]
            try:
                if kind == "int" or attr == "fps":
                    v = int(v)
                elif kind == "float":
                    v = min(max(float(v), lo), hi)
                elif kind == "bool":
                    v = bool(v)
                elif v not in hi:
                    continue
            except (TypeError, ValueError):
                continue
            setattr(self.a, attr, v)
        name = self.cfg.get("crop")
        if name in CROPS:
            self.a.crop = CROPS[name]

    def apply_ui_config(self):
        """Restores widget-only state (overlays, follow cursor, speed, crop)."""
        c = self.cfg
        for tag in OVERLAYS:
            if tag in c.get("overlays", {}):
                dpg.set_value(tag, bool(c["overlays"][tag]))
        if "follow" in c:
            dpg.set_value("follow", bool(c["follow"]))
        if isinstance(c.get("follow_w"), (int, float)):
            dpg.set_value("follow_w", float(c["follow_w"]))
        sp = c.get("speed")
        if isinstance(sp, str) and sp in ("0.25x", "0.5x", "1x", "2x"):
            dpg.set_value("p_speed", sp)
            self.speed = float(sp.rstrip("x"))
        name = c.get("crop")
        if name in CROPS:
            dpg.set_value("p_crop", name)

    def snapshot_config(self):
        a = self.a
        crop_name = next((k for k, v in CROPS.items() if v == a.crop), "full video")
        return {
            "params": {attr: getattr(a, attr) for _, attr, *_ in PARAMS},
            "crop": crop_name,
            "overlays": {t: bool(dpg.get_value(t)) for t in OVERLAYS},
            "follow": bool(dpg.get_value("follow")),
            "follow_w": float(dpg.get_value("follow_w")),
            "speed": dpg.get_value("p_speed"),
            "theme": self.theme_name,
            "windows": {t: bool(dpg.get_value("mi_" + t)) for t, _, _ in WINDOWS},
            "ov_mode": dpg.get_value("ov_mode"),
            "live": dpg.get_value("p_live"),
            "preset_ref": self.preset_ref,
            "show_raw": bool(dpg.get_value("show_raw")),
            "last_path": self.cfg.get("ultimo") or self.cfg.get("last_path"),
        }

    def save_config(self, force=False):
        """Saves settings if they changed (at most once per second, or immediately with force)."""
        now = time.time()
        if not force and now - self.cfg_time < 1.0:
            return
        self.cfg_time = now
        try:
            snap = self.snapshot_config()
        except Exception:
            return
        if snap == self.cfg_last:
            return
        self.cfg_last = snap
        self.cfg = snap
        try:
            path = config_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except OSError:
            pass

    # ------------------------------------------------ stale flow detection
    @staticmethod
    def flow_sig(a):
        return {"crop": a.crop, "zoom": a.zoom, "width": a.width, "cell": a.cell, "fps": a.fps}

    def cache_ok(self):
        """Is there a cached flow that matches current settings? (If so, opening the video is instant.)"""
        return fl.flow_available(self.a)

    def fps(self):
        return self.sig["fps"] if self.sig else self.a.fps

    def geometry_stale(self):
        """The selected crop or zoom no longer matches the computed flow."""
        return self.sig is not None and (self.sig["crop"] != self.a.crop or self.sig["zoom"] != self.a.zoom)

    def stale_reasons(self):
        if self.sig is None:
            return []
        now = self.flow_sig(self.a)
        names = {"crop": "3D/VR crop", "zoom": "zoom", "width": "analysis width",
                 "cell": "cell size", "fps": "analysis fps"}
        return [names[k] for k in names if now[k] != self.sig[k]]

    # ------------------------------------------------------------------ background worker
    def request_compute(self):
        if self.bulk:
            return
        self.need_compute = True
        self.last_change = time.time()

    def cancel(self):
        """Stops flow computation. Everything stays as it was before the button was clicked."""
        if self.flowing or self.bulk:
            self.cancel_flag = True
            self.status = "cancelling..."

    def request_flow(self):
        if self.src is None or self.bulk:
            return
        if self.geometry_stale():
            self.a.region = None            # old region was relative to the old geometry
            self.update_sel_text()
            self.sync_chapter()
        self.need_flow = True
        self.last_change = time.time()

    def worker(self):
        while self.alive:
            time.sleep(0.03)
            if not (self.need_flow or self.need_compute or self.need_ov):
                continue
            if time.time() - self.last_change < 0.12:
                continue
            ov_only = not (self.need_flow or self.need_compute)
            self.working = not ov_only      # overview update does not block anything: can be interrupted
            try:
                if self.need_flow:
                    self.need_flow = self.need_compute = False
                    self.do_flow()
                elif self.need_compute:
                    self.need_compute = False
                    self.do_compute()
                elif time.time() - self.last_change >= 0.8 and not self.bulk:
                    # only when idle: rebuild the overview for other chapters
                    self.need_ov = False
                    self.do_overview()
            except fl.Cancelled:
                self.status = "calculation cancelled"
                self.progress = 0.0
            except (Exception, SystemExit) as e:
                self.error = f"error: {e}"
                self.status = self.error
            finally:
                self.working = False
                self.flowing = False
                self.cancel_flag = False

    def busy(self):
        return self.need_flow or self.need_compute or self.working or self.bulk

    def start_progress(self, prefix):
        """Status text while flow runs: what is being done, with what backend and at what speed."""
        self.prog_prefix, self.prog_t0, self.progress = prefix, time.time(), 0.0
        self.status = prefix

    def on_progress(self, n, total):
        if self.cancel_flag:
            raise fl.Cancelled()
        self.progress = n / total if total else 0.0
        el = time.time() - self.prog_t0
        if self.prog_prefix and el > 1.0 and n:
            fps_calc = n / el
            eta = (total - n) / fps_calc if fps_calc > 0 else 0.0
            self.status = f"{self.prog_prefix} | {fl.BACKEND['text']} | {fps_calc:.0f} fps | ETA {_fmt_eta(eta)}"

    def do_flow(self):
        a = copy.copy(self.a)
        self.flowing = True
        self.cancel_flag = False
        self.start_progress("Computing optical flow (Esc to stop)")
        _, W, H, _ = fl.decode_cmd(a)
        sig = self.flow_sig(a)
        dx, dy, mag = fl.flow_for(a, self.on_progress)
        self.prog_prefix = ""
        self.progress = 1.0
        self.worker_fid += 1
        self.worker_flow = (dx, dy, mag)
        self.worker_sig = sig
        self.pending_flow = (dx, dy, mag, W, H, sig, self.worker_fid)
        self.do_compute()

    def do_compute(self):
        if self.worker_flow is None:
            return
        a = copy.copy(self.a)
        for k, v in self.worker_sig.items():     # stored flow parameters take precedence
            setattr(a, k, v)
        a.invert = self.eff_invert()              # the active chapter's invert flag applies here
        if self.chapters[self.active].get("axis_ref"):
            a.axis_ref = str(self.chapters[self.active]["axis_ref"])   # and its axis-ref too
        dx, dy, mag = self.worker_flow
        t0 = time.time()
        dbg = []
        r = fl.compute(a, dx, dy, mag, dbg)
        self.pending = self.derive(a, r, dbg, mag)
        self.compute_ms = (time.time() - t0) * 1000.0
        elapsed = time.time() - self.prog_t0
        self.status = f"ready  (done in {elapsed:.1f} s)"
        self.need_ov = True

    def derive(self, a, r, dbg, mag):
        v, windows = r["v"], r["windows"]
        T, fps = len(v), a.fps
        t = a.start + np.arange(T) / fps
        pts = r["pts"]
        if pts:
            kx = np.array([x for x, _ in pts]) / 1000.0
            ky = np.array([y for _, y in pts])
            fs = np.interp(t, kx, ky)
        else:
            kx, ky, fs = np.array([]), np.array([]), np.zeros(T)
        widx = np.zeros(T, int)
        for k, (s, e, _) in enumerate(windows):
            widx[s:e] = k
        out = dict(r)
        out.update(fid=self.worker_fid, t=t, raw=np.cumsum(v), fs=fs, fs_speed=np.gradient(fs) * fps, kx=kx, ky=ky,
                   widx=widx, dbg=dbg, vm=float(np.percentile(np.abs(v), 99)) + 1e-9,
                   mag_max=float(np.percentile(mag[::5].astype(np.float32), 99)) + 1e-9)
        for key in ("mag", "period", "score"):
            out["ch_" + key] = np.array([dbg[k]["chosen_info"][key] for k in widx], float)
        out["best"] = np.array([dbg[k]["best_score"] for k in widx], float)
        # the initial selection is not a switch
        out["flip_t"] = [float(t[windows[k][0]]) for k, d in enumerate(dbg) if d.get("axis_flip")]
        out["switch_t"] = [float(t[windows[d["window"]][0]]) for d in dbg
                           if d["switched"] and d["prev"] is not None]
        return out

    # --------------------------------------------------------------- video
    def open_path(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(path):
            self.cfg["last_path"] = path
        if os.path.isdir(path):
            vids = fl.find_videos(path, False)
            self.folder = path
            self.fill_folder(vids)
            self.show_window("w_folder", True)
            self.status = f"{len(vids)} videos in {path}"
            if vids and not self.bulk:
                self.load_video(vids[0])
        elif os.path.exists(path):
            self.load_video(path)
        else:
            self.status = f"not found: {path}"

    def load_video(self, path):
        a = self.a
        if self.src:
            self.src.close()
        a.video = path
        a.output = os.path.splitext(path)[0] + ".funscript"
        self.src = FrameSource(path)
        self.chapters = fl.load_chapters("auto", path) or self.default_chapters()
        self.active = 0
        self.apply_chapter_args()
        self.fresh = True
        self.set_mode("none")
        dpg.set_value("out_path", a.output)
        self.update_sel_text()
        dpg.set_viewport_title("FunAutoScripter: " + os.path.basename(path))
        for i, v in enumerate(self.rows):
            dpg.set_value(f"vs_{i}", v == path)
        # no saved chapters: detect them now (if the full video flow exists or is computed automatically)
        auto = (not os.path.exists(fl.chapters_file(path)) and getattr(self, "auto_detect", True)
                and (self.auto_flow or fl.cache_valid(fl._full_args(self.a))))
        self.reset_flow_state(request=not auto)
        self.refresh_chapters()
        if auto:
            self.suggest_chapters()

    # ----------------------------------------------------------------- UI
    def txt(self, kind, text="", **kw):
        """Text whose colour depends on the theme (kind: text, head, muted, warn)."""
        item = dpg.add_text(text, color=self.pal[kind], **kw)
        self.styled.append((item, kind))
        return item

    def fit_text(self, text, px):
        """Truncates text with "..." to fit in px pixels (single line). Returns full text if measurement fails."""
        try:
            if px <= 20 or dpg.get_text_size(text)[0] <= px:
                return text
            lo, hi = 0, len(text)
            while lo < hi:                      # longest prefix that fits with the ellipsis
                mid = (lo + hi + 1) // 2
                if dpg.get_text_size(text[:mid].rstrip() + "...")[0] <= px:
                    lo = mid
                else:
                    hi = mid - 1
            return text[:lo].rstrip() + "..."
        except Exception:
            return text

    def set_fitted(self, tag, tip, text):
        """Puts text on one line, truncated to the available width of the control window. Full text
        goes in the tooltip. Only recalculates when text or width changes."""
        try:
            win = dpg.get_item_rect_size("w_ctrl")[0]
        except Exception:
            win = dpg.get_item_width("w_ctrl")
        x = dpg.get_item_pos(tag)[0]            # relative to the control window
        avail = int(win - max(x, 0) - 24)
        memo = self._fit_memo.get(tag)
        if memo == (text, avail):
            return
        self._fit_memo[tag] = (text, avail)
        dpg.set_value(tag, self.fit_text(text, avail))
        dpg.set_value(tip, text)

    def win(self, tag, shown, **kw):
        """Dockable window. The title (without internal id) is what identifies the window in the layout file."""
        label = next(l for t, l, _ in WINDOWS if t == tag)
        return dpg.window(tag=tag, label=label, use_internal_label=False, show=shown[tag],
                          on_close=lambda s: self.on_win_close(tag), **kw)

    def build_ui(self):
        dpg.create_context()
        try:
            os.makedirs(os.path.dirname(self.layout_path), exist_ok=True)
        except OSError:
            pass
        if not os.path.exists(self.layout_path):
            try:
                import shutil
                src = os.path.join(os.path.dirname(__file__), "default_layout.ini")
                if os.path.exists(src):
                    shutil.copy2(src, self.layout_path)
            except OSError:
                pass
        self.have_layout = os.path.exists(self.layout_path)
        dpg.configure_app(docking=True, docking_space=True, docking_shift_only=False, init_file=self.layout_path)
        self.buf = np.zeros((CH, CW, 4), np.float32)
        self.buf[..., 3] = 1.0
        self.buf[..., :3] = 0.09
        self.flat = self.buf.reshape(-1)
        with dpg.texture_registry():
            dpg.add_raw_texture(CW, CH, self.flat, format=dpg.mvFormat_Float_rgba, tag="tex")
        self.cand_themes = [line_theme(c, 1.8) for c in PALETTE_RGB]
        with dpg.theme() as self.th_stale:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 110, 20, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 135, 30, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 160, 50, 255))
        with dpg.theme() as self.th_changed:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (190, 110, 20, 120))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (215, 130, 30, 160))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (235, 150, 40, 190))
        with dpg.theme() as self.th_ok:
            with dpg.theme_component(dpg.mvButton):
                pass

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.on_file, tag="dlg_file",
                             width=720, height=420, modal=True):
            for ext in (".*", ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"):
                dpg.add_file_extension(ext)
        dpg.add_file_dialog(directory_selector=True, show=False, callback=self.on_file, tag="dlg_dir",
                            width=720, height=420, modal=True)
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.on_export_cfg,
                             tag="dlg_export_cfg", width=720, height=420, modal=True, default_filename="settings.json"):
            dpg.add_file_extension(".json")
            dpg.add_file_extension(".*")
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.on_import_cfg,
                             tag="dlg_import_cfg", width=720, height=420, modal=True):
            dpg.add_file_extension(".json")
            dpg.add_file_extension(".*")

        shown = {t: bool(self.cfg.get("windows", {}).get(t, d)) for t, _, d in WINDOWS}
        self.build_menu(shown)
        with self.win("w_ctrl", shown):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Recalculate flow", tag="btn_flow", callback=lambda: self.request_flow())
                dpg.add_button(label="Save funscript", tag="btn_save_top", callback=lambda: self.save())
                dpg.add_button(label="Cancel", tag="btn_cancel", show=False, callback=lambda: self.cancel())
                dpg.add_progress_bar(tag="prog", width=200, default_value=0.0)
                dpg.add_text("", tag="status")
                with dpg.tooltip("status"):
                    dpg.add_text("", tag="status_tip", wrap=600)
            self.txt("warn", "", tag="warn")
            with dpg.tooltip("warn"):
                dpg.add_text("", tag="warn_tip", wrap=600)
            with dpg.group(tag="batch_ctrl", show=False):
                dpg.add_separator()
                self.txt("head", "Batch")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="All", tag="b_all", callback=lambda: self.mark_all(True))
                    dpg.add_button(label="None", tag="b_none", callback=lambda: self.mark_all(False))
                    dpg.add_button(label="No script", tag="b_missing", callback=lambda: self.mark_missing())
                dpg.add_checkbox(label="redo existing scripts", tag="b_over", default_value=False)
                dpg.add_checkbox(label="apply current zoom and region to all", tag="b_same", default_value=False)
                dpg.add_checkbox(label="detect chapters (and axis-ref)", tag="b_chap", default_value=True)
                dpg.add_button(label="Process selected", tag="btn_bulk", width=-1,
                               callback=lambda: self.start_bulk())
                dpg.add_button(label="Apply settings (without recalculating flow)", tag="btn_bulk_apply",
                               width=-1, callback=lambda: self.start_bulk(cache_only=True))
                dpg.add_text("", tag="bulk_text", wrap=0)
        with self.win("w_video", shown):
            self.build_video_panel()
        with self.win("w_ov", shown, no_scrollbar=True):
            self.build_overview()
        with self.win("w_params", shown):
            self.build_params()
        with self.win("w_insp", shown):
            self.build_inspector()
        with self.win("w_folder", shown):
            self.build_folder_panel()
        with self.win("w_status", shown):
            self.build_status_panel()
        with self.win("w_tl", shown, no_scrollbar=True):
            self.build_timeline()

        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: self.key("space"))
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: self.cancel())
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=lambda: self.key("left"))
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda: self.key("right"))
            dpg.add_key_press_handler(dpg.mvKey_I, callback=lambda: self.key("i"))
            dpg.add_key_press_handler(dpg.mvKey_Open_Brace, callback=lambda: self.key("prev"))
            dpg.add_key_press_handler(dpg.mvKey_Close_Brace, callback=lambda: self.key("next"))
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=lambda: self.key("home"))
            dpg.add_key_press_handler(dpg.mvKey_Return, callback=lambda: self.key("enter"))
            dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda: self.key("s"))
            dpg.add_key_press_handler(dpg.mvKey_D, callback=lambda: self.key("d"))
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: self.key("r"))
            dpg.add_key_press_handler(dpg.mvKey_N, callback=lambda: self.key("n"))
            dpg.add_key_press_handler(dpg.mvKey_G, callback=lambda: self.key("g"))
        with dpg.window(label="Keyboard shortcuts", tag="w_keys", show=False, width=420, height=290, pos=(300, 200)):
            for key, what in self.SHORTCUTS:
                with dpg.group(horizontal=True):
                    dpg.add_text(key, color=(255, 200, 80))
                    dpg.add_text(what)
            dpg.add_separator()
            dpg.add_text("Hover over any element to see what it does.", wrap=380)
        with dpg.window(label="Acceleration", tag="w_accel", show=False, width=760, height=420, pos=(200, 150)):
            dpg.add_input_text(tag="accel_text", multiline=True, readonly=True, width=-1, height=-1)
        self.apply_tips()
        dpg.create_viewport(title="FunAutoScripter", width=1640, height=1010)
        dpg.set_viewport_resize_callback(self.on_resize)
        dpg.setup_dearpygui()
        self.set_theme(self.theme_name)

    SHORTCUTS = [
        ("Space", "play or pause"),
        ("Left and right arrows", "one frame (with Shift, ten)"),
        ("I", "invert the active chapter"),
        ("[ and ]", "previous and next chapter"),
        ("Home", "start of the active chapter"),
        ("D", "draw axis-ref direction on the video"),
        ("Enter", "apply pending slider changes"),
        ("Ctrl+O", "open video"),
        ("Ctrl+Shift+O", "open folder"),
        ("Ctrl+S", "save the funscript"),
        ("Esc", "cancel computation"),
        ("R", "recalculate flow"),
        ("N", "new chapter at cursor"),
        ("G", "suggest chapters (automatic)"),
    ]

    def typing(self):
        """Is a text field active? If so, keys are not shortcuts."""
        return any(dpg.does_item_exist(t) and dpg.is_item_active(t) for t in ("preset_name", "out_path"))

    def key(self, k):
        if self.typing():
            return
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        if k == "space":
            self.toggle_play()
        elif k in ("left", "right"):
            self.frame_step((10 if shift else 1) * (-1 if k == "left" else 1))
        elif k == "o":
            if ctrl and shift:
                dpg.show_item("dlg_dir")
            elif ctrl:
                dpg.show_item("dlg_file")
        elif k == "s":
            if ctrl:
                self.save()
        elif k == "r":
            if not ctrl:
                self.request_flow()
        elif k == "n" and not ctrl:
            if self.src is not None:
                self.new_chapter()
        elif k == "g" and not ctrl:
            if self.src is not None:
                self.suggest_chapters()
        elif self.src is None or ctrl:
            return
        elif k == "i":
            self.on_chap_invert(None, not self.chapters[self.active].get("invert"))
        elif k in ("prev", "next"):
            j = self.active + (-1 if k == "prev" else 1)
            if 0 <= j < len(self.chapters):
                self.activate(j)
        elif k == "home":
            self.scrub_to(self.chapters[self.active]["start"])
        elif k == "d":
            self.set_mode("axis" if self.mode != "axis" else "none")
        elif k == "enter" and self.pending_apply:
            self.apply_pending()

    def run_accel_test(self):
        """GPU and CPU diagnostic (fl.accel_report), runs in a thread and shown in a window."""
        dpg.set_value("accel_text", "Testing... (first OpenCL run compiles kernels, may take a few seconds)")
        dpg.show_item("w_accel")
        a = copy.copy(self.a)

        def work():
            try:
                self.accel_lines = fl.accel_report(a)
            except Exception as e:
                self.accel_lines = [f"test error: {e}"]
        threading.Thread(target=work, daemon=True).start()

    def apply_tips(self):
        """Hover tooltip for each element (parameters and buttons)."""
        tips = {"p_" + k: v for k, v in TIPS.items()}
        tips.update(TIPS_UI)
        for tag, text in tips.items():
            if dpg.does_item_exist(tag):
                with dpg.tooltip(tag):
                    dpg.add_text(text, wrap=340)

    # ------------------------------------------------ menu, windows, theme, layout
    def build_menu(self, shown):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open video...    Ctrl+O", callback=lambda: dpg.show_item("dlg_file"))
                dpg.add_menu_item(label="Open folder...   Ctrl+Shift+O", callback=lambda: dpg.show_item("dlg_dir"))
                dpg.add_separator()
                dpg.add_menu_item(label="Export settings...", callback=lambda: dpg.show_item("dlg_export_cfg"))
                dpg.add_menu_item(label="Import settings...", callback=lambda: dpg.show_item("dlg_import_cfg"))
            with dpg.menu(label="Windows"):
                for tag, label, _ in WINDOWS:
                    dpg.add_menu_item(label=label, check=True, default_value=shown[tag], tag="mi_" + tag,
                                      callback=(lambda t: lambda s, v: self.show_window(t, bool(v)))(tag))
                dpg.add_separator()
                dpg.add_menu_item(label="Save layout now", callback=lambda: self.save_layout(msg=True))
                dpg.add_menu_item(label="Reset layout (restart)", callback=lambda: self.reset_layout())
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Keyboard shortcuts", callback=lambda: dpg.show_item("w_keys"))
                dpg.add_menu_item(label="Test acceleration (GPU / OpenCL)...", callback=lambda: self.run_accel_test())
            with dpg.menu(label="Theme"):
                for name in THEMES:
                    dpg.add_menu_item(label=name, check=True, default_value=(name == self.theme_name),
                                      tag="th_" + name, callback=(lambda n: lambda s, v: self.set_theme(n))(name))

    def show_window(self, tag, on):
        dpg.configure_item(tag, show=on)
        dpg.set_value("mi_" + tag, on)
        if on:
            dpg.focus_item(tag)
        self.ov_dirty = True

    def on_win_close(self, tag):
        dpg.set_value("mi_" + tag, False)

    def bind_series(self, tag):
        rgb, wgt = self.series_rgb[tag]
        th = line_theme(self.pal["fg"] if rgb == "fg" else rgb, wgt)
        old = self.theme_items.get(tag)
        dpg.bind_item_theme(tag, th)
        self.theme_items[tag] = th
        if old is not None:
            dpg.delete_item(old)

    def set_theme(self, name):
        sp = THEMES.get(name)
        if sp is None:
            return
        self.theme_name, self.pal = name, sp
        old = self.theme_items.get("global")
        th = make_theme(sp)
        dpg.bind_theme(th)
        self.theme_items["global"] = th
        if old is not None:
            dpg.delete_item(old)
        dpg.set_viewport_clear_color(list(sp["bg"]) + [255])
        for item, kind in self.styled:
            dpg.configure_item(item, color=sp[kind])
        for tag in self.series_rgb:
            self.bind_series(tag)
        for i in range(5):
            if dpg.does_item_exist(f"ph{i}"):
                dpg.configure_item(f"ph{i}", color=list(sp["fg"]) + [200])
        for n in THEMES:
            dpg.set_value("th_" + n, n == name)
        self.ov_dirty = True

    def save_layout(self, msg=False):
        if self.layout_reset:
            return
        try:
            dpg.save_init_file(self.layout_path)
            if msg:
                self.status = "layout saved"
        except Exception:
            pass

    def reset_layout(self):
        try:
            os.remove(self.layout_path)
        except OSError:
            pass
        self.layout_reset = True            # do not save it again on exit
        self.status = "layout reset: close and reopen the program"

    def apply_default_layout(self):
        """First run (no layout file): arranges windows side by side. With the layout file active,
        DearPyGui ignores creation-time positions, so they are applied after the window is open."""
        W, H = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        y0 = 24
        ctrl_h = 56
        y1 = y0 + ctrl_h
        ov_h = OV_H + 70
        rh = 500 if H >= 900 else max(300, int((H - y1 - ov_h) * 0.6))
        vw = CW + 44
        pw = 360
        iw = max(300, W - vw - pw)
        boxes = {
            "w_ctrl": (0, y0, W, ctrl_h),
            "w_video": (0, y1, vw, rh),
            "w_params": (vw, y1, pw, rh),
            "w_insp": (vw + pw, y1, iw, rh),
            "w_folder": (max(0, W - 330), y1, 330, rh),
            "w_ov": (0, y1 + rh, W, ov_h),
            "w_tl": (0, y1 + rh + ov_h, W, max(200, H - (y1 + rh + ov_h))),
        }
        for tag, (x, y, w, h) in boxes.items():
            dpg.set_item_pos(tag, [x, y])
            dpg.set_item_width(tag, w)
            dpg.set_item_height(tag, h)

    def build_video_panel(self):
        with dpg.plot(tag="vplot", width=CW + 14, height=CH + 14, no_title=True, no_menus=True,
                      no_mouse_pos=True, no_box_select=True, equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, tag="vx", no_gridlines=True, no_tick_marks=True,
                              no_tick_labels=True, lock_min=True, lock_max=True)
            with dpg.plot_axis(dpg.mvYAxis, tag="vy", no_gridlines=True, no_tick_marks=True,
                               no_tick_labels=True, lock_min=True, lock_max=True):
                dpg.add_image_series("tex", [0, 0], [CW, CH], tag="vimg")
            dpg.set_axis_limits("vx", 0, CW)
            dpg.set_axis_limits("vy", 0, CH)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play/Pause", tag="btn_play", callback=lambda: self.toggle_play())
            dpg.add_combo(["0.25x", "0.5x", "1x", "2x"], default_value="1x", width=70, tag="p_speed", callback=self.on_speed)
            self.txt("muted", "seek in the Overview window")
        dpg.add_text("", tag="t_label")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="follow cursor", tag="follow", default_value=True)
            dpg.add_slider_float(label="width (s)", tag="follow_w", default_value=20.0, min_value=3.0,
                                 max_value=120.0, format="%.0f", width=140)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="ROI", tag="ly_roi", default_value=True)
            dpg.add_checkbox(label="candidates", tag="ly_cand", default_value=True)
            dpg.add_checkbox(label="window average", tag="ly_heat", default_value=False)
            dpg.add_checkbox(label="frame magnitude", tag="ly_mag", default_value=False)
            dpg.add_checkbox(label="periodicity", tag="ly_per", default_value=False)
            dpg.add_checkbox(label="axis", tag="ly_axis", default_value=True)
        with dpg.group(horizontal=True):
            dpg.add_text("Selection:")
            dpg.add_radio_button(["none", "region", "zoom", "direction"], horizontal=True, default_value="none",
                                 tag="sel_mode", callback=lambda s, v: self.set_mode({"none": "none", "region": "region", "zoom": "zoom", "direction": "axis"}[v]))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply selection", tag="b_sel_apply", callback=lambda: self.apply_selection())
            dpg.add_button(label="Clear", tag="b_sel_clear", callback=lambda: self.clear_selection())
            dpg.add_text("region: automatic   zoom: none", tag="sel_text")

    def build_params(self):
        dpg.add_text("3D/VR Crop")
        dpg.add_combo(list(CROPS), default_value=next((k for k, v in CROPS.items() if v == self.a.crop), "full video"), width=200, tag="p_crop", callback=self.on_crop)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="recalculate while dragging", tag="p_live", default_value=bool(self.cfg.get("live", True)),
                             callback=lambda s, v: self.on_live(v))
            dpg.add_button(label="Apply changes", tag="btn_apply", callback=lambda: self.apply_pending())
        self.txt("muted", "Without the checkbox, sliders only change the value and calculation happens on the button.", wrap=0)
        dpg.add_checkbox(label="invert this chapter", tag="chap_inv", callback=self.on_chap_invert)
        self.txt("muted", "To correct the direction at the end. Saved to the video's .chapters.json.", wrap=0)
        self.txt("head", "Chapter direction (axis-ref)")
        dpg.add_combo(AXMODES, tag="chap_axmode", default_value=AXMODES[0], width=200)
        dpg.add_slider_int(label="angle (°)", tag="chap_ang", default_value=90, min_value=0, max_value=359, width=170)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply direction", tag="btn_apply_axis", callback=lambda: self.apply_chap_axis())
            dpg.add_button(label="Draw on video", tag="b_axis_draw", callback=lambda: self.set_mode("axis"))
        self.txt("muted", "", tag="chap_axinfo", wrap=0)
        dpg.add_separator()
        self.txt("head", "Presets")
        dpg.add_combo(self.preset_names(), tag="p_preset", width=200, default_value=self.preset_ref, callback=self.on_preset)
        self.txt("muted", "", tag="preset_info", wrap=0)
        self.txt("warn", "", tag="preset_diff", wrap=0)
        dpg.add_button(label="Reset to preset", tag="b_pre_reset", show=False, callback=lambda: self.reset_to_preset())
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="preset_name", hint="name", width=130)
            dpg.add_button(label="Save", tag="b_pre_save", callback=lambda: self.save_preset())
            dpg.add_button(label="Delete", tag="b_pre_del", callback=lambda: self.delete_preset())
        self.txt("muted", "Signal parameters only. Does not affect flow or invert.", wrap=0)
        with dpg.collapsing_header(label="Chapters (automatic or manual)", default_open=True, tag="chap_hdr"):
            self.txt("muted", "Each chapter has its own axis-ref, zoom and region. The funscript joins them "
                              "over time. Without chapters, the whole video is one.", wrap=0)
            dpg.add_button(label="Suggest chapters (automatic)", width=-1, tag="btn_suggest",
                           callback=lambda: self.suggest_chapters())
            self.txt("muted", "Detects cuts and axis or zone changes from the full video flow and assigns "
                              "each chapter its axis-ref. Settings are in \"Auto Chapters\".", wrap=0)
            dpg.add_listbox(items=[], tag="chap_list", num_items=2, width=-1, callback=self.on_pick_chapter)
            dpg.add_button(label="New chapter at cursor", tag="b_chap_new", width=-1, callback=lambda: self.new_chapter())
            with dpg.group(horizontal=True):
                dpg.add_button(label="Remove", tag="b_chap_del", callback=lambda: self.remove_chapter())
                dpg.add_button(label="Remove all", tag="b_chap_del_all", callback=lambda: self.remove_all_chapters())
            dpg.add_button(label="Compute flow for all chapters", tag="b_chap_flow", width=-1, callback=lambda: self.start_chapters_flow())
            dpg.add_text("", tag="chap_text", wrap=0)
        groups = {}
        for spec in PARAMS:
            grp = spec[0]
            groups.setdefault(grp, []).append(spec)

        OPEN_GROUPS = {"Enhancement", "Region", "Signal", "Keyframes"}
        for grp, specs in groups.items():
            tag_hdr = "phdr_" + grp.lower().replace(" ", "_").replace("(", "").replace(")", "")
            with dpg.collapsing_header(label=grp, default_open=(grp in OPEN_GROUPS), tag=tag_hdr):
                if grp == "Auto Chapters":
                    self.txt("muted", AUTO_HELP, wrap=0)
                for spec in specs:
                    grp_, attr, label, kind, lo, hi, fmt, stage = spec
                    val = getattr(self.a, attr)
                    cb = self.make_cb((attr, kind, stage))
                    tag = "p_" + attr
                    if kind == "float":
                        dpg.add_slider_float(label=label, tag=tag, default_value=float(val), min_value=lo, max_value=hi,
                                             format=fmt, callback=cb, width=170)
                    elif kind == "int":
                        dpg.add_slider_int(label=label, tag=tag, default_value=int(val), min_value=lo, max_value=hi,
                                           callback=cb, width=170)
                    elif kind == "bool":
                        dpg.add_checkbox(label=label, tag=tag, default_value=bool(val), callback=cb)
                    else:
                        dpg.add_combo(hi, label=label, tag=tag, default_value=str(val), callback=cb, width=170)
        dpg.add_separator()
        dpg.add_text("Save")
        dpg.add_input_text(tag="out_path", width=-1)
        dpg.add_button(label="Save funscript", tag="b_save", callback=lambda: self.save())
        self.refresh_diff()

    def build_inspector(self):
        self.txt("text", "", tag="in_win")
        self.txt("muted", "", tag="in_formula")
        for i in range(MAXC):
            with dpg.group(horizontal=True):
                dpg.add_text("", tag=f"ct{i}")
                dpg.add_progress_bar(tag=f"cb{i}", width=110, default_value=0.0)
        dpg.add_text("", tag="in_prev")
        dpg.add_text("", tag="in_thr")
        dpg.add_text("", tag="in_best")
        dpg.add_text("", tag="in_dec", color=(255, 255, 255))
        dpg.add_separator()
        dpg.add_text("Autocorrelation by candidate (s)")
        with dpg.plot(height=170, width=-1, no_title=True, no_menus=True, no_box_select=True):
            dpg.add_plot_axis(dpg.mvXAxis, tag="ac_x")
            with dpg.plot_axis(dpg.mvYAxis, tag="ac_y"):
                for i in range(MAXC):
                    dpg.add_line_series([0.0, 1.0], [0.0, 0.0], tag=f"ac{i}", show=False)
                    dpg.bind_item_theme(f"ac{i}", self.cand_themes[i % len(self.cand_themes)])
            dpg.set_axis_limits("ac_y", -1.0, 1.05)

    def build_timeline(self):
        names = ["ROI", "Vel.", "Integral", "Script", "Script vel."]
        show_raw = bool(self.cfg.get("show_raw", False))
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="raw integral (right axis)", tag="show_raw", default_value=show_raw,
                             callback=lambda s, v: self.on_show_raw(bool(v)))
            self.txt("muted", "Vel. and Integral scales follow only the visible range.")
        with dpg.subplots(5, 1, tag="subs", width=-1, height=-1, link_all_x=True, no_title=True,
                          row_ratios=[1.0, 1.0, 1.0, 1.4, 1.0]):
            for i, name in enumerate(names):
                with dpg.plot(tag=f"pl{i}", no_title=True, no_box_select=True):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                    # time ticks only on the last row; others gain height (x axis is shared)
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f"tx{i}", no_label=True, no_tick_labels=(i < 4))
                    # range_fit: vertical scale adjusts only to points in the visible time range.
                    # Without this, the integral (which has drift) squashes the signal of interest.
                    with dpg.plot_axis(dpg.mvYAxis, tag=f"ty{i}", label=name, auto_fit=(i in (1, 2)),
                                       range_fit=(i in (1, 2))):
                        pass
                    if i == 2:
                        dpg.add_plot_axis(dpg.mvYAxis2, tag="ty2b", label="raw", auto_fit=True, range_fit=True,
                                          opposite=True, show=show_raw)
                    dpg.add_drag_line(tag=f"ph{i}", default_value=0.0, vertical=True, show_label=False,
                                      color=list(self.pal["fg"]) + [200], callback=self.on_playhead)
        spec = [(0, "s_mag", "magnitude", (255, 160, 60)), (0, "s_per", "periodicity", (60, 220, 255)),
                (0, "s_score", "score", (80, 220, 80)), (0, "s_best", "best candidate", (170, 170, 170)),
                (1, "s_v", "velocity", (255, 220, 60)),
                (2, "s_p", "filtered", (255, 160, 60)),
                (3, "s_pos", "signal", (150, 150, 150)), (3, "s_fs", "funscript", "fg"),
                (4, "s_spd", "u/s", (80, 220, 80))]
        for row, tag, label, rgb in spec:
            dpg.add_line_series([0.0, 1.0], [0.0, 0.0], label=label, tag=tag, parent=f"ty{row}")
            self.series_rgb[tag] = (rgb, 1.5)
        dpg.add_line_series([0.0, 1.0], [0.0, 0.0], label="raw integral", tag="s_raw", parent="ty2b", show=show_raw)
        self.series_rgb["s_raw"] = ((90, 150, 255), 1.0)
        dpg.add_scatter_series([0.0], [0.0], label="keyframes", tag="s_kf", parent="ty3")
        dpg.set_axis_limits("ty0", 0.0, 1.1)
        dpg.set_axis_limits("ty3", -3.0, 103.0)

    def on_resize(self):
        pass

    def on_show_raw(self, on):
        dpg.configure_item("s_raw", show=on)
        dpg.configure_item("ty2b", show=on)

    # --------------------------------------------- overview timeline
    def build_overview(self):
        with dpg.group(horizontal=True):
            dpg.add_text("colour by")
            dpg.add_combo(list(self.OV_MODES), default_value=self.cfg.get("ov_mode", "velocity")
                          if self.cfg.get("ov_mode") in self.OV_MODES else "velocity", width=130,
                          tag="ov_mode", callback=lambda s, v: self.on_ov_mode())
            self.txt("muted", self.OV_MODES[dpg.get_value("ov_mode")], tag="ov_legend")
            help_item = self.txt("head", "(?)")
            with dpg.tooltip(help_item):
                dpg.add_text("velocity: how fast the funscript moves.\n"
                             "position: funscript value (0 to 100).\n"
                             "motion: how much the selected region moves.\n"
                             "periodicity: how rhythmic the motion is (what the region selection uses).\n"
                             "confidence: score of the selected region (periodicity and motion).\n"
                             "amplitude: stroke range in 2 s windows.\n"
                             "pace: strokes (one up and one down) per second in 4 s windows.\n"
                             "Red means 4 strokes per second or more. Shows where the action speeds up or slows down.\n"
                             "direction: motion axis angle. A colour change mid-video usually marks\n"
                             "a good chapter split point.\n"
                             "The white line at the bottom is the funscript outline.\n"
                             "Click or drag to seek. Click on the chapter band to switch chapters.\n"
                             "The light rectangle shows the visible range in the Signals window.")
        with dpg.drawlist(width=800, height=OV_H, tag="ov_dl"):
            dpg.add_draw_layer(tag="ov_static")
            dpg.add_draw_layer(tag="ov_dyn")
        self.txt("text", "", tag="ov_info")

    def mark_ov(self):
        self.ov_dirty = True

    def on_ov_mode(self):
        dpg.set_value("ov_legend", self.OV_MODES[dpg.get_value("ov_mode")])
        self.ov_dirty = True

    def ov_segments(self):
        """Per chapter: (start, end, data). Active chapter comes from the current result, others from the flow cache."""
        segs = []
        for i, c in enumerate(self.chapters):
            data = None
            if i == self.active and self.res is not None and not self.geometry_stale():
                data = self.ov_series_res(self.res, self.fps())
            else:
                data = self.ov_other.get(i)          # o ativo, enquanto carrega, mostra o que havia
            segs.append((c["start"], self.chapter_end(i), data))
        return segs

    def ov_total(self, segs):
        dur = self.src.dur if (self.src and self.src.dur < 1e8) else 0.0
        last = max([float(d["t"][-1]) for _, _, d in segs if d is not None] or [0.0])
        return max(dur, last, 1.0)

    def draw_overview(self):
        """Static part: chapters, colour intensity, funscript outline and time ruler."""
        w = self.ov_w
        P = "ov_static"
        dpg.delete_item(P, children_only=True)
        self.ov_cache = []
        dpg.draw_rectangle((0, 0), (w, OV_H), color=(0, 0, 0, 0), fill=(22, 22, 27, 255), parent=P)
        if self.src is None:
            return
        segs = self.ov_segments()
        D = self.ov_D = self.ov_total(segs)
        X = lambda t: min(max(t / D, 0.0), 1.0) * w
        mode = dpg.get_value("ov_mode")
        hb0, hb1 = OV_BAND + 3, OV_BAND + 33
        eb0, eb1 = hb1 + 3, hb1 + 31
        for i, (s, e, data) in enumerate(segs):
            e = min(e, D)
            x0, x1 = X(s), X(e)
            col = CH_COLORS[i % len(CH_COLORS)]
            act = i == self.active
            dpg.draw_rectangle((x0 + 1, 1), (max(x1 - 1, x0 + 2), OV_BAND), rounding=4, thickness=2 if act else 1,
                               color=(255, 255, 255, 255) if act else col + (255,),
                               fill=col + (215 if act else 130,), parent=P)
            c = self.chapters[i]
            label = (f"{i + 1}" + (f"  {self.ref_text(c.get('axis_ref'))}" if c.get("axis_ref") else "")
                     + ("  zoom" if c.get("zoom") else "") + ("  inverted" if c.get("invert") else ""))
            while label and 7 * len(label) + 10 > x1 - x0:
                label = label[:-1]
            if label:
                dpg.draw_text((x0 + 6, 4), label, color=(255, 255, 255, 255), size=13, parent=P)
            if data is None:
                dpg.draw_rectangle((x0, hb0), (x1, eb1), color=(0, 0, 0, 0), fill=(40, 40, 46, 255), parent=P)
                if x1 - x0 > 130:
                    dpg.draw_text((x0 + 8, hb0 + 12), "no flow computed", color=(150, 150, 156, 255), size=13, parent=P)
                continue
            t, fs, spd = data["t"], data["fs"], data["spd"]
            col_series = data["m"][mode]
            self.ov_cache.append((s, e, t, fs, spd, data["m"]))
            nb = max(1, int((x1 - x0) // 2))
            edges = np.linspace(s, e, nb + 1)
            idx = np.searchsorted(t, edges)
            for k in range(nb):
                i0 = int(idx[k])
                i1 = max(int(idx[k + 1]), i0 + 1)
                if i0 >= len(t):
                    break
                f, cs = fs[i0:i1], col_series[i0:i1]
                xa, xb = X(edges[k]), X(edges[k + 1])
                rgb = hue_rgb(float(cs.mean())) if mode == "direction" else heat_rgb(float(cs.mean()))
                dpg.draw_rectangle((xa, hb0), (xb, hb1), color=rgb + (255,), fill=rgb + (255,), parent=P)
                ya = eb1 - float(f.max()) / 100.0 * (eb1 - eb0)
                yb = eb1 - float(f.min()) / 100.0 * (eb1 - eb0)
                xm = (xa + xb) / 2.0
                dpg.draw_line((xm, ya), (xm, max(yb, ya + 1)), color=(205, 210, 220, 235), thickness=2, parent=P)
        step = nice_step(D, w)
        tk = 0.0
        while tk < D:
            x = X(tk)
            dpg.draw_line((x, OV_H - 13), (x, OV_H - 8), color=(150, 150, 158, 255), parent=P)
            dpg.draw_text((x + 3, OV_H - 15), f"{int(tk // 60)}:{int(tk % 60):02d}", color=(150, 150, 158, 255),
                          size=12, parent=P)
            tk += step

    def draw_overview_dyn(self):
        """Dynamic part updated each frame: visible signal window, mouse and cursor."""
        w, D = self.ov_w, self.ov_D
        if w <= 0 or D <= 0 or self.src is None:
            return
        try:
            lo, hi = dpg.get_axis_limits("tx0")
        except Exception:
            lo = hi = 0.0
        key = (round(self.t, 3), round(lo, 2), round(hi, 2), None if self.ov_hover is None else int(self.ov_hover * 10),
               w, D)
        if key == self.ov_dyn_key:
            return
        self.ov_dyn_key = key
        P = "ov_dyn"
        dpg.delete_item(P, children_only=True)
        X = lambda t: min(max(t / D, 0.0), 1.0) * w
        bottom = OV_H - 14
        if hi > lo:
            dpg.draw_rectangle((X(lo), 0), (max(X(hi), X(lo) + 2), bottom), color=(255, 255, 255, 150),
                               fill=(255, 255, 255, 24), parent=P)
        if self.ov_hover is not None:
            xh = X(self.ov_hover)
            dpg.draw_line((xh, 0), (xh, bottom), color=(255, 255, 255, 110), parent=P)
        x = X(self.t)
        dpg.draw_line((x, 0), (x, bottom), color=(255, 255, 255, 255), thickness=2, parent=P)
        dpg.draw_triangle((x - 5, 0), (x + 5, 0), (x, 7), color=(255, 255, 255, 255), fill=(255, 255, 255, 255),
                          parent=P)

    def ov_at(self, t):
        """(chapter, funscript, velocity, colour-mode value) at time t, or None where there is no data."""
        ci = self.chapter_at(t)
        mode = dpg.get_value("ov_mode")
        for s, e, tt, fs, spd, m in self.ov_cache:
            if s - 1e-6 <= t <= e + 1e-6 and len(tt):
                j = min(max(int(np.searchsorted(tt, t)), 0), len(tt) - 1)
                return ci, float(fs[j]), float(spd[j]), float(m[mode][j])
        return ci, None, None, None

    def ov_press(self, x, y):
        t = min(max(x / self.ov_w * self.ov_D, 0.0), self.ov_D)
        band = y < OV_BAND + 1
        ci = self.chapter_at(t)
        if ci != self.active:
            if self.busy():
                self.status = "wait for the computation to finish (or cancel) before switching chapters"
                self.ov_drag = None
                return
            self.activate(ci, jump=band)
            if not band:
                self.t = t
                self.dirty_view = True
        else:
            self.scrub_to(t)

    def scrub_to(self, t):
        self.playing = False
        self.set_time(t)
        if dpg.get_value("follow"):
            wd = dpg.get_value("follow_w")
            for i in range(5):
                dpg.set_axis_limits(f"tx{i}", self.t - wd * 0.5, self.t + wd * 0.5)

    def handle_overview(self):
        if self.src is None or not dpg.is_item_visible("ov_dl"):
            return
        now = time.time()
        avail = int(dpg.get_item_rect_size("w_ov")[0]) - 18
        if avail >= 120 and abs(avail - self.ov_w) >= 4 and now - self.ov_redraw_t > 0.12:
            dpg.configure_item("ov_dl", width=avail)
            self.ov_w = avail
            self.ov_dirty = True
        if self.ov_dirty and self.ov_w > 0 and now - self.ov_redraw_t > 0.05:
            self.ov_dirty = False
            self.ov_redraw_t = now
            self.draw_overview()
            self.ov_dyn_key = None
        if self.ov_w <= 0 or self.ov_D <= 0:
            return
        mx, my = dpg.get_mouse_pos(local=False)
        rx, ry = dpg.get_item_rect_min("ov_dl")
        x, y = mx - rx, my - ry
        hov = dpg.is_item_hovered("ov_dl") and 0 <= x <= self.ov_w and 0 <= y <= OV_H
        down = dpg.is_mouse_button_down(0)
        if down and self.ov_drag is None and hov:
            self.ov_drag = "band" if y < OV_BAND + 1 else "scrub"
            self.ov_press(x, y)
        elif not down:
            self.ov_drag = None
        elif self.ov_drag == "scrub":
            self.scrub_to(min(max(x / self.ov_w * self.ov_D, 0.0), self.ov_D))
        self.ov_hover = (min(max(x, 0), self.ov_w) / self.ov_w * self.ov_D) if hov else None
        tt = self.ov_hover if self.ov_hover is not None else self.t
        ci, v, s, mv = self.ov_at(tt)
        mode = dpg.get_value("ov_mode")
        extra = "" if v is None else (f"   funscript {v:.0f}   velocity {s:.0f} u/s"
                                      + ("" if mode in ("velocity", "position") else
                                         f"   {mode} {mv * 180:.0f}°" if mode == "direction" else f"   {mode} {mv * 100:.0f}%"))
        hover_band = hov and y < OV_BAND + 1
        if hover_band:
            ch = self.chapters[ci]
            ch_end = self.chapter_end(ci)
            ch_dur = ch_end - ch["start"]
            parts = [f"chapter {ci + 1}  |  {fmt_time(ch['start'])} – {fmt_time(ch_end)}  ({ch_dur:.1f} s)"]
            if ch.get("zoom"):
                parts.append(f"zoom {ch['zoom']}")
            if ch.get("region"):
                r2 = ch["region"]
                parts.append(f"region {r2}")
            why = ch.get("why")
            if why and why != "manual":
                parts.append(f"detected: {why}")
            elif why == "manual":
                parts.append("manual chapter")
            dpg.set_value("ov_info", "  |  ".join(parts))
        else:
            dpg.set_value("ov_info", f"{'mouse' if self.ov_hover is not None else 'cursor'} {fmt_time(tt)}   chapter {ci + 1}{extra}")
        self.draw_overview_dyn()

    OV_MODES = {
        "velocity": "blue = slow, red = at the limit (max speed)",
        "position": "blue = 0, red = 100",
        "motion": "blue = region still, red = lots of motion",
        "periodicity": "blue = irregular motion, red = very rhythmic",
        "confidence": "blue = low-confidence region, red = very confident",
        "amplitude": "blue = short strokes, red = full-range strokes",
        "pace": "strokes per second: blue = few, red = 4 or more",
        "direction": "colour = motion axis angle (same colour = same direction)",
    }

    def ov_series(self, t, fs, v, widx, dbg, kx, fps):
        """Per-frame series for the overview, each normalised to 0..1 (or a hue in direction mode)."""
        from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
        T = len(t)
        spd = np.abs(np.gradient(fs, t)) if T > 1 else np.zeros(T)
        av = np.abs(v)
        k3 = max(3, int(2.0 * fps))
        per = np.array([dbg[k]["chosen_info"]["period"] for k in widx], float)
        sco = np.array([dbg[k]["chosen_info"]["score"] for k in widx], float)
        ang = np.array([(np.arctan2(dbg[k]["axis"][1], dbg[k]["axis"][0]) % np.pi) / np.pi if "axis" in dbg[k] else 0.0
                        for k in widx], float)
        if len(kx):
            hist = np.zeros(T)
            ix = np.clip(np.round((np.asarray(kx) - t[0]) * fps).astype(int), 0, T - 1)
            np.add.at(hist, ix, 1.0)
            rate = uniform_filter1d(hist, size=2 * k3) * fps / 2.0       # strokes (2 keyframes) per second
        else:
            rate = np.zeros(T)
        rng = maximum_filter1d(fs, k3) - minimum_filter1d(fs, k3)
        vm = float(np.percentile(av, 99)) + 1e-9
        m = {"velocity": spd / max(float(self.a.max_speed), 1.0), "position": fs / 100.0, "motion": av / vm,
             "periodicity": per, "confidence": sco, "amplitude": rng / 100.0, "pace": rate / 4.0, "direction": ang}
        return {"t": t, "fs": fs, "spd": spd, "m": {k: np.clip(a_, 0.0, 1.0) for k, a_ in m.items()}}

    def ov_series_res(self, r, fps):
        key = id(r)
        if self.ov_res_key != key:
            self.ov_res_key = key
            self.ov_res_val = self.ov_series(np.asarray(r["t"], float), np.asarray(r["fs"], float), r["v"],
                                             r["widx"], r["dbg"], r["kx"], fps)
        return self.ov_res_val

    def ov_series_flow(self, b, fw):
        """Same as ov_series_res but for a non-active chapter, computed from its cached flow."""
        dx, dy, mag = fw
        dbg = []
        r = fl.compute(b, dx, dy, mag, dbg)
        v = r["v"]
        t = b.start + np.arange(len(v)) / b.fps
        pts = r["pts"]
        kx = np.array([x / 1000.0 for x, _ in pts])
        fs = np.interp(t, kx, [y for _, y in pts]) if pts else np.zeros(len(v))
        widx = np.zeros(len(v), int)
        for k, (s0, e0, _) in enumerate(r["windows"]):
            widx[s0:e0] = k
        return self.ov_series(t, fs, v, widx, dbg, kx, b.fps)

    def ov_param_key(self, base):
        """Parameters outside the chapter that affect the funscript shown in the overview."""
        return (tuple(getattr(base, k) for k in PRESET_ATTRS), base.axis_ref, base.axis, base.width, base.cell,
                base.fps, base.crop, bool(base.invert))

    def do_overview(self):
        """Funscript for non-active chapters, computed from whatever flow caches exist.

        Processes one chapter at a time, caches results (only redoes what changed), and aborts
        as soon as the user interacts, so it never delays the UI."""
        chs = [dict(c) for c in self.chapters]
        act = self.active
        base = copy.copy(self.a)
        pk = self.ov_param_key(base)
        for i in [k for k in self.ov_other if k >= len(chs)]:
            self.ov_other.pop(i, None)
        if len(chs) > 1:
            for i in range(len(chs)):
                if i == act:
                    continue
                if self.need_flow or self.need_compute or self.active != act or self.alive is False:
                    self.need_ov = True              # interrupted: resumes when idle again
                    break
                key = (pk, json.dumps({k: v for k, v in chs[i].items() if k != "why"}, sort_keys=True),
                       self.chapter_end(i))
                if self.ov_memo.get(i, (None,))[0] == key and i in self.ov_other:
                    continue
                b = fl.chapter_args(base, chs, i)
                try:
                    fw = fl.flow_for(b, cached_only=True)
                    if fw is not None:
                        self.ov_other[i] = self.ov_series_flow(b, fw)
                        self.ov_memo[i] = (key, True)
                    else:
                        self.ov_other.pop(i, None)
                except Exception:
                    pass
                time.sleep(0.01)                    # let the UI breathe between chapters
                self.ov_dirty = True
        self.ov_dirty = True

    # ---------------------------------------------------------- callbacks
    def on_file(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if path:
            self.open_path(path)

    def make_cb(self, user):
        # DearPyGui passes (sender, value, user_data) to callbacks with 3 args,
        # which overrides the default value. With 2 args it receives only (sender, value).
        def cb(sender, value):
            self.on_param(value, user)
        return cb

    def on_param(self, value, user):
        attr, kind, stage = user
        if kind == "int" or attr == "fps":
            value = int(value)
        elif kind == "float":
            value = float(value)
        setattr(self.a, attr, value)
        if attr in PARAM_BY_ATTR and attr in PRESET_ATTRS:
            self.refresh_diff()
        if stage == "c":
            if dpg.get_value("p_live"):
                self.request_compute()
            else:
                self.pending_apply = True
                dpg.configure_item("btn_apply", label="Apply changes *")

    def on_live(self, live):
        if live and self.pending_apply:
            self.apply_pending()

    def apply_pending(self):
        self.pending_apply = False
        dpg.configure_item("btn_apply", label="Apply changes")
        self.request_compute()

    # ------------------------------------------------------- presets
    def preset_names(self):
        return list(BUILTIN_PRESETS) + [n for n in self.user_presets if n not in BUILTIN_PRESETS]

    def refresh_presets(self, select=None):
        dpg.configure_item("p_preset", items=self.preset_names())
        if select is not None:
            dpg.set_value("p_preset", select)

    def on_preset(self, sender, name):
        """Loads the selected preset: starts from default values and applies the differences."""
        if name in BUILTIN_PRESETS:
            desc, vals = BUILTIN_PRESETS[name]
        elif name in self.user_presets:
            desc, vals = "Your preset.", self.user_presets[name]
        else:
            return
        full = dict(self.defaults)
        full.update(vals)
        for attr in PRESET_ATTRS:
            v = clean_param(attr, full[attr])
            if v is None:
                continue
            setattr(self.a, attr, v)
            dpg.set_value("p_" + attr, str(v) if PARAM_BY_ATTR[attr][3] == "combo" else v)
        dpg.set_value("preset_info", desc)
        dpg.set_value("preset_name", "" if name in BUILTIN_PRESETS else name)
        self.preset_ref = name
        self.refresh_diff()
        self.status = f"preset '{name}' loaded"
        self.request_compute()

    def preset_desc(self, name):
        if name in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[name][0]
        return "Your preset." if name in self.user_presets else ""

    def ref_values(self, name):
        vals = BUILTIN_PRESETS[name][1] if name in BUILTIN_PRESETS else self.user_presets.get(name, {})
        full = dict(self.defaults)
        full.update(vals)
        out = {}
        for k, v in full.items():
            v = clean_param(k, v)
            if v is not None:
                out[k] = v
        return out

    def refresh_diff(self):
        """Marks parameters that differ from the reference preset and lists which ones."""
        if not dpg.does_item_exist("preset_diff"):
            return
        ref = self.preset_ref
        vals = self.ref_values(ref)
        fmt = lambda v: f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        changed = []
        for attr in PRESET_ATTRS:
            cur, r = getattr(self.a, attr), vals.get(attr)
            if r is None:
                continue
            kind = PARAM_BY_ATTR[attr][3]
            diff = (abs(float(cur) - float(r)) > 1e-9) if kind in ("float", "int") else str(cur) != str(r)
            tag, label = "p_" + attr, PARAM_BY_ATTR[attr][2]
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, self.th_changed if diff else 0)
                dpg.configure_item(tag, label=label + (" *" if diff else ""))
            if diff:
                changed.append(f"{label}: {fmt(r)} -> {fmt(cur)}")
        dpg.set_value("preset_info", self.preset_desc(ref))
        if changed:
            shown = "; ".join(changed[:6]) + (f"; and {len(changed) - 6} more" if len(changed) > 6 else "")
            dpg.set_value("preset_diff", f"{len(changed)} changed from '{ref}' (highlighted in orange):\n{shown}")
        else:
            dpg.set_value("preset_diff", f"Identical to '{ref}'.")
        dpg.configure_item("b_pre_reset", show=bool(changed), label=f"Reset to preset '{ref}'")

    def reset_to_preset(self):
        self.on_preset(None, self.preset_ref)

    def write_user_presets(self):
        path = presets_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.user_presets, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def save_preset(self):
        name = dpg.get_value("preset_name").strip()
        if not name:
            self.status = "enter a name for the preset"
            return
        if name in BUILTIN_PRESETS:
            self.status = f"'{name}' is a built-in preset, choose a different name"
            return
        self.user_presets[name] = {attr: getattr(self.a, attr) for attr in PRESET_ATTRS}
        try:
            self.write_user_presets()
        except OSError as e:
            self.status = f"error saving preset: {e}"
            return
        self.refresh_presets(name)
        self.preset_ref = name
        self.refresh_diff()
        self.status = f"preset '{name}' saved"

    def delete_preset(self):
        name = dpg.get_value("p_preset")
        if name in BUILTIN_PRESETS:
            self.status = "built-in presets cannot be deleted"
            return
        if name not in self.user_presets:
            self.status = "select one of your presets first"
            return
        del self.user_presets[name]
        try:
            self.write_user_presets()
        except OSError as e:
            self.status = f"error saving: {e}"
            return
        self.refresh_presets("")
        if self.preset_ref == name:
            self.preset_ref = "Default"
            self.refresh_diff()
        dpg.set_value("preset_info", "")
        self.status = f"preset '{name}' deleted"

    def on_crop(self, sender, value):
        self.a.crop = CROPS[value]
        self.dirty_view = True

    def on_speed(self, sender, value):
        self.speed = float(value.rstrip("x"))

    def on_playhead(self, sender):
        self.drag_until = time.time() + 0.25
        self.set_time(dpg.get_value(sender))

    def toggle_play(self):
        self.playing = not self.playing
        self.last_tick = time.perf_counter()

    def frame_step(self, n):
        """Left and right arrow keys: one frame back or forward (signals follow the cursor)."""
        if self.res is None:
            return
        self.scrub_to(self.t + n / self.fps())

    def set_time(self, t):
        r = self.res
        if r is not None:
            t = min(max(t, r["t"][0]), r["t"][-1])
        self.t = t
        self.dirty_view = True

    # -------------------------------------------------- region and zoom (mouse)
    def set_mode(self, mode):
        if mode != "none" and self.src is not None and not self.bulk:
            ci = self.chapter_at(self.t)
            if ci != self.active and not self.busy():
                self.activate(ci, jump=False)
        self.mode = mode
        names = {"none": "none", "region": "region", "zoom": "zoom", "axis": "direction"}
        dpg.set_value("sel_mode", names[mode])
        self.axis_drag = None
        self.rect_reset = mode in ("region", "zoom")
        self.dirty_view = True

    def place_rect(self):
        """Places the rectangle at the current selection (or centred), in screen pixels (origin top-left)."""
        ox, oy, fw, fh = self.view
        spec = self.a.region if self.mode == "region" else self.a.zoom
        f = [0.2, 0.2, 0.8, 0.8]
        if spec:
            f = [float(v) for v in spec.split(":")]
        self.sel = [ox + f[0] * fw, oy + f[1] * fh, ox + f[2] * fw, oy + f[3] * fh]
        self.dirty_view = True

    def rect_fractions(self):
        x0, y0, x1, y1 = self.sel
        ox, oy, fw, fh = self.view
        fx0, fx1 = (min(x0, x1) - ox) / fw, (max(x0, x1) - ox) / fw
        fy0, fy1 = (min(y0, y1) - oy) / fh, (max(y0, y1) - oy) / fh
        c = lambda v: min(max(v, 0.0), 1.0)
        return c(fx0), c(fy0), c(fx1), c(fy1)

    def mouse_px(self):
        """Mouse position over the video in canvas pixels, or None if outside."""
        if not dpg.is_item_hovered("vplot"):
            return None
        px, py = dpg.get_plot_mouse_pos()
        if not (0 <= px <= CW and 0 <= py <= CH):
            return None
        return px, CH - py

    def handle_mouse(self):
        """Draw, move or resize the rectangle with the mouse (dpg's drag_rect was not visible in this plot)."""
        if self.mode == "axis":
            down = dpg.is_mouse_button_down(0)
            m = self.mouse_px()
            if down and m is not None:
                if self.axis_drag is None:
                    self.axis_drag = [m[0], m[1], m[0], m[1]]
                else:
                    self.axis_drag[2:] = list(m)
                self.dirty_view = True
            elif not down and self.axis_drag is not None:
                x0, y0, x1, y1 = self.axis_drag
                self.axis_drag = None
                n = float(np.hypot(x1 - x0, y1 - y0))
                if n >= 14:
                    self.set_axis_ref_from_screen((x1 - x0) / n, (y1 - y0) / n)
                    self.set_mode("none")
                self.dirty_view = True
            return
        if self.mode == "none" or self.sel is None:
            self.drag = None
            return
        down = dpg.is_mouse_button_down(0)
        m = self.mouse_px()
        if down and self.drag is None and m is not None:
            x, y = m
            x0, y0, x1, y1 = self.sel
            lx, hx, ly, hy = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            e = 9
            near_x = [("l", abs(x - lx)), ("r", abs(x - hx))]
            near_y = [("t", abs(y - ly)), ("b", abs(y - hy))]
            inside = lx - e <= x <= hx + e and ly - e <= y <= hy + e
            ex = min(near_x, key=lambda q: q[1]) if inside else None
            ey = min(near_y, key=lambda q: q[1]) if inside else None
            sx = ex[0] if ex and ex[1] <= e else None
            sy = ey[0] if ey and ey[1] <= e else None
            if sx or sy:
                self.drag = ("edge", sx, sy)
            elif inside:
                self.drag = ("move", x - lx, y - ly, hx - lx, hy - ly)
            else:
                self.drag = ("new", x, y)
        elif not down:
            self.drag = None
        if self.drag is not None and m is not None:
            x, y = m
            ox, oy, fw, fh = self.view
            x = min(max(x, ox), ox + fw)
            y = min(max(y, oy), oy + fh)
            d = self.drag
            lx, hx = min(self.sel[0], self.sel[2]), max(self.sel[0], self.sel[2])
            ly, hy = min(self.sel[1], self.sel[3]), max(self.sel[1], self.sel[3])
            if d[0] == "new":
                lx, hx, ly, hy = min(d[1], x), max(d[1], x), min(d[2], y), max(d[2], y)
            elif d[0] == "move":
                w, h = d[3], d[4]
                lx = min(max(x - d[1], ox), ox + fw - w)
                ly = min(max(y - d[2], oy), oy + fh - h)
                hx, hy = lx + w, ly + h
            else:
                if d[1] == "l": lx = min(x, hx - 4)
                if d[1] == "r": hx = max(x, lx + 4)
                if d[2] == "t": ly = min(y, hy - 4)
                if d[2] == "b": hy = max(y, ly + 4)
            self.sel = [lx, ly, hx, hy]
            self.dirty_view = True

    def draw_sel(self, canvas):
        if self.mode == "axis":
            fl.put(canvas, "drag toward 100", (10, CH - 10), 0.5, (255, 220, 0))
            if self.axis_drag is not None:
                x0, y0, x1, y1 = [int(v) for v in self.axis_drag]
                cv2.arrowedLine(canvas, (x0, y0), (x1, y1), (255, 220, 0), 3, tipLength=0.18)
                fl.put(canvas, "100", (x1 + 6, y1 - 6), 0.5, (255, 220, 0))
            return
        if self.mode == "none" or self.sel is None:
            return
        x0, y0, x1, y1 = [int(round(v)) for v in self.sel]
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        col = (0, 200, 255) if self.mode == "region" else (255, 160, 0)
        ov = canvas.copy()
        cv2.rectangle(ov, (x0, y0), (x1, y1), col, -1)
        canvas[:] = cv2.addWeighted(canvas, 0.78, ov, 0.22, 0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2)
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1),
                       (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)):
            cv2.rectangle(canvas, (cx - 4, cy - 4), (cx + 4, cy + 4), (255, 255, 255), -1)
            cv2.rectangle(canvas, (cx - 4, cy - 4), (cx + 4, cy + 4), col, 1)

    def apply_selection(self):
        if self.mode in ("none", "axis") or (self.res is None and self.mode == "region"):
            return
        x0, y0, x1, y1 = self.rect_fractions()
        if x1 - x0 < 0.03 or y1 - y0 < 0.03:
            self.status = "selection too small"
            return
        spec = f"{x0:.4f}:{y0:.4f}:{x1:.4f}:{y1:.4f}"
        if self.mode == "region":
            if self.geometry_stale():
                self.status = "recalculate flow before choosing the region (crop or zoom changed)"
                return
            self.a.region = spec
            self.request_compute()
        else:
            self.a.zoom = spec               # preview only; flow recalculates on button press
            self.a.region = None
        self.update_sel_text()
        self.sync_chapter()
        self.set_mode("none")

    def clear_selection(self):
        if self.mode == "axis":                  # clearing direction reverts to the settings default
            self.chapters[self.active]["axis_ref"] = None
            dpg.set_value("chap_axmode", AXMODES[0])
            self.set_mode("none")
            self.chapter_changed()
            return
        if self.mode == "region" or (self.mode == "none" and self.a.region and not self.a.zoom):
            self.a.region = None
            self.request_compute()
        else:
            self.a.zoom = None
            self.a.region = None
            self.dirty_view = True
        self.update_sel_text()
        self.sync_chapter()
        self.set_mode("none")

    def update_warning(self):
        rs = self.stale_reasons()
        no_flow = self.sig is None and self.src is not None and not self.busy()
        key = (tuple(rs), no_flow)
        if key == getattr(self, "warn_key", None):
            return
        self.warn_key = key
        if no_flow:
            self.warn_text = ("Warning: this video has no computed flow yet. Set the crop and zoom while watching "
                              "the video, then click Recalculate flow when ready.")
            dpg.bind_item_theme("btn_flow", self.th_stale)
        elif rs:
            geo = self.geometry_stale()
            msg = ("Warning: you changed " + ", ".join(rs) + ". The plot and signals still use the old flow; "
                   "click Recalculate flow to update them.")
            if geo:
                msg += " Meanwhile the video shows a preview with the new crop."
            self.warn_text = msg
            dpg.bind_item_theme("btn_flow", self.th_stale)
        else:
            self.warn_text = ""
            dpg.bind_item_theme("btn_flow", self.th_ok)

    def update_sel_text(self):
        dpg.set_value("sel_text", f"region: {'fixed' if self.a.region else 'automatic'}   "
                                  f"zoom: {('active' if not self.geometry_stale() else 'set, needs recalculation') if self.a.zoom else 'none'}")

    def save(self):
        path = dpg.get_value("out_path")
        if len(self.chapters) > 1:
            self.sync_chapter()
            missing = [str(i + 1) for i in range(len(self.chapters))
                     if not fl.flow_available(fl.chapter_args(self.a, self.chapters, i))]
            if missing:
                self.status = "missing flow for chapters " + ", ".join(missing) + ': use "Compute flow for all chapters"'
                return
            try:
                pts = fl.process_chapters(copy.copy(self.a), self.chapters)
                n = fl.write_funscript(path, pts)
                self.status = f"saved: {n} actions in {path} ({len(self.chapters)} chapters)"
                if self.folder_items:
                    self.open_path_refresh()
            except (OSError, Exception) as e:
                self.status = f"error saving: {e}"
            return
        if self.res is None:
            return
        try:
            n = fl.write_funscript(path, self.res["pts"])
            self.status = f"saved: {n} actions in {path}"
            if self.folder_items:
                self.open_path_refresh()
        except OSError as e:
            self.status = f"error saving: {e}"

    def on_export_cfg(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            data = {}
            for spec in PARAMS:
                _, attr, _, kind, _, _, _, _ = spec
                val = getattr(self.a, attr, None)
                if val is not None:
                    data[attr] = val
            with open(path, "w") as f:
                import json as _json
                _json.dump(data, f, indent=2)
            self.status = f"settings exported: {path}"
        except OSError as e:
            self.status = f"export error: {e}"

    def on_import_cfg(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path or not os.path.exists(path):
            return
        try:
            import json as _json
            with open(path) as f:
                data = _json.load(f)
            for spec in PARAMS:
                _, attr, _, kind, lo, hi, _, stage = spec
                if attr in data:
                    val = data[attr]
                    setattr(self.a, attr, val)
                    tag = "p_" + attr
                    if dpg.does_item_exist(tag):
                        if kind == "bool":
                            dpg.set_value(tag, bool(val))
                        elif kind == "float":
                            dpg.set_value(tag, float(val))
                        elif kind == "int":
                            dpg.set_value(tag, int(val))
                        else:
                            dpg.set_value(tag, str(val))
            self.need_compute = True
            self.last_change = time.time()
            self.status = f"settings imported: {path}"
        except (OSError, Exception) as e:
            self.status = f"import error: {e}"

    def open_path_refresh(self):
        """Re-reads which videos already have a script (preserves marks)."""
        marked = {v for v in self.rows if self.is_marked(v)}
        self.fill_folder(list(self.rows), marked)

    # ------------------------------------------------------------ chapters
    def default_chapters(self):
        return [{"start": 0.0, "zoom": None, "region": None, "invert": False, "axis_ref": None, "why": None}]

    def eff_invert(self):
        """Effective invert for the active chapter: global invert XOR chapter invert."""
        return bool(self.a.invert) != bool(self.chapters[self.active].get("invert"))

    def on_chap_invert(self, sender, value):
        if not self.a.video:
            return
        self.chapters[self.active]["invert"] = bool(value)
        self.write_chapters_file()
        self.refresh_chapters()
        self.request_compute()

    @staticmethod
    def ref_text(ref):
        """Chapter axis-ref in compact form: an angle (90°) or a name string."""
        try:
            return f"{int(round(float(ref)))}°"
        except (TypeError, ValueError):
            return str(ref)

    def apply_chap_axis(self):
        """Applies the selected direction from the list and angle slider. Only applied here, not while dragging."""
        if not self.a.video:
            return
        value = dpg.get_value("chap_axmode")
        c = self.chapters[self.active]
        if value == AXMODES[0]:
            c["axis_ref"] = None
        elif value == "angle":
            c["axis_ref"] = str(int(dpg.get_value("chap_ang")))
        else:
            c["axis_ref"] = value
        self.chapter_changed()

    def chapter_changed(self):
        self.write_chapters_file()
        self.refresh_chapters()
        self.request_compute()

    def set_axis_ref_from_screen(self, ux, uy):
        """Sets the chapter axis-ref from a screen direction where the user wants 100 to point.
        'invert' swaps the direction afterwards, so the effective invert is undone here."""
        s = -1.0 if self.eff_invert() else 1.0
        deg = int(round(np.degrees(np.arctan2(s * uy, s * ux)))) % 360
        self.chapters[self.active]["axis_ref"] = str(deg)
        dpg.set_value("chap_ang", deg)
        dpg.set_value("chap_axmode", "angle")
        self.chapter_changed()

    def chapter_end(self, i):
        if i + 1 < len(self.chapters):
            return self.chapters[i + 1]["start"]
        return self.src.dur if self.src else 1e9

    def chapter_at(self, t):
        k = 0
        for i, c in enumerate(self.chapters):
            if c["start"] <= t + 1e-6:
                k = i
        return k

    def apply_chapter_args(self):
        """Applies the active chapter's time offset, zoom and region to the namespace."""
        a, c = self.a, self.chapters[self.active]
        a.zoom, a.region = c.get("zoom"), c.get("region")
        if len(self.chapters) > 1:
            a.start = c["start"]
            a.duration = (self.chapter_end(self.active) - c["start"]) if self.active + 1 < len(self.chapters) else 0.0
            a.cache_tag = f".ch{self.active}"
        else:
            a.start, a.duration, a.cache_tag = 0.0, 0.0, ""

    def sync_chapter(self):
        """Saves the current zoom and region into the active chapter and writes the chapters file."""
        if not self.a.video:
            return
        c = self.chapters[self.active]
        c["zoom"], c["region"] = self.a.zoom, self.a.region
        self.write_chapters_file()
        self.refresh_chapters()

    def write_chapters_file(self, auto=False):
        """Writes the chapters file next to the video. auto marks a file from detection: the batch can
        redo it, but any manual edit removes that mark."""
        path = fl.chapters_file(self.a.video)
        trivial = (len(self.chapters) == 1 and not self.chapters[0]["zoom"] and not self.chapters[0]["region"]
                   and not self.chapters[0].get("invert") and not self.chapters[0].get("axis_ref"))
        try:
            if trivial:
                if os.path.exists(path):
                    os.remove(path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"auto": bool(auto), "chapters": self.chapters}, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def refresh_chapters(self):
        if not dpg.does_item_exist("chap_list"):
            return
        items = []
        for i, c in enumerate(self.chapters):
            end = self.chapter_end(i)
            b = fl.chapter_args(self.a, self.chapters, i) if self.a.video else None
            ok = bool(b) and fl.flow_available(b)
            tags = ("100 -> " + self.ref_text(c["axis_ref"]) + "  " if c.get("axis_ref") else "") + ("zoom" if c["zoom"] else "no zoom") + (" + region" if c["region"] else "") + (" + inverted" if c.get("invert") else "") + ("  [flow ok]" if ok else "")
            items.append(f"{i + 1}. {fmt_time(c['start'])} a {fmt_time(end)}  {tags}")
        self.chap_items = items
        dpg.configure_item("chap_list", items=items, num_items=min(max(len(items), 2), 6))
        dpg.set_value("chap_list", items[self.active])
        dpg.set_value("chap_inv", bool(self.chapters[self.active].get("invert")))
        ar = self.chapters[self.active].get("axis_ref")
        try:
            float(ar)
            dpg.set_value("chap_axmode", "angle")
            dpg.set_value("chap_ang", int(round(float(ar))) % 360)
        except (TypeError, ValueError):
            dpg.set_value("chap_axmode", ar if ar in AXMODES else AXMODES[0])
        n = len(self.chapters)
        dpg.set_value("chap_text", "single chapter (whole video)" if n == 1 else
                      f"{n} chapters. Zoom or region applies to the chapter where the cursor is.")
        self.ov_dirty = True            # overview shows chapters
        self.need_ov = n > 1

    def reset_flow_state(self, request=True):
        """Drops the current flow and results. Reads the cache if it exists, otherwise shows a warning (nothing computes automatically)."""
        self.res, self.flow, self.pending, self.error = None, None, None, None
        self.pending_flow = None
        self.worker_flow = None
        self.sig = None
        self.worker_sig = None
        self.cur_win = -1
        self.warn_key = None
        if request and (self.auto_flow or self.cache_ok()):
            self.request_flow()
        else:
            self.status = "no flow computed"
        self.dirty_view = True

    def activate(self, i, jump=True):
        if self.busy():
            self.status = "wait for the computation to finish (or cancel) before switching chapters"
            return False
        if self.res is not None and not self.geometry_stale():
            try:                     # the outgoing chapter is shown in the overview without waiting for recalculation
                self.ov_other[self.active] = self.ov_series_res(self.res, self.fps())
                self.ov_memo.pop(self.active, None)
            except Exception:
                pass
        self.active = i
        self.apply_chapter_args()
        self.set_mode("none")
        self.update_sel_text()
        if jump:
            self.set_time(self.chapters[i]["start"])
        self.reset_flow_state()
        self.refresh_chapters()
        return True

    def on_pick_chapter(self, sender, value):
        try:
            i = int(value.split(".")[0]) - 1
        except ValueError:
            return
        if i != self.active:
            if not self.activate(i):
                dpg.set_value("chap_list", self.chap_items[self.active])

    def new_chapter(self):
        t = self.t
        j = self.chapter_at(t)
        if t - self.chapters[j]["start"] < 1.0 or self.chapter_end(j) - t < 1.0:
            self.status = "place the cursor at least 1 s from the chapter edges to split it"
            return
        if self.busy():
            self.status = "wait for the computation to finish (or cancel) before creating a chapter"
            return
        self.sync_chapter()
        self.chapters.insert(j + 1, {"start": round(float(t), 2), "zoom": self.chapters[j]["zoom"], "region": None, "invert": False,
                              "axis_ref": None, "why": "manual"})
        self.ov_other, self.ov_memo = {}, {}
        self.write_chapters_file()
        self.activate(j + 1, jump=False)

    def remove_all_chapters(self):
        if len(self.chapters) == 1:
            self.status = "already one chapter"
            return
        if self.busy():
            self.status = "wait for the computation to finish (or cancel) before removing"
            return
        self.chapters = self.default_chapters()
        self.ov_other, self.ov_memo = {}, {}
        self.write_chapters_file()
        self.activate(0, jump=False)
        self.status = "all chapters removed"

    def remove_chapter(self):
        if len(self.chapters) == 1:
            self.status = "there is only one chapter"
            return
        if self.busy():
            self.status = "wait for the computation to finish (or cancel) before removing"
            return
        if self.active == 0:
            del self.chapters[1]          # the first remains and absorbs the next
            self.status = "chapter 2 merged into chapter 1"
        else:
            del self.chapters[self.active]  # o anterior passa a ocupar o seu tempo
        self.ov_other, self.ov_memo = {}, {}
        self.write_chapters_file()
        self.activate(max(self.active - 1, 0), jump=False)

    def suggest_chapters(self):
        """Detects chapters in the whole video (no zoom). If manual work exists, asks to click again."""
        if self.src is None or self.bulk or self.busy():
            return
        work = len(self.chapters) > 1 or any(c.get("zoom") or c.get("region") or c.get("invert") or c.get("axis_ref")
                                              for c in self.chapters)
        if work and time.time() - self.suggest_armed > 6.0:
            self.suggest_armed = time.time()
            self.status = "chapters or settings already exist: click again within 6 s to replace them"
            dpg.configure_item("btn_suggest", label="Click again to replace current chapters")
            return
        self.suggest_armed = 0.0
        dpg.configure_item("btn_suggest", label="Suggest chapters (automatic)")
        base = copy.copy(self.a)
        self.bulk = True
        self.cancel_flag = False
        threading.Thread(target=self.suggest_run, args=(base,), daemon=True).start()

    def suggest_run(self, base):
        try:
            self.start_progress("Computing full video flow for chapter detection")
            fa = fl._full_args(base)
            fa.zoom, fa.region = None, None
            dx, dy, mag = fl.load_or_extract(fa, self.on_progress)
            self.status = "Detecting chapters..."
            chs = fl.detect_chapters(dx, dy, mag, fa.fps, fa)
            self.suggested = chs
            self.status = f"{len(chs)} chapter(s) detected"
        except fl.Cancelled:
            self.status = "detection cancelled"
        except (Exception, SystemExit) as e:
            self.status = f"detection error: {e}"
        finally:
            self.prog_prefix = ""
            self.bulk = False
            self.cancel_flag = False
            self.progress = 0.0

    def apply_suggested(self):
        chs, self.suggested = self.suggested, None
        self.chapters = chs
        self.active = 0
        self.ov_other, self.ov_memo = {}, {}
        self.write_chapters_file(auto=True)
        self.apply_chapter_args()
        self.set_mode("none")
        self.update_sel_text()
        self.set_time(0.0)
        self.reset_flow_state()
        self.refresh_chapters()

    def start_chapters_flow(self):
        """Computes the flow for all chapters in sequence (each cached separately)."""
        if self.bulk or self.busy() or self.src is None:
            return
        self.sync_chapter()
        chs = [dict(c) for c in self.chapters]
        base = copy.copy(self.a)
        self.bulk = True
        self.cancel_flag = False
        threading.Thread(target=self.chapters_run, args=(chs, base), daemon=True).start()

    def chapters_run(self, chs, base):
        cancelled, failed = False, []
        try:
            for i in range(len(chs)):
                if self.cancel_flag:
                    cancelled = True
                    break
                self.start_progress(f"Chapter {i + 1}/{len(chs)}")
                b = fl.chapter_args(base, chs, i)
                try:
                    fl.flow_for(b, self.on_progress)
                except fl.Cancelled:
                    cancelled = True
                    break
                except (Exception, SystemExit) as e:
                    failed.append(f"{i + 1}: {e}")
        finally:
            self.bulk = False
            self.cancel_flag = False
            self.progress = 0.0
            self.prog_prefix = ""
            self.status = "chapters: " + ("cancelled" if cancelled else (f"{len(failed)} failed" if failed else "all flows ready"))
            self.chap_done = True

    # ------------------------------------------------- folder and batch
    def build_folder_panel(self):
        self.txt("head", "Folder Videos")
        self.txt("muted", "click name to open  ·  checkbox to include in batch  ·  batch controls in Control panel",
                 wrap=0)
        with dpg.child_window(tag="vid_rows", height=-1, border=False):
            pass

    def fill_folder(self, vids, marked=None):
        if dpg.does_item_exist("vid_rows"):
            dpg.delete_item("vid_rows", children_only=True)
        self.folder_items = {}
        self.rows = []
        self.vid_progress = [(0.0, "")] * len(vids)
        for i, v in enumerate(vids):
            stem = os.path.splitext(v)[0]
            has_script = os.path.exists(stem + ".funscript")
            has_cache = os.path.exists(stem + ".funscript.flowcache.npz")
            try:
                size_b = os.path.getsize(v)
                size_str = (f"{size_b/1e9:.1f} GB" if size_b >= 1e9 else
                            f"{size_b/1e6:.0f} MB" if size_b >= 1e6 else
                            f"{size_b/1e3:.0f} KB")
            except OSError:
                size_str = ""
            label = os.path.basename(v)
            self.folder_items[label] = v
            with dpg.group(parent="vid_rows"):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(tag=f"vc_{i}", default_value=(v in marked) if marked is not None else False,
                                     callback=lambda: self.update_bulk_text())
                    dpg.add_selectable(label=label, tag=f"vs_{i}", callback=self.on_pick_row, user_data=v,
                                       default_value=(v == self.a.video))
                    dpg.add_text(size_str, color=self.pal["muted"])
                    dpg.add_text("—", tag=f"vd_{i}", color=self.pal["muted"])
                    if has_script:
                        dpg.add_text("[script]", color=(80, 200, 80))
                    if has_cache:
                        dpg.add_text("[cache]", color=(80, 160, 220))
                dpg.add_progress_bar(tag=f"vp_{i}", default_value=0.0, width=-1, show=False, overlay="")
            self.rows.append(v)
        if dpg.does_item_exist("batch_ctrl"):
            dpg.configure_item("batch_ctrl", show=bool(vids))
        self.update_bulk_text()
        if vids:
            threading.Thread(target=self._probe_durations, args=(list(vids),), daemon=True).start()

    def _probe_durations(self, vids):
        for i, v in enumerate(vids):
            try:
                _, _, dur = fl.probe(v)
                m, s = int(dur) // 60, int(dur) % 60
                h = m // 60
                dur_str = f"{h}:{m % 60:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            except Exception:
                dur_str = "?"
            if dpg.does_item_exist(f"vd_{i}"):
                try:
                    dpg.set_value(f"vd_{i}", dur_str)
                except Exception:
                    pass

    def build_status_panel(self):
        self.txt("head", "Processing Status")
        dpg.add_text("", tag="st_mode", color=self.pal["muted"])
        dpg.add_progress_bar(tag="st_prog", width=-1, default_value=0.0, overlay="")
        dpg.add_separator()
        dpg.add_input_text(tag="st_detail", multiline=True, readonly=True, width=-1, height=-1)

    def tick_status(self):
        """Called from tick(): refresh the Status window contents."""
        if not dpg.does_item_exist("st_mode"):
            return
        if self.flowing:
            dpg.set_value("st_mode", "Computing optical flow...")
            dpg.set_value("st_prog", self.progress)
            dpg.configure_item("st_prog", overlay=f"{self.progress*100:.0f}%")
            dpg.set_value("st_detail", self.status)
        elif self.bulk:
            dpg.set_value("st_mode", self.bulk_msg or "Batch processing...")
            dpg.set_value("st_prog", self.progress)
            dpg.configure_item("st_prog", overlay=f"{self.progress*100:.0f}%")
            lines = []
            for ri, (frac, txt) in enumerate(self.vid_progress):
                if ri < len(self.rows) and (frac > 0 or txt):
                    name = os.path.basename(self.rows[ri])
                    bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
                    lines.append(f"{name}\n  [{bar}] {frac*100:.0f}%  {txt}")
            dpg.set_value("st_detail", "\n".join(lines) if lines else "")
        else:
            dpg.set_value("st_mode", "Idle")
            dpg.set_value("st_prog", 0.0)
            dpg.configure_item("st_prog", overlay="")
            # keep last status text visible until next batch starts
            if not (self.flowing or self.bulk):
                dpg.set_value("st_detail", self.status)

    def is_marked(self, v):
        return v in self.rows and dpg.get_value(f"vc_{self.rows.index(v)}")

    def marked_videos(self):
        return [v for i, v in enumerate(self.rows) if dpg.get_value(f"vc_{i}")]

    def mark_all(self, on):
        for i in range(len(self.rows)):
            dpg.set_value(f"vc_{i}", on)
        self.update_bulk_text()

    def mark_missing(self):
        for i, v in enumerate(self.rows):
            dpg.set_value(f"vc_{i}", not os.path.exists(os.path.splitext(v)[0] + ".funscript"))
        self.update_bulk_text()

    def update_bulk_text(self):
        if self.bulk:
            return
        n = len(self.marked_videos())
        dpg.set_value("bulk_text", f"{n} selected")
        dpg.configure_item("btn_bulk", label=f"Process selected ({n})")
        dpg.configure_item("btn_bulk_apply", label=f"Apply settings, no flow recalculation ({n})")

    def on_pick_row(self, sender, value, user_data):
        for i in range(len(self.rows)):
            dpg.set_value(f"vs_{i}", self.rows[i] == user_data)
        if self.bulk:
            self.status = "computing: wait or cancel (Esc) before opening another video"
        else:
            self.load_video(user_data)      # also the currently open video: useful to reload, e.g. in a single-video folder

    def start_bulk(self, cache_only=False):
        vids = self.marked_videos()
        if self.bulk or self.busy() or not vids:
            if not vids:
                dpg.set_value("bulk_text", "mark at least one video")
            return
        over, same, chap = dpg.get_value("b_over"), dpg.get_value("b_same"), dpg.get_value("b_chap")
        base = copy.copy(self.a)
        self.bulk = True
        self.cancel_flag = False
        dpg.set_value("bulk_text", f"processing {len(vids)} video(s)...")
        self.show_window("w_status", True)
        if cache_only:
            over = True                      # goal is to regenerate scripts with new settings
        threading.Thread(target=self.bulk_run, args=(vids, base, over, same, chap, cache_only), daemon=True).start()

    def bulk_run(self, vids, base, over, same, chap, cache_only=False):
        """Generates scripts with current settings, several videos in parallel if requested.

        Each video uses its cache if it exists and saves it when done, so an interrupted batch
        resumes from where it stopped (videos that already have a script are skipped unless redo is set)."""
        skipped, items = [], []
        res = []
        done_n = 0
        t0 = time.time()
        try:
            for v in vids:
                out = os.path.splitext(v)[0] + ".funscript"
                if os.path.exists(out) and not over:
                    skipped.append(v)
                    continue
                b = copy.copy(base)
                b.video, b.output = v, out
                b.plot, b.debug = None, None
                b.start, b.duration, b.cache_tag = 0.0, 0.0, ""      # active chapter does not apply in batch
                if not same:
                    b.region, b.zoom = None, None
                b.chapters = "auto" if chap else "none"
                b.cache_only = cache_only          # only uses cached flow, never computes it
                items.append(b)
            workers = max(1, int(getattr(base, "folder_jobs", 1) or 1))
            frames = [0] * len(items)          # frames processed per video
            totals = [0] * len(items)
            ended = [False] * len(items)
            running = set()
            vid_t0 = {}          # batch index → time when first frame reported for that video
            # probe durations upfront so ETA for not-started videos uses real lengths, not averages
            est_frames = []
            for b in items:
                try:
                    _, _, dur = fl.probe(b.video)
                    est_frames.append(max(1, int(dur * base.fps)))
                except Exception:
                    est_frames.append(0)
            self.start_progress("")
            # reset per-video progress bars for videos not in this batch
            self.vid_progress = [(0.0, "")] * len(self.rows)
            # map batch index → row index so progress bars can be updated
            row_idx = {}
            for j, b in enumerate(items):
                try:
                    row_idx[j] = self.rows.index(b.video)
                except ValueError:
                    pass

            def show():
                nonlocal done_n
                tot_frac = sum((1.0 if ended[k] else (frames[k] / totals[k] if totals[k] else 0.0))
                               for k in range(len(items)))
                self.progress = tot_frac / max(1, len(items))
                el = time.time() - t0
                fps = sum(frames) / el if el > 1.0 else 0
                names = ", ".join(os.path.basename(items[k].video) for k in sorted(running)[:2])
                self.bulk_msg = f"{done_n}/{len(items)} done"
                eta_str = ""
                if fps > 0:
                    known_remaining = sum(max(0, totals[k] - frames[k])
                                         for k in range(len(items)) if totals[k] > 0 and not ended[k])
                    not_started_est = sum(est_frames[k]
                                         for k in range(len(items)) if totals[k] == 0 and not ended[k])
                    est_remaining = known_remaining + not_started_est
                    eta_str = f" | ETA {_fmt_eta(est_remaining / fps)}"
                self.status = (f"Batch {done_n}/{len(items)}" + (f", {len(running)} running" if workers > 1 else "")
                               + f": {names} | {fl.BACKEND['text']} | {fps:.0f} fps{eta_str}")

            def prog(i, n, total):
                if self.cancel_flag:
                    raise fl.Cancelled()
                if i not in vid_t0:
                    vid_t0[i] = time.time()
                running.add(i)
                frames[i], totals[i] = n, total
                show()
                ri = row_idx.get(i)
                if ri is not None:
                    frac = n / total if total else 0.0
                    el_vid = time.time() - vid_t0[i]
                    fps_vid = n / el_vid if el_vid > 0.5 and n > 0 else 0.0
                    eta_vid = (total - n) / fps_vid if fps_vid > 0 else 0.0
                    detail = (f"{fps_vid:.0f} fps  ETA {_fmt_eta(eta_vid)}" if fps_vid > 0
                              else f"{n}/{total}")
                    self.vid_progress[ri] = (frac, detail)

            _STATUS_LABEL = {"ok": "done", "empty": "no keyframes", "failed": "failed",
                             "cancelled": "cancelled", "noflow": "no flow cache"}

            def on_item(i, st, det):
                nonlocal done_n
                ended[i] = True
                running.discard(i)
                done_n += 1
                show()
                ri = row_idx.get(i)
                if ri is not None:
                    self.vid_progress[ri] = (1.0, _STATUS_LABEL.get(st, st))

            res = fl.process_many(items, workers, prog, lambda: self.cancel_flag, on_item)
        except Exception as e:
            self.status = f"batch error: {e}"
        finally:
            self.prog_prefix = ""
            self.bulk = False
            self.progress = 0.0
        done = [b.video for b, r in zip(items, res) if r[0] == "ok"]
        empty = [b.video for b, r in zip(items, res) if r[0] == "empty"]
        failed = [(b.video, r[1]) for b, r in zip(items, res) if r[0] == "failed"]
        noflow = [b.video for b, r in zip(items, res) if r[0] == "noflow"]
        cancelled = self.cancel_flag or any(r[0] == "cancelled" for r in res)
        self.cancel_flag = False
        elapsed = time.time() - t0
        txt = (f"done in {elapsed:.1f} s — {len(done)} created, {len(skipped)} skipped, "
               f"{len(empty)} no keyframes, {len(failed)} failed"
               + (f", {len(noflow)} no cached flow" if cache_only else "")
               + (", cancelled" if cancelled else ""))
        self.status = "batch: " + txt
        self.bulk_summary = (txt, failed + [(v, "no cached flow for current settings") for v in noflow], empty)
        self.bulk_done = True

    # ------------------------------------------------- apply results
    def apply_flow(self):
        dx, dy, mag, W, H, sig, fid = self.pending_flow
        self.pending_flow = None
        self.flow = (dx, dy, mag, W, H)
        self.sig = sig
        self.fid = fid
        self.update_sel_text()
        self.res = None              # old result no longer matches this flow
        self.geom = (mag.shape[2], mag.shape[1])
        if not (self.a.start <= self.t < self.chapter_end(self.active)):
            self.t = self.a.start
        self.dirty_view = True

    def apply_result(self):
        r, self.pending = self.pending, None
        self.res = r
        self.dbg = r["dbg"]
        t = r["t"]
        tl = t.tolist()
        series = {"s_mag": r["ch_mag"], "s_per": r["ch_period"], "s_score": r["ch_score"], "s_best": r["best"],
                  "s_v": r["v"], "s_p": r["p"], "s_raw": r["raw"], "s_pos": r["pos"], "s_fs": r["fs"],
                  "s_spd": r["fs_speed"]}
        for tag, y in series.items():
            dpg.set_value(tag, [tl, y.tolist()])
        dpg.set_value("s_kf", [r["kx"].tolist(), r["ky"].tolist()])
        for tag in ("sw_line", "lim_line", "fl_line"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        if r["switch_t"]:
            dpg.add_inf_line_series(r["switch_t"], label="region switch", tag="sw_line", parent="ty0")
        if r["flip_t"]:
            # windows where the axis direction was decided by continuity and flipped relative to the previous window
            dpg.add_inf_line_series(r["flip_t"], label="axis direction flipped", tag="fl_line", parent="ty1")
        lim = float(self.a.max_speed)
        dpg.add_inf_line_series([lim, -lim], horizontal=True, label="speed limit", tag="lim_line", parent="ty4")
        dpg.set_axis_limits("ty4", -1.3 * lim, 1.3 * lim)
        up = "0" if self.eff_invert() else "100"      # where the positive signal points
        dpg.configure_item("s_v", label=f"velocity (+ = {up})")
        dpg.configure_item("s_p", label=f"filtered (+ = {up})")
        dpg.configure_item("s_raw", label=f"raw (+ = {up})")
        self.ov_dirty = True
        if self.fresh:
            self.fresh = False
            self.t = float(t[0])
            span = float(t[-1] - t[0])
            for i in range(5):
                dpg.set_axis_limits(f"tx{i}", float(t[0]), float(t[0] + min(span, 25.0)))
            if self.tempo_ini is not None:
                self.t = float(t[0] + self.tempo_ini)
        self.cur_win = -1
        self.dirty_view = True
        if self.mode != "none":
            self.rect_reset = True

    # ----------------------------------------------------------- rendering
    def frame_index(self):
        r = self.res
        return int(min(max(round((self.t - self.a.start) * self.fps()), 0), len(r["t"]) - 1))

    def rect(self, y0, x0, fw, fh):
        gx, gy = self.geom
        rh, rw = self.res["grid"]
        return (int(x0 * fw / gx), int(y0 * fh / gy), int((x0 + rw) * fw / gx), int((y0 + rh) * fh / gy))

    def overlay(self, img, ti):
        r = self.res
        fh, fw = img.shape[:2]
        gx, gy = self.geom
        d = self.dbg[int(r["widx"][ti])]
        get = dpg.get_value
        if get("ly_heat"):
            E = d["E"]
            h = cv2.applyColorMap((np.clip(E / (E.max() + 1e-9), 0, 1) * 255).astype(np.uint8),
                                  cv2.COLORMAP_INFERNO)
            h = cv2.resize(h, (fw, fh), interpolation=cv2.INTER_NEAREST)
            img[:] = cv2.addWeighted(img, 0.65, h, 0.35, 0)
        if get("ly_mag"):
            m = np.clip(self.flow[2][ti].astype(np.float32) / r["mag_max"], 0, 1)
            h = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            h = cv2.resize(h, (fw, fh), interpolation=cv2.INTER_NEAREST)
            img[:] = cv2.addWeighted(img, 0.5, h, 0.5, 0)
        if get("ly_per") and d["cands"]:
            ov = img.copy()
            for c in d["cands"]:
                b = self.rect(c["y"], c["x"], fw, fh)
                col = cv2.applyColorMap(np.array([[int(255 * min(max(c["period"], 0), 1))]], np.uint8),
                                        cv2.COLORMAP_JET)[0, 0]
                cv2.rectangle(ov, (b[0], b[1]), (b[2], b[3]), tuple(int(q) for q in col), -1)
            img[:] = cv2.addWeighted(img, 0.6, ov, 0.4, 0)
            for c in d["cands"]:
                b = self.rect(c["y"], c["x"], fw, fh)
                fl.put(img, f"P {c['period']:.2f}", (b[0] + 3, b[1] + 13), 0.4)
        if get("ly_cand"):
            for c in d["cands"]:
                b = self.rect(c["y"], c["x"], fw, fh)
                col = fl.PALETTE[c["i"] % len(fl.PALETTE)]
                fl.box(img, b, col, 1)
                fl.put(img, f"{c['score']:.2f}", (b[0] + 3, b[1] + 13), 0.4, col)
            if d["prev"] is not None and tuple(d["prev"]) != tuple(d["chosen"]):
                b = self.rect(d["prev"][0], d["prev"][1], fw, fh)
                fl.box(img, b, (255, 255, 0), 2)
        if get("ly_axis"):
            self.draw_compass(img, d, r)
        roi = self.rect(d["chosen"][0], d["chosen"][1], fw, fh)
        if get("ly_roi"):
            fl.box(img, roi, (0, 255, 0), 3)
        if get("ly_axis") and "axis" in d:
            ux, uy = d["axis"]
            cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
            # (ux, uy) is the motion axis. With "invert" the funscript 100 end points the other way,
            # so all scaling uses dx, dy = s * (ux, uy).
            s = -1.0 if self.eff_invert() else 1.0
            dx, dy = s * ux, s * uy
            px, py = -uy, ux                                    # perpendicular ao eixo
            YEL, MAG = (0, 255, 255), (255, 0, 255)             # BGR: yellow and magenta

            def ends(bx, by, ln, col, side, off):
                """Marks both ends of a vector: 100 (filled) and 0 (empty), with the label alongside."""
                for val, sg in (("100", 1.0), ("0", -1.0)):
                    ex, ey = bx + dx * ln * sg, by + dy * ln * sg
                    if sg > 0:
                        cv2.circle(img, (int(ex), int(ey)), 4, col, -1)
                    else:
                        cv2.circle(img, (int(ex), int(ey)), 4, col, 1)
                    tx, ty = ex + px * side * off, ey + py * side * off
                    fl.put(img, val, (int(tx - 4 * len(val)), int(ty + 4)), 0.4, col)

            # velocity (yellow): line with 0 and 100 ends, arrow pointing in the actual direction of motion
            cv2.line(img, (int(cx - dx * 30), int(cy - dy * 30)), (int(cx + dx * 30), int(cy + dy * 30)), (255, 255, 255), 1)
            ends(cx, cy, 30, YEL, -1.0, 14)
            ln = 50 * float(np.clip(r["v"][ti] / r["vm"], -1, 1))
            cv2.arrowedLine(img, (cx, cy), (int(cx + ux * ln), int(cy + uy * ln)), YEL, 3, tipLength=0.3)
            # funscript position (magenta), alongside, same scale. Arrow goes from 50 to the current value
            pos = float(r["fs"][ti])
            lp = 30 * (pos - 50.0) / 50.0
            bx, by = cx + px * 9, cy + py * 9
            cv2.line(img, (int(bx - dx * 30), int(by - dy * 30)), (int(bx + dx * 30), int(by + dy * 30)), (110, 110, 110), 1)
            ends(bx, by, 30, MAG, 1.0, 14)
            cv2.arrowedLine(img, (int(bx), int(by)), (int(bx + dx * lp), int(by + dy * lp)), MAG, 3, tipLength=0.3)
            fl.put(img, f"fs {pos:.0f}", (int(bx + dx * 46), int(by + dy * 46 + 4)), 0.4, MAG)

    @staticmethod
    def _angle_label(deg):
        # deg: 0=right, 90=down, 180=left, 270=up (screen coords)
        deg = deg % 360
        if deg < 23 or deg >= 338: return "right"
        if deg < 68: return "down-right"
        if deg < 113: return "down"
        if deg < 158: return "down-left"
        if deg < 203: return "left"
        if deg < 248: return "up-left"
        if deg < 293: return "up"
        return "up-right"

    def draw_compass(self, img, d, r):
        """Compass in the video corner: shows where 100 points (the axis-ref in effect for this chapter)."""
        ref = d.get("axis_ref")
        if ref is None:
            txt = "100: continuity (auto)"
        else:
            s = -1.0 if self.eff_invert() else 1.0
            ux, uy = s * ref[0], s * ref[1]
            cx, cy, R = 38, 38, 24
            CY = (255, 220, 0)
            cv2.circle(img, (cx, cy), R + 6, (30, 30, 30), -1)
            cv2.circle(img, (cx, cy), R + 6, (120, 120, 120), 1)
            cv2.arrowedLine(img, (int(cx - ux * R), int(cy - uy * R)), (int(cx + ux * R), int(cy + uy * R)), CY, 3,
                            tipLength=0.35)
            fl.put(img, "100", (int(cx + ux * (R + 13) - 10), int(cy + uy * (R + 13) + 4)), 0.35, CY)
            fl.put(img, "0", (int(cx - ux * (R + 11) - 3), int(cy - uy * (R + 11) + 4)), 0.35, (200, 200, 200))
            deg = int(round(np.degrees(np.arctan2(ref[1], ref[0])))) % 360
            c = self.chapters[self.active]
            lbl = self._angle_label(deg)
            txt = f"{lbl} ({deg}°)  ({'chapter' if c.get('axis_ref') else 'computed'})"
            fl.put(img, txt, (cx + R + 12, cy + 4), 0.4, CY)
            info = f"in effect: 100 → {lbl} ({deg}°)  ({'set in this chapter' if c.get('axis_ref') else 'computed from motion'})"
            if info != self.axinfo:
                self.axinfo = info
                dpg.set_value("chap_axinfo", info)
            return
        if txt != self.axinfo:
            self.axinfo = txt
            dpg.set_value("chap_axinfo", "active: " + txt)
        fl.put(img, txt, (10, 20), 0.4, (255, 220, 0))

    def render(self):
        if self.src is None:
            return
        a = self.a
        zoom_mode = self.mode == "zoom"
        ci = self.chapter_at(self.t)
        zoom_now = a.zoom if ci == self.active else self.chapters[ci]["zoom"]
        x, y, w, h = crop_rect(a.crop, zoom_now, self.src.sw, self.src.sh,
                               use_zoom=not zoom_mode)
        frame = self.src.get(self.t)
        if frame is None:
            return
        crop = frame[y:y + h, x:x + w]
        s = min(CW / w, CH / h)
        fw, fh = max(2, int(w * s)), max(2, int(h * s))
        img = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_AREA)
        ox, oy = (CW - fw) // 2, (CH - fh) // 2
        self.view = (ox, oy, fw, fh)
        if self.res is not None and self.flow is not None and not zoom_mode and not self.geometry_stale() and ci == self.active:
            self.overlay(img, self.frame_index())
        canvas = np.full((CH, CW, 3), 24, np.uint8)
        canvas[oy:oy + fh, ox:ox + fw] = img
        if self.rect_reset:
            self.rect_reset = False
            self.place_rect()
        self.draw_sel(canvas)
        self.buf[..., :3] = canvas[..., ::-1] * (1.0 / 255.0)

    def refresh_inspector(self, k):
        d = self.dbg[k]
        fps, a = self.fps(), self.a
        dpg.set_value("in_win", f"Window {k + 1}/{len(self.dbg)}   {d['start'] / fps + a.start:.1f}s to {d['end'] / fps + a.start:.1f}s")
        dpg.set_value("in_formula", f"score = {a.period_weight:.2f} x P + {1 - a.period_weight:.2f} x M")
        cands = sorted(d["cands"], key=lambda c: -c["score"])
        for i in range(MAXC):
            if i < len(cands):
                c = cands[i]
                tag = "*" if (c["y"], c["x"]) == tuple(d["chosen"]) else " "
                dpg.configure_item(f"ct{i}", default_value=f"{tag}y{c['y']:>2} x{c['x']:>2}  M {c['mag']:.2f}  P {c['period']:.2f}",
                                   color=PALETTE_RGB[c["i"] % len(PALETTE_RGB)])
                dpg.set_value(f"cb{i}", min(max(c["score"], 0.0), 1.0))
                dpg.configure_item(f"cb{i}", overlay=f"{c['score']:.2f}", show=True)
            else:
                dpg.configure_item(f"ct{i}", default_value="")
                dpg.configure_item(f"cb{i}", show=False)
        for i in range(MAXC):
            c = next((c for c in d["cands"] if c["i"] == i), None)
            if c is not None and c.get("ac"):
                ac = c["ac"]["ac"]
                dpg.set_value(f"ac{i}", [(np.arange(len(ac)) / fps).tolist(), ac.tolist()])
                dpg.configure_item(f"ac{i}", show=True)
            else:
                dpg.configure_item(f"ac{i}", show=False)
        if d["fixed"]:
            for tag in ("in_prev", "in_thr", "in_best"):
                dpg.set_value(tag, "")
            dpg.set_value("in_dec", "Fixed region: no automatic decision")
            return
        if d["prev"] is None:
            dpg.set_value("in_prev", "No previous region.")
            dpg.set_value("in_thr", "")
        else:
            dpg.set_value("in_prev", f"Previous region y{d['prev'][0]} x{d['prev'][1]}: stored {d['prev_score']:.2f}, now {d['now']['score']:.2f}")
            dpg.set_value("in_thr", f"Threshold = {a.switch:.2f} x {d['prev_score']:.2f} = {d['threshold']:.2f}")
        dpg.set_value("in_best", f"Best candidate y{d['best'][0]} x{d['best'][1]}: {d['best_score']:.2f}")
        if d["switched"]:
            dpg.configure_item("in_dec", default_value="SWITCHED region" if d["prev"] else "INITIAL CHOICE", color=(60, 255, 60))
        else:
            why = "already the best" if d["best"] == d["prev"] else "did not pass threshold"
            dpg.configure_item("in_dec", default_value=f"KEPT ({why})", color=(255, 165, 0))

    # -------------------------------------------------------------- main loop
    def tick(self):
        now = time.perf_counter()
        dt, self.last_tick = now - self.last_tick, now
        if self.chap_done:
            self.chap_done = False
            self.refresh_chapters()
            if self.sig is None and self.src is not None and self.cache_ok():
                self.request_flow()
        if self.accel_lines is not None:
            dpg.set_value("accel_text", "\n".join(self.accel_lines))
            self.accel_lines = None
        if self.suggested is not None and not self.busy():
            self.apply_suggested()
        if self.bulk_done:
            self.bulk_done = False
            txt, failed, empty = self.bulk_summary
            extra = "".join("\nerror: " + os.path.basename(v) + ": " + e[:60] for v, e in failed[:3])
            extra += "".join("\nno keyframes: " + os.path.basename(v) for v in empty[:3])
            if self.rows:
                self.open_path_refresh()
            dpg.set_value("bulk_text", txt + extra)
        dpg.configure_item("btn_cancel", show=self.flowing or self.bulk)
        self.handle_mouse()
        self.save_config()
        self.update_warning()
        self.handle_overview()
        if self.src is not None and self.src._seek_result is not None:
            self.dirty_view = True
        if self.pending_flow is not None:
            self.apply_flow()
        if self.pending is not None and self.flow is not None:
            if self.pending["fid"] == self.fid:
                self.apply_result()
            elif self.pending["fid"] < self.fid:
                self.pending = None          # stale result from a previous flow, discard
        dpg.set_value("prog", self.progress if (self.flowing or self.bulk) else 0.0)
        self.tick_status()
        if self.vid_progress:
            for ri, (frac, txt) in enumerate(self.vid_progress):
                if dpg.does_item_exist(f"vp_{ri}"):
                    show = frac > 0 or bool(txt)
                    dpg.set_value(f"vp_{ri}", frac)
                    dpg.configure_item(f"vp_{ri}", overlay=txt, show=show)
        ms = f"   (recompute {self.compute_ms:.0f} ms)" if self.res is not None else ""
        self.set_fitted("status", "status_tip", self.status + ms)
        self.set_fitted("warn", "warn_tip", self.warn_text)
        if self.res is None:
            return
        r = self.res
        if self.playing:
            self.set_time(self.t + dt * self.speed)
            if self.t >= r["t"][-1]:
                self.playing = False
        if self.dirty_view or self.playing:
            self.dirty_view = False
            ti = self.frame_index()
            k = int(r["widx"][ti])
            if k != self.cur_win:
                self.cur_win = k
                self.refresh_inspector(k)
            self.render()
            dpg.set_value("t_label", f"{self.t:.2f} s   funscript {r['fs'][ti]:.0f}   sinal {r['pos'][ti]:.0f}   v {r['v'][ti]:+.2f}")
            if time.time() > self.drag_until:
                for i in range(5):
                    dpg.set_value(f"ph{i}", self.t)
            if self.playing and dpg.get_value("follow"):
                lo = self.t - dpg.get_value("follow_w") * 0.5
                for i in range(5):
                    dpg.set_axis_limits(f"tx{i}", lo, lo + dpg.get_value("follow_w"))

    def step(self):
        """One step of the main loop (called before rendering the frame)."""
        self.tick()
        if self.frame == 4 and not self.have_layout:
            self.apply_default_layout()
        self.frame += 1
        if time.time() - self.layout_t > 15.0:
            self.layout_t = time.time()
            self.save_layout()
        if self.captura:
            self.capture_step()

    def run(self):
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()
        self.save_config(force=True)
        self.save_layout()
        self.alive = False
        dpg.destroy_context()

    def capture_step(self):
        """For headless testing: waits for the result, saves a screenshot and exits."""
        if self.cap_state == 0 and self.res is not None:
            self.settle += 1
            if self.settle > 20:
                dpg.output_frame_buffer(self.captura)
                self.cap_state = 1
                self.settle = 0
        elif self.cap_state == 1:
            self.settle += 1
            if self.settle > 10:
                dpg.stop_dearpygui()


def main():
    ap = argparse.ArgumentParser(description="FunAutoScripter GUI (DearPyGui).")
    ap.add_argument("path", nargs="?", help="video or folder")
    ap.add_argument("--captura", metavar="PNG", help="save a screenshot and exit (for testing)")
    ap.add_argument("--tempo", type=float, help="initial cursor time in seconds (for testing)")
    a = ap.parse_args()
    path = a.path
    if not path and not a.captura:
        cfg = load_config()
        ult = cfg.get("last_path") or cfg.get("ultimo")
        path = ult if ult and os.path.exists(ult) else None
    app = App(path, a.captura, a.tempo)
    app.run()


if __name__ == "__main__":
    main()
