#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemini Cleaner - local, offline, bulk Gemini / Veo video watermark remover (Android)."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner

APP_NAME = "Gemini Cleaner"
VERSION = "1.0"
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")
METHOD_KEYS = {"Auto (best)": "auto", "Unblend (sharp)": "unblend",
               "Delogo (fast)": "delogo", "Crop (band cut)": "crop"}

FFMPEG = None
FFPROBE = None


# --------------------------------------------------------------------- ffmpeg
def _runs(path):
    try:
        return subprocess.run([path, "-version"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except Exception:
        return False


def _locate(name):
    p = shutil.which(name)
    if p and _runs(p):
        return p
    here = Path(__file__).resolve().parent
    for c in (here / "ffbin" / name, here / name):
        if c.exists():
            try:
                os.chmod(c, 0o755)
            except Exception:
                pass
            if _runs(str(c)):
                return str(c)
    return None


def setup_binaries(workdir):
    """Bundled ffmpeg may not be executable in the app dir -> copy to a writable dir."""
    global FFMPEG, FFPROBE
    FFMPEG = _locate("ffmpeg")
    FFPROBE = _locate("ffprobe")
    if FFMPEG and FFPROBE:
        return
    here = Path(__file__).resolve().parent / "ffbin"
    dst = Path(workdir) / "bin"
    dst.mkdir(parents=True, exist_ok=True)
    for name, var in (("ffmpeg", "FFMPEG"), ("ffprobe", "FFPROBE")):
        tgt = dst / name
        src = here / name
        try:
            if src.exists() and not tgt.exists():
                shutil.copy2(src, tgt)
            if tgt.exists():
                os.chmod(tgt, 0o755)
                if _runs(str(tgt)):
                    globals()[var] = str(tgt)
        except Exception:
            pass


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        lines = (p.stderr or "").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "ffmpeg failed")


# ------------------------------------------------------------------ geometry
def default_box(W, H):
    ref = min(W, H)
    s = max(8, int(round(ref / 11.25)))
    m = max(4, int(round(ref / 16.875)))
    return max(0, W - m - s), max(0, H - m - s), s, s


def pad_box(box, W, H):
    x, y, w, h = [int(v) for v in box]
    p = max(2, int(round(min(w, h) * 0.08)))
    x, y, w, h = x - p, y - p, w + 2 * p, h + 2 * p
    x, y = max(1, x), max(1, y)
    return x, y, min(w, W - x - 1), min(h, H - y - 1)


def _safe_box(box, W, H):
    x, y, w, h = [int(v) for v in box]
    w, h = max(4, min(w, W - 4)), max(4, min(h, H - 4))
    return max(1, min(x, W - w - 1)), max(1, min(y, H - h - 1)), w, h


def sparkle_alpha(w, h, p=0.66, soft=0.10):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs + 0.5) / w * 2.0 - 1.0
    v = (ys + 0.5) / h * 2.0 - 1.0
    r = np.abs(u) ** p + np.abs(v) ** p
    return np.clip((1.0 - r) / soft, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- probe
def probe(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                        "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                        str(path)], capture_output=True, text=True)
    vals = [l.strip() for l in (p.stdout or "").splitlines() if l.strip()]
    if len(vals) < 4:
        raise RuntimeError("ffprobe failed")
    W, H = int(vals[0]), int(vals[1])
    rfr = vals[2] if "/" in vals[2] else "30/1"
    try:
        n, d = rfr.split("/")
        fps = float(n) / float(d) if float(d) else 30.0
    except Exception:
        fps, rfr = 30.0, "30/1"
    dur = float(vals[3]) if len(vals) > 3 else 0.0
    nf = int(dur * fps) if dur else 60
    return {"W": W, "H": H, "fps": fps, "fps_str": rfr, "dur": dur, "nf": max(nf, 2)}


def _sample_gray(path, nf, pw, ph, n=20, box=None):
    step = max(1, int(nf // max(1, n)))
    parts = ["select='not(mod(n\\,%d))'" % step]
    if box:
        x, y, w, h = box
        parts.append("crop=%d:%d:%d:%d" % (w, h, x, y))
    else:
        parts.append("scale=%d:%d:flags=bilinear" % (pw, ph))
    parts.append("format=gray")
    p = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vf", ",".join(parts),
                        "-vsync", "0", "-frames:v", str(n), "-f", "rawvideo", "-"],
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return []
    buf = np.frombuffer(p.stdout, np.uint8)
    fs = pw * ph
    return [buf[i * fs:(i + 1) * fs].reshape(ph, pw).copy() for i in range(buf.size // fs)]


def _edges(g):
    g = g.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > max(6.0, float(np.percentile(mag, 78)))


def detect_box(path, W, H, nf, samples=20):
    fallback = default_box(W, H)
    if nf < 3:
        return fallback, 0.0
    pw = min(W, 640)
    ph = max(2, int(round(H * pw / W)))
    frames = _sample_gray(path, nf, pw, ph, samples)
    if len(frames) < 3:
        return fallback, 0.0
    persist = np.mean([_edges(f) for f in frames], axis=0).astype(np.float32)
    sc = pw / float(W)
    ref = min(W, H)
    s0 = max(6, int(round(ref / 11.25 * sc)))
    m0 = max(3, int(round(ref / 16.875 * sc)))
    offs = (0, -m0 // 2, m0 // 2)
    best_s, best_b = -9e9, fallback
    for k in (0.75, 1.0, 1.35):
        sz = max(5, int(round(s0 * k)))
        for corner in ("br", "bl", "tr", "tl"):
            for ox in offs:
                for oy in offs:
                    y = (ph - m0 - sz + oy) if corner[0] == "b" else (m0 + oy)
                    x = (pw - m0 - sz + ox) if corner[1] == "r" else (m0 + ox)
                    x = max(0, min(x, pw - sz))
                    y = max(0, min(y, ph - sz))
                    inner = persist[y:y + sz, x:x + sz]
                    if inner.size == 0:
                        continue
                    p = max(2, sz // 2)
                    ex0, ey0 = max(0, x - p), max(0, y - p)
                    ex1, ey1 = min(pw, x + sz + p), min(ph, y + sz + p)
                    ring = persist[ey0:ey1, ex0:ex1].copy()
                    ring[y - ey0:y - ey0 + sz, x - ex0:x - ex0 + sz] = np.nan
                    with np.errstate(invalid="ignore"):
                        rin = float(np.nanmean(ring)) if np.isfinite(ring).any() else 0.0
                    s = float(inner.mean()) - rin
                    if s > best_s:
                        best_s, best_b = s, (int(round(x / sc)), int(round(y / sc)),
                                             int(round(sz / sc)), int(round(sz / sc)))
    return (fallback, best_s) if best_s < 0.10 else (best_b, best_s)


def calibrate_alpha(path, box, nf, n=16):
    x, y, w, h = box
    frames = _sample_gray(path, nf, w, h, n, box=(x, y, w, h))
    if len(frames) < 4:
        return 0.28, False
    arr = np.stack([f.astype(np.float32) for f in frames])
    mask = sparkle_alpha(w, h)
    core, bg = mask > 0.7, mask < 0.15
    if core.sum() < 16 or bg.sum() < 16:
        return 0.28, False
    a_min = float(np.median(arr.min(axis=0)[core] / 255.0))
    sd = arr.std(axis=0)
    ref_sd, core_sd = float(np.median(sd[bg])), float(np.median(sd[core]))
    a_std = (1.0 - core_sd / ref_sd) if ref_sd > 1.5 else None
    peak = float(min([a_min] + ([a_std] if a_std else [])))
    conf = 0.12 <= peak <= 0.45 and (a_std is None or abs(a_min - a_std) < 0.18)
    return float(np.clip(peak, 0.08, 0.45)), conf


# ------------------------------------------------------------------ removal
def remove_delogo(src, dst, box, W, H, crf):
    x, y, w, h = _safe_box(box, W, H)
    _run([FFMPEG, "-y", "-i", str(src), "-vf", "delogo=x=%d:y=%d:w=%d:h=%d" % (x, y, w, h),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
          "-c:a", "copy", "-movflags", "+faststart", str(dst)])


def remove_crop(src, dst, box, W, H, crf):
    x, y, w, h = _safe_box(box, W, H)
    _run([FFMPEG, "-y", "-i", str(src),
          "-vf", "crop=%d:%d:0:0,scale=%d:%d:flags=lanczos" % (W - w, H - h, W, H),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
          "-c:a", "copy", "-movflags", "+faststart", str(dst)])


def _write_pgm(path, W, H, box, peak):
    x, y, w, h = box
    img = np.zeros((H, W), np.uint8)
    img[y:y + h, x:x + w] = (sparkle_alpha(w, h) * peak * 255.0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(img.tobytes())


def remove_unblend(src, dst, box, W, H, fps_str, peak, crf, tmpdir):
    """O = (W - a*L) / (1 - a) done entirely inside ffmpeg's blend filter."""
    x, y, w, h = _safe_box(box, W, H)
    pgm = Path(tmpdir) / "mask.pgm"
    _write_pgm(pgm, W, H, (x, y, w, h), peak)
    fc = ("[0:v]format=rgb24,fps=%s,settb=AVTB[a];"
          "[1:v]format=rgb24,fps=%s,settb=AVTB[b];"
          "[a][b]blend=all_expr='(A-B)*255/max(255-B,1)',format=yuv420p[v]") % (fps_str, fps_str)
    _run([FFMPEG, "-y", "-i", str(src), "-loop", "1", "-framerate", fps_str, "-i", str(pgm),
          "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0?",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
          "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(dst)])


def process_file(src, outdir, method, strength, crf, emit, tmpdir):
    src = Path(src)
    info = probe(src)
    W, H, nf = info["W"], info["H"], info["nf"]
    box, score = detect_box(src, W, H, nf)
    box = pad_box(box, W, H)

    peak = 0.28
    m = method
    if m == "auto":
        peak, conf = calibrate_alpha(src, box, nf)
        m = "unblend" if conf else "delogo"
    elif m == "unblend":
        peak, _ = calibrate_alpha(src, box, nf)
    peak *= strength

    out = Path(outdir) / (src.stem + "_clean.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    emit("log", "%s  box=%s score=%.2f -> %s" % (src.name, box, score, m))

    try:
        if m == "crop":
            remove_crop(src, out, box, W, H, crf)
        elif m == "unblend":
            remove_unblend(src, out, box, W, H, info["fps_str"], peak, crf, tmpdir)
        else:
            remove_delogo(src, out, box, W, H, crf)
    except Exception as e:
        if m != "delogo":
            emit("log", "  %s failed (%s), retrying with delogo" % (m, e))
            remove_delogo(src, out, box, W, H, crf)
        else:
            raise
    return out.name


# ----------------------------------------------------------------------- UI
class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=dp(6), spacing=dp(6), **kw)
        self.selected = []
        self.ev = queue.Queue()
        self.stop = threading.Event()
        self.running = False
        self.outdir = None
        self.tmpdir = tempfile.mkdtemp(prefix="gclean_")

        bar = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self.method = Spinner(text="Auto (best)", values=list(METHOD_KEYS), size_hint_x=0.5)
        bar.add_widget(self.method)
        bar.add_widget(Button(text="Add selected", size_hint_x=0.28, on_release=self.add_sel))
        bar.add_widget(Button(text="Clear", size_hint_x=0.22, on_release=self.clear))
        self.add_widget(bar)

        start = "/storage/emulated/0"
        if not os.path.isdir(start):
            start = os.path.expanduser("~")
        self.chooser = FileChooserListView(path=start, multiselect=True,
                                           filters=["*.mp4", "*.mov", "*.webm", "*.mkv", "*.m4v"])
        self.add_widget(self.chooser)

        row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        self.out_lbl = Label(text="Output: same folder as source", size_hint_x=0.62)
        row.add_widget(self.out_lbl)
        row.add_widget(Button(text="Use this folder", on_release=self.set_out))
        self.add_widget(row)

        self.go = Button(text="START", size_hint_y=None, height=dp(50), on_release=self.start)
        self.add_widget(self.go)
        self.pb = ProgressBar(max=100)
        self.add_widget(self.pb)
        self.status = Label(text="idle", size_hint_y=None, height=dp(26), font_size="12sp")
        self.add_widget(self.status)

        sv = ScrollView()
        self.loglbl = Label(text="", size_hint_y=None, halign="left", valign="top", font_size="11sp")
        self.loglbl.bind(width=lambda *_: setattr(self.loglbl, "text_size", (self.loglbl.width, None)))
        self.loglbl.bind(texture_size=lambda *_: setattr(self.loglbl, "height", self.loglbl.texture_size[1]))
        sv.add_widget(self.loglbl)
        self.add_widget(sv)

        Clock.schedule_interval(self.poll, 0.2)

    def log(self, s):
        lines = (self.loglbl.text + s + "\n").splitlines()[-60:]
        self.loglbl.text = "\n".join(lines) + "\n"

    def add_sel(self, *_):
        for p in self.chooser.selection:
            if p.lower().endswith(VIDEO_EXTS) and p not in self.selected:
                self.selected.append(p)
        self.status.text = "%d file(s) queued" % len(self.selected)

    def clear(self, *_):
        self.selected.clear()
        self.status.text = "0 file(s) queued"

    def set_out(self, *_):
        self.outdir = self.chooser.path
        self.out_lbl.text = "Output: " + self.outdir

    def start(self, *_):
        if self.running:
            self.stop.set()
            self.go.text = "STOPPING..."
            return
        if not self.selected:
            self.status.text = "Add some videos first"
            return
        if not FFMPEG:
            self.status.text = "ffmpeg not found (ffbin/ missing)"
            return
        self.running = True
        self.stop.clear()
        self.go.text = "STOP"
        self.pb.value = 0
        threading.Thread(target=self.work, args=(list(self.selected),), daemon=True).start()

    def work(self, batch):
        total, done = len(batch), 0
        for f in batch:
            if self.stop.is_set():
                break
            self.ev.put(("status", "working: " + Path(f).name))
            try:
                outdir = self.outdir or str(Path(f).parent)
                name = process_file(f, outdir, METHOD_KEYS[self.method.text], 1.0, 18,
                                    lambda k, s: self.ev.put((k, s)), self.tmpdir)
                self.ev.put(("log", "[ok] " + name))
            except Exception as e:
                self.ev.put(("log", "[FAIL] %s : %s" % (Path(f).name, e)))
            done += 1
            self.ev.put(("prog", done * 100.0 / total))
        self.ev.put(("done", "finished"))

    def poll(self, *_):
        try:
            while True:
                k, v = self.ev.get_nowait()
                if k == "log":
                    self.log(v)
                elif k == "status":
                    self.status.text = v
                elif k == "prog":
                    self.pb.value = v
                elif k == "done":
                    self.running = False
                    self.go.text = "START"
                    self.status.text = "done"
        except queue.Empty:
            pass


class CleanerApp(App):
    def build(self):
        self.title = APP_NAME
        root = Root()
        Clock.schedule_once(lambda dt: self._boot(root), 0.4)
        return root

    def _boot(self, root):
        try:
            from android.permissions import request_permissions, Permission
            perms = [getattr(Permission, n) for n in
                     ("READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE", "READ_MEDIA_VIDEO")
                     if hasattr(Permission, n)]
            if perms:
                request_permissions(perms)
        except Exception:
            pass
        setup_binaries(self.user_data_dir)
        root.log("ffmpeg: %s" % (FFMPEG or "NOT FOUND"))
        root.log("ffprobe: %s" % (FFPROBE or "NOT FOUND"))
        if not FFMPEG:
            root.log("ffbin/ khali hai - workflow me ffmpeg download step check karo.")

    def on_stop(self):
        pass


if __name__ == "__main__":
    CleanerApp().run()