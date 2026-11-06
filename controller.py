"""
Manual controller and telemetry viewer for the robot (pygame).

Left:    live 3D orientation from robot telemetry.
Right:   per-servo sliders (FL, FR, BL, BR).
Bottom:  roll / pitch / yaw history and mode buttons.

Axes: X+ forward (head), Y+ right, Z+ up.

Usage:
    python controller.py                      # BLE
    python controller.py --serial /dev/cu.X   # USB serial

Keys:
    1-4 + left/right    nudge the selected servo 1 deg (shift: 5)
    space               center all
    s / i               sweep / IMU-only modes
    r                   snap sliders to robot angles
    esc                 quit
"""

import argparse
import asyncio
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame
from pygame.locals import (
    QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_LEFT, K_RIGHT,
    K_1, K_2, K_3, K_4, K_s, K_i, K_r, KMOD_SHIFT,
    MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
)

DEVICE_NAMES = ("Eshy", "TechTalkies")   # firmware/, firmware_walk/
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

# =============================================================================
# Shared state — accessed from UI + comm thread, protected by state_lock
# =============================================================================
@dataclass
class SharedState:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    device_angles: list = field(default_factory=lambda: [90, 90, 90, 90])
    slider_angles: list = field(default_factory=lambda: [90, 90, 90, 90])
    connected: bool = False
    mode_label: str = "..."
    roll_hist: deque = field(default_factory=lambda: deque(maxlen=300))
    pitch_hist: deque = field(default_factory=lambda: deque(maxlen=300))
    yaw_hist: deque = field(default_factory=lambda: deque(maxlen=300))
    rx_buffer: str = ""


state = SharedState()
state_lock = threading.Lock()

# Outgoing command queue — UI thread produces, comm thread consumes
out_queue = []
out_lock = threading.Lock()


def send(cmd: str) -> None:
    """Enqueue a command to the device (newline is added by the sender)."""
    with out_lock:
        out_queue.append(cmd)


# =============================================================================
# Parser — handles chunked BLE notifications
# =============================================================================
_line_re = re.compile(
    r'R:([+-]?[\d.]+)\s+P:([+-]?[\d.]+)\s+Y:([+-]?[\d.]+)\s+S:(\d+),(\d+),(\d+),(\d+)'
)


def ingest(data: str) -> None:
    with state_lock:
        state.rx_buffer += data
        # Find *last* complete message (don't replay old ones if we fell behind)
        matches = list(_line_re.finditer(state.rx_buffer))
        if not matches:
            # Cap runaway buffer
            if len(state.rx_buffer) > 512:
                state.rx_buffer = state.rx_buffer[-128:]
            return
        m = matches[-1]
        state.roll = float(m.group(1))
        state.pitch = float(m.group(2))
        state.yaw = float(m.group(3))
        state.device_angles = [int(m.group(i)) for i in (4, 5, 6, 7)]
        state.roll_hist.append(state.roll)
        state.pitch_hist.append(state.pitch)
        state.yaw_hist.append(state.yaw)
        state.rx_buffer = state.rx_buffer[m.end():]


# =============================================================================
# BLE thread (default) — single persistent event loop
# =============================================================================
async def ble_main():
    from bleak import BleakClient, BleakScanner

    print(f"Scanning for {' / '.join(DEVICE_NAMES)}...")
    device = None
    while device is None:
        devices = await BleakScanner.discover(timeout=3.0)
        for d in devices:
            if d.name and any(n in d.name for n in DEVICE_NAMES):
                device = d
                break
        if device is None:
            print("  not found, retrying...")

    print(f"Connecting to {device.name} ({device.address})")

    def on_notify(_sender, data: bytearray):
        ingest(data.decode('utf-8', errors='ignore'))

    async with BleakClient(device.address) as client:
        with state_lock:
            state.connected = True
            state.mode_label = "BLE connected"
        print("BLE connected!")
        await client.start_notify(UART_TX_UUID, on_notify)

        while True:
            pending = []
            with out_lock:
                if out_queue:
                    pending = out_queue[:]
                    out_queue.clear()
            for cmd in pending:
                try:
                    await client.write_gatt_char(
                        UART_RX_UUID, (cmd + "\n").encode('utf-8'),
                        response=True,
                    )
                    print(f"-> {cmd}")
                except Exception as e:
                    print("BLE write failed:", e)
            await asyncio.sleep(0.03)


def ble_thread():
    try:
        asyncio.run(ble_main())
    except Exception as e:
        print("BLE thread error:", e)
        with state_lock:
            state.mode_label = f"BLE error: {e}"


# =============================================================================
# Serial thread — alternative to BLE
# =============================================================================
def serial_thread(port: str):
    import serial
    ser = serial.Serial(port, 115200, timeout=0.1)
    with state_lock:
        state.connected = True
        state.mode_label = f"Serial {port}"
    print(f"Serial connected on {port}")

    def reader():
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore')
            except Exception as e:
                print("serial read error:", e); return
            if line:
                ingest(line)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        pending = []
        with out_lock:
            if out_queue:
                pending = out_queue[:]
                out_queue.clear()
        for cmd in pending:
            try:
                ser.write((cmd + "\n").encode('utf-8'))
            except Exception as e:
                print("serial write error:", e)
        time.sleep(0.03)


# =============================================================================
# 3D helpers
# =============================================================================
def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Euler ZYX: yaw around Z, pitch around Y, roll around X."""
    r, p, y = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Camera behind and above the robot, so X+ points into the screen and Y+ is
# screen-right.
_EYE = np.array([-5.5, 0.0, 2.5])
_TARGET = np.array([0.0, 0.0, 0.0])
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def _camera_basis():
    f = _TARGET - _EYE
    f = f / np.linalg.norm(f)               # camera forward (into screen)
    r = np.cross(f, _WORLD_UP); r /= np.linalg.norm(r)  # camera right
    u = np.cross(r, f)                      # camera up
    return r, u, f


_CAM_R, _CAM_U, _CAM_F = _camera_basis()
_FOCAL = 260.0


def project(pt, cx, cy):
    rel = pt - _EYE
    x = np.dot(rel, _CAM_R)
    y = np.dot(rel, _CAM_U)
    z = np.dot(rel, _CAM_F)            # depth into screen
    if z < 0.2:
        z = 0.2
    return int(cx + x * _FOCAL / z), int(cy - y * _FOCAL / z)


# =============================================================================
# Drawing primitives
# =============================================================================
def draw_text(surf, font, text, xy, color=(220, 220, 220)):
    surf.blit(font.render(text, True, color), xy)


def draw_axis_triad(surf, R, cx, cy):
    """Small XYZ triad in world frame, rotated by R."""
    origin = np.array([0.0, 0.0, 0.0])
    o2d = project(origin, cx, cy)
    for vec, col, label in [
        (np.array([1.5, 0, 0]), (240, 80, 80), "X"),    # forward / head
        (np.array([0, 1.5, 0]), (80, 200, 80), "Y"),    # right
        (np.array([0, 0, 1.5]), (120, 160, 255), "Z"),  # up
    ]:
        tip_2d = project(R @ vec, cx, cy)
        pygame.draw.line(surf, col, o2d, tip_2d, 3)
        pygame.draw.circle(surf, col, tip_2d, 4)


def draw_robot_box(surf, R, cx, cy):
    """Wide-and-short box colored so each face is distinguishable."""
    # Corners in body frame. X+ forward, Y+ right, Z+ up.
    box = np.array([
        [-1.8, -1.0, -0.35], [1.8, -1.0, -0.35],
        [1.8,  1.0, -0.35], [-1.8,  1.0, -0.35],
        [-1.8, -1.0,  0.35], [1.8, -1.0,  0.35],
        [1.8,  1.0,  0.35], [-1.8,  1.0,  0.35],
    ])
    rotated = (R @ box.T).T
    pts = [project(v, cx, cy) for v in rotated]
    faces = [
        ([0, 1, 2, 3], (40, 40, 80),   "bottom (Z-)"),
        ([4, 5, 6, 7], (90, 90, 180),  "top (Z+)"),
        ([0, 1, 5, 4], (140, 140, 140), "left (Y-)"),
        ([2, 3, 7, 6], (170, 170, 170), "right (Y+)"),
        ([0, 3, 7, 4], (190, 60, 60),   "tail (X-)"),
        ([1, 2, 6, 5], (60, 190, 60),   "head (X+)"),
    ]
    # Depth-sort: camera frame Z is 'into screen', so larger z == farther away
    ordered = sorted(
        faces,
        key=lambda f: np.mean([np.dot(rotated[v] - _EYE, _CAM_F) for v in f[0]]),
        reverse=True,
    )
    for verts, col, _ in ordered:
        pygame.draw.polygon(surf, col, [pts[v] for v in verts])
        pygame.draw.polygon(surf, (230, 230, 230), [pts[v] for v in verts], 1)


def draw_legs(surf, R, cx, cy, angles):
    """Draw each leg rotated from vertical by its raw servo angle (90 = down)."""
    shoulders = {
        "FL": np.array([ 1.2, -0.9, -0.35]),
        "FR": np.array([ 1.2,  0.9, -0.35]),
        "BL": np.array([-1.2, -0.9, -0.35]),
        "BR": np.array([-1.2,  0.9, -0.35]),
    }
    colors = {"FL": (255, 180, 80), "FR": (255, 120, 120),
              "BL": (120, 200, 255), "BR": (180, 140, 255)}
    names = ["FL", "FR", "BL", "BR"]
    for i, name in enumerate(names):
        h = math.radians(angles[i] - 90)
        leg_local = np.array([math.sin(h) * 1.4, 0.0, -math.cos(h) * 1.4])
        shoulder_w = R @ shoulders[name]
        foot_w = R @ (shoulders[name] + leg_local)
        s2d = project(shoulder_w, cx, cy)
        f2d = project(foot_w, cx, cy)
        pygame.draw.line(surf, colors[name], s2d, f2d, 5)
        pygame.draw.circle(surf, colors[name], f2d, 5)


def draw_graph(surf, rect, data, color, label, font, vmin=-45, vmax=45):
    x, y, w, h = rect
    pygame.draw.rect(surf, (30, 30, 35), rect)
    pygame.draw.rect(surf, (70, 70, 80), rect, 1)
    # zero line
    zy = y + int(h * vmax / (vmax - vmin))
    pygame.draw.line(surf, (60, 60, 65), (x, zy), (x + w, zy), 1)
    draw_text(surf, font, label, (x + 6, y + 3), color)
    if len(data) >= 2:
        pts = []
        for i, v in enumerate(data):
            px = x + int(i * w / data.maxlen)
            clamped = max(vmin, min(vmax, v))
            py = y + int((vmax - clamped) * h / (vmax - vmin))
            pts.append((px, py))
        pygame.draw.lines(surf, color, False, pts, 2)
    if data:
        txt = f"{data[-1]:+6.1f}"
        draw_text(surf, font, txt, (x + w - 70, y + 3), color)


# =============================================================================
# Slider widget
# =============================================================================
class Slider:
    def __init__(self, label, rect, vmin=0, vmax=180, value=90, color=(255, 200, 80)):
        self.label = label
        self.rect = pygame.Rect(rect)
        self.vmin, self.vmax = vmin, vmax
        self.value = value
        self.color = color
        self.dragging = False

    def _value_to_px(self):
        r = self.rect
        track_x = r.x + 40
        track_w = r.width - 110
        frac = (self.value - self.vmin) / (self.vmax - self.vmin)
        return track_x + int(frac * track_w), track_x, track_w

    def _px_to_value(self, mx):
        _, track_x, track_w = self._value_to_px()
        frac = max(0.0, min(1.0, (mx - track_x) / track_w))
        return int(round(self.vmin + frac * (self.vmax - self.vmin)))

    def handle(self, ev) -> bool:
        """Returns True if this slider's value was changed by the event."""
        if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.dragging = True
                new = self._px_to_value(ev.pos[0])
                if new != self.value:
                    self.value = new
                    return True
        elif ev.type == MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == MOUSEMOTION and self.dragging:
            new = self._px_to_value(ev.pos[0])
            if new != self.value:
                self.value = new
                return True
        elif ev.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if self.rect.collidepoint((mx, my)):
                new = max(self.vmin, min(self.vmax, self.value + ev.y))
                if new != self.value:
                    self.value = new
                    return True
        return False

    def nudge(self, delta) -> bool:
        new = max(self.vmin, min(self.vmax, self.value + delta))
        if new != self.value:
            self.value = new
            return True
        return False

    def draw(self, surf, font, device_value=None):
        r = self.rect
        pygame.draw.rect(surf, (28, 30, 36), r)
        pygame.draw.rect(surf, (70, 75, 85), r, 1)
        draw_text(surf, font, self.label, (r.x + 8, r.y + r.height // 2 - 10), self.color)
        knob_x, track_x, track_w = self._value_to_px()
        track_y = r.y + r.height // 2
        pygame.draw.line(surf, (90, 90, 100), (track_x, track_y),
                         (track_x + track_w, track_y), 3)
        # Center marker at 90 deg
        mid_frac = (90 - self.vmin) / (self.vmax - self.vmin)
        mid_x = track_x + int(mid_frac * track_w)
        pygame.draw.line(surf, (120, 120, 140), (mid_x, track_y - 6),
                         (mid_x, track_y + 6), 1)
        # Device's actual value (ghost knob)
        if device_value is not None and device_value != self.value:
            frac = (device_value - self.vmin) / (self.vmax - self.vmin)
            gx = track_x + int(frac * track_w)
            pygame.draw.circle(surf, (110, 110, 120), (gx, track_y), 5)
        # Active knob
        pygame.draw.circle(surf, self.color, (knob_x, track_y), 9)
        pygame.draw.circle(surf, (15, 15, 20), (knob_x, track_y), 9, 2)
        draw_text(surf, font, f"{self.value:3d}",
                  (r.right - 52, r.y + r.height // 2 - 10), self.color)


class Button:
    def __init__(self, label, rect, action):
        self.label = label
        self.rect = pygame.Rect(rect)
        self.action = action
        self.hover = False

    def handle(self, ev):
        if ev.type == MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif (ev.type == MOUSEBUTTONDOWN and ev.button == 1
              and self.rect.collidepoint(ev.pos)):
            self.action()

    def draw(self, surf, font):
        col = (70, 90, 130) if self.hover else (45, 60, 90)
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (100, 120, 160), self.rect, 1, border_radius=6)
        label = font.render(self.label, True, (230, 235, 240))
        surf.blit(label, (self.rect.centerx - label.get_width() // 2,
                          self.rect.centery - label.get_height() // 2))


# =============================================================================
# Main
# =============================================================================
def push_slider_angles():
    with state_lock:
        a = list(state.slider_angles)
    send(f"M:{a[0]},{a[1]},{a[2]},{a[3]}")


def main(args):
    # Comm thread
    if args.serial:
        threading.Thread(target=serial_thread, args=(args.serial,), daemon=True).start()
    else:
        threading.Thread(target=ble_thread, daemon=True).start()

    pygame.init()
    W, H = 1100, 720
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Eshy Controller")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    font_bold = pygame.font.SysFont("monospace", 18, bold=True)
    font_title = pygame.font.SysFont("monospace", 22, bold=True)
    font_sm = pygame.font.SysFont("monospace", 13)

    view_rect = pygame.Rect(20, 60, 520, 360)
    info_rect = pygame.Rect(560, 60, 520, 110)
    slider_area_y = 180
    sliders = [
        Slider("FL (D0)", (560, slider_area_y + 0 * 55, 520, 45),
               color=(255, 180, 80)),
        Slider("FR (D1)", (560, slider_area_y + 1 * 55, 520, 45),
               color=(255, 120, 120)),
        Slider("BL (D2)", (560, slider_area_y + 2 * 55, 520, 45),
               color=(120, 200, 255)),
        Slider("BR (D3)", (560, slider_area_y + 3 * 55, 520, 45),
               color=(180, 140, 255)),
    ]

    last_sent_angles = [90, 90, 90, 90]

    def set_mode_center():
        send("c")
        for s in sliders:
            s.value = 90
        with state_lock:
            state.slider_angles = [90, 90, 90, 90]
        last_sent_angles[:] = [90, 90, 90, 90]  # suppress auto-resend

    def set_mode_sweep():  send("s")
    def set_mode_imu():    send("i")
    def send_manual():     push_slider_angles()

    def snap_to_device():
        with state_lock:
            angles = list(state.device_angles)
        for i, s in enumerate(sliders):
            s.value = angles[i]
            state.slider_angles[i] = angles[i]
        last_sent_angles[:] = angles

    btn_y = 642
    buttons = [
        Button("Center (c)",  (20,  btn_y, 170, 40), set_mode_center),
        Button("Sweep (s)",   (200, btn_y, 170, 40), set_mode_sweep),
        Button("IMU-only (i)",(380, btn_y, 170, 40), set_mode_imu),
        Button("Send sliders",(560, btn_y, 170, 40), send_manual),
        Button("Snap to robot",(740, btn_y, 190, 40), snap_to_device),
    ]

    last_send_time = 0.0
    selected_slider = 0

    running = True
    while running:
        # --- events ---
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False; break
            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False; break
                if ev.key == K_SPACE:
                    set_mode_center()
                elif ev.key == K_s:
                    set_mode_sweep()
                elif ev.key == K_i:
                    set_mode_imu()
                elif ev.key == K_r:
                    snap_to_device()
                elif ev.key in (K_1, K_2, K_3, K_4):
                    selected_slider = [K_1, K_2, K_3, K_4].index(ev.key)
                elif ev.key in (K_LEFT, K_RIGHT):
                    step = (5 if (ev.mod & KMOD_SHIFT) else 1) * (1 if ev.key == K_RIGHT else -1)
                    if 0 <= selected_slider < 4 and sliders[selected_slider].nudge(step):
                        with state_lock:
                            state.slider_angles[selected_slider] = sliders[selected_slider].value

            for s in sliders:
                if s.handle(ev):
                    with state_lock:
                        state.slider_angles[sliders.index(s)] = s.value
            for b in buttons:
                b.handle(ev)

        # Send slider changes, at most 20 Hz.
        now = time.time()
        with state_lock:
            cur_angles = list(state.slider_angles)
        if cur_angles != last_sent_angles and (now - last_send_time) > 0.05:
            push_slider_angles()
            last_sent_angles = cur_angles
            last_send_time = now

        # --- draw ---
        screen.fill((20, 22, 28))

        # Title
        draw_text(screen, font_title, "Eshy Controller", (20, 15),
                  (235, 240, 250))
        with state_lock:
            connected = state.connected
            mode_label = state.mode_label
            r, p, y = state.roll, state.pitch, state.yaw
            dev_angles = list(state.device_angles)
            r_hist = state.roll_hist
            p_hist = state.pitch_hist
            y_hist = state.yaw_hist

        status_col = (80, 220, 120) if connected else (230, 90, 90)
        status_text = f"{mode_label}" if connected else "waiting for device..."
        draw_text(screen, font_bold, status_text, (280, 20), status_col)

        # 3D view panel
        pygame.draw.rect(screen, (14, 15, 20), view_rect)
        pygame.draw.rect(screen, (60, 65, 80), view_rect, 1)
        cx, cy = view_rect.centerx, view_rect.centery + 20
        R = rotation_matrix(r, p, y)
        draw_robot_box(screen, R, cx, cy)
        draw_legs(screen, R, cx, cy, dev_angles)
        draw_axis_triad(screen, R, view_rect.x + 55, view_rect.bottom - 55)
        # Axis legend
        draw_text(screen, font_sm, "X+ head   Y+ right   Z+ up",
                  (view_rect.x + 10, view_rect.y + 10), (160, 160, 170))

        # Readouts panel
        pygame.draw.rect(screen, (24, 26, 34), info_rect)
        pygame.draw.rect(screen, (60, 65, 80), info_rect, 1)
        draw_text(screen, font_bold, f"Roll  (X)  {r:+7.1f} deg",
                  (info_rect.x + 14, info_rect.y + 12),  (255, 130, 130))
        draw_text(screen, font_bold, f"Pitch (Y)  {p:+7.1f} deg",
                  (info_rect.x + 14, info_rect.y + 42),  (120, 230, 130))
        draw_text(screen, font_bold, f"Yaw   (Z)  {y:+7.1f} deg",
                  (info_rect.x + 14, info_rect.y + 72),  (140, 180, 255))
        draw_text(screen, font_sm,
                  f"robot angles  FL={dev_angles[0]:3d}  FR={dev_angles[1]:3d}"
                  f"  BL={dev_angles[2]:3d}  BR={dev_angles[3]:3d}",
                  (info_rect.x + 260, info_rect.y + 20), (180, 185, 195))

        # Sliders (highlight the keyboard-selected one)
        for i, s in enumerate(sliders):
            s.draw(screen, font, device_value=dev_angles[i])
            if i == selected_slider:
                pygame.draw.rect(screen, (90, 150, 220), s.rect, 2, border_radius=4)

        # Graphs along the bottom of the 3D view area
        gy = 430
        gh = 60
        draw_graph(screen, (20,  gy,          520, gh), r_hist, (255, 130, 130), "Roll",  font_sm, -90, 90)
        draw_graph(screen, (20,  gy +  gh+8,  520, gh), p_hist, (120, 230, 130), "Pitch", font_sm, -90, 90)
        draw_graph(screen, (20,  gy + 2*(gh+8), 520, gh), y_hist, (140, 180, 255), "Yaw",   font_sm, -180, 180)

        # Buttons
        for b in buttons:
            b.draw(screen, font)

        # Help strip
        help_text = "Drag / wheel sliders   1-4 select   left/right arrow nudge (shift=5)   space=center  s=sweep  i=imu  r=snap"
        draw_text(screen, font_sm, help_text, (20, 696), (140, 145, 155))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None,
                        help="USB serial port (e.g. /dev/cu.usbmodem...). Omit for BLE.")
    args = parser.parse_args()
    main(args)
