"""Live AR overlay: motion detection drawn over the captured window itself.

The model output is drawn in a transparent, always-on-top, click-through window over the captured window,
so the filter acts as an AR layer while interacting with the source application.

Pipeline: X11Window capture -> torch motion detector -> CUDAWritableTexture -> transparent overlay quad.

Quit via Esc, or Ctrl+C in the terminal.

Spotlight mode refuses to start when the Esc grab is unavailable, and auto-quits after 10 seconds by default
    (--timeout N changes it, 0 disables it at your own risk).
Avoid spotlight over true fullscreen applications:
    an active keyboard grab there can subvert the panic key, leaving no visible way out until the timeout.
"""
import argparse
import struct
import time
from pathlib import Path

import glfw
import moderngl as mgl
import torch
from _detectors import MODELS
from Xlib import XK, X
from Xlib import display as xdisplay
from Xlib import error as xerror

from frame2tensor import CUDAWritableTexture
from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.exceptions import CaptureSourceLostError

SMOOTH_ALPHA      = 0.5  # EMA on the displayed motion map; damps blink without much trail
POSITION_POLL_SEC = 1.0  # how often to re-query the target's screen position and follow it

# Spotlight auto-quit default: the last-resort failsafe for when the Esc panic key is defeated
# (a true fullscreen application holding an active keyboard grab receives all keys directly).
DEFAULT_SPOTLIGHT_TIMEOUT_SEC = 10.0

SHADER_DIR = Path(__file__).parent / "shaders"

# Fullscreen quad; texcoord v flipped so tensor row 0 (top-down capture) appears at the visual top.
QUAD = [
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
     1.0,  1.0,  1.0, 0.0,
    -1.0, -1.0,  0.0, 1.0,
     1.0,  1.0,  1.0, 0.0,
    -1.0,  1.0,  0.0, 0.0,
]

MODE_IDS = {"highlight": 0, "spotlight": 1}


# -----------------------------------------------------------------------------
# Overlay window and GL plumbing

def create_overlay_window(xlib_dpy, origin, size, vsync):
    """Create the overlay as an override-redirect window so the WM never manages or layers it.

    The overlay is parked at `origin` (absolute screen x, y) with pixel dimensions `size` (width, height).
    Returns (glfw_window, xlib_window_resource); the caller restacks the xlib resource every frame.
    """
    x, y          = origin
    width, height = size
    if not glfw.init():
        raise RuntimeError("glfw.init() failed.")

    glfw.window_hint(hint=glfw.CLIENT_API,              value=glfw.OPENGL_API)
    glfw.window_hint(hint=glfw.CONTEXT_CREATION_API,    value=glfw.NATIVE_CONTEXT_API)
    glfw.window_hint(hint=glfw.RESIZABLE,               value=glfw.FALSE)
    glfw.window_hint(hint=glfw.DECORATED,               value=glfw.FALSE)
    glfw.window_hint(hint=glfw.FLOATING,                value=glfw.TRUE)
    glfw.window_hint(hint=glfw.FOCUS_ON_SHOW,           value=glfw.FALSE)
    glfw.window_hint(hint=glfw.TRANSPARENT_FRAMEBUFFER, value=glfw.TRUE)
    glfw.window_hint(hint=glfw.MOUSE_PASSTHROUGH,       value=glfw.TRUE)
    # Stay unmapped until override_redirect is set below
    # The WM only fails to adopt the window if the attribute is already set at map time.
    glfw.window_hint(hint=glfw.VISIBLE,                 value=glfw.FALSE)

    win = glfw.create_window(width=width, height=height, title="Motion Overlay - frame2tensor", monitor=None, share=None)  # noqa: E501
    if not win:
        raise RuntimeError("GLFW overlay window creation failed.")

    overlay_xid = glfw.get_x11_window(win)
    overlay_win = xlib_dpy.create_resource_object("window", overlay_xid)
    overlay_win.change_attributes(override_redirect=1)
    xlib_dpy.sync()

    glfw.set_window_pos(win, x, y)
    glfw.show_window(win)
    glfw.set_window_attrib(win, glfw.MOUSE_PASSTHROUGH, glfw.TRUE)
    glfw.make_context_current(win)
    glfw.swap_interval(1 if vsync else 0)

    return win, overlay_win


def target_origin(root, target):
    """Absolute screen position of the target window's top-left corner."""
    reply = root.translate_coords(target, 0, 0)

    return reply.x, reply.y


def follow_target(win, root, target, origin, pos_clock):
    """Follow the target at 1 Hz: re-query its screen origin and move the overlay when it changed."""
    now = time.perf_counter()
    if now - pos_clock < POSITION_POLL_SEC:
        return origin, pos_clock

    new_origin = target_origin(root, target)
    if new_origin != origin:
        glfw.set_window_pos(win, *new_origin)

    return new_origin, now


# -----------------------------------------------------------------------------
# Panic key

# Esc is grabbed per lock-state combination rather than with AnyModifier:
# an AnyModifier grab conflicts with any existing grab involving the same key
# (e.g. a WM's Alt+Escape binding) and the whole request fails with BadAccess.
LOCK_MASKS = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)  # none, CapsLock, NumLock, both


def grab_escape_key(xlib_dpy, root):
    """Globally grab unmodified Esc as the panic key; returns its keycode, or None if grabbing failed.

    The grab is global: while the overlay runs, a plain Esc quits it from anywhere,
    and the focused application does not receive the key.
    """
    keycode = xlib_dpy.keysym_to_keycode(XK.XK_Escape)
    catches = []
    for mask in LOCK_MASKS:
        catch = xerror.CatchError(xerror.BadAccess)
        root.grab_key(
            key=keycode, modifiers=mask, owner_events=False,
            pointer_mode=X.GrabModeAsync, keyboard_mode=X.GrabModeAsync, onerror=catch,
        )
        catches.append(catch)

    xlib_dpy.sync()
    if any(catch.get_error() for catch in catches):
        release_escape_key(xlib_dpy, root, keycode)
        return None

    return keycode


def release_escape_key(xlib_dpy, root, keycode):
    """Release the panic-key grabs; a no-op when the grab never armed."""
    if keycode is None:
        return

    for mask in LOCK_MASKS:
        root.ungrab_key(key=keycode, modifiers=mask)

    xlib_dpy.sync()


def arm_panic_key(xlib_dpy, root, mode):
    """Arm the Esc panic key, failing loudly when it cannot be armed.

    The overlay is click-through and can never take focus,
    so a high-strength spotlight veil over a fullscreen window would otherwise leave no visible way out.
    Spotlight mode therefore refuses to run without the panic key.
    """
    keycode = grab_escape_key(xlib_dpy, root)
    if keycode is not None:
        return keycode

    print("WARNING: could not grab Esc (another client holds a conflicting grab); no panic key armed.")

    if mode == "spotlight":
        print("Refusing to run spotlight mode without a panic key: the veil could leave no visible way out.")
        print("Free the Esc grab or use highlight mode.")
        raise SystemExit(1)

    print("Ctrl+C in this terminal is the only way to quit.")

    return None


def escape_pressed(xlib_dpy, keycode):
    """Drain pending X events; True if the grabbed panic key was pressed."""
    if keycode is None:
        return False

    pressed = False
    while xlib_dpy.pending_events():
        event = xlib_dpy.next_event()
        if event.type == X.KeyPress and event.detail == keycode:
            pressed = True

    return pressed


# -----------------------------------------------------------------------------
# Rendering


def build_size_state(ctx, model_name, width, height):
    """(Re)build everything sized to the source: texture, interop handle, model, smoothing state."""
    tex      = ctx.texture(size=(width, height), components=4, dtype="f4")
    cuda_tex = CUDAWritableTexture(texture=tex, width=width, height=height, components=4)
    model    = MODELS[model_name](height, width)
    smoothed = torch.zeros(height, width, 4, device="cuda")

    return tex, cuda_tex, model, smoothed


def build_pipeline(ctx, mode, strength):
    """Compile the overlay shaders and wire the fullscreen quad; returns the ready-to-render VAO."""
    prog = ctx.program(
        vertex_shader=(SHADER_DIR / "overlay.vert").read_text(),
        fragment_shader=(SHADER_DIR / "overlay.frag").read_text(),
    )
    vbo = ctx.buffer(data=struct.pack(f"{len(QUAD)}f", *QUAD))
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "in_position", "in_texcoord")])

    prog["u_motion"]   = 0
    prog["u_mode"]     = MODE_IDS[mode]
    prog["u_strength"] = strength

    return vao


def release_size_state(tex, cuda_tex):
    """Release the interop handle and its GL texture, in that order."""
    cuda_tex.close()
    tex.release()


def draw_overlay(ctx, vao, tex, overlay_win, win):
    """Restack above the target and composite the motion texture into the transparent framebuffer."""
    overlay_win.configure(stack_mode=X.Above)
    ctx.clear(0.0, 0.0, 0.0, 0.0)
    tex.use(location=0)
    vao.render(mgl.TRIANGLES)
    glfw.swap_buffers(win)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def limit_framerate(t0, fps):
    """Limit framerate to `fps`.

    Args:
        t0 : Frame start.
        fps: Frames per second. 0 for vsync, negative to disable.
    """
    if fps <= 0:
        return

    budget = 1.0 / fps - (time.perf_counter() - t0)
    if budget > 0:
        time.sleep(budget)


class FpsReporter:
    """Once per second, rewrite the terminal status line with the measured frame rate."""

    def __init__(self):
        self.frame_count = 0
        self.clock       = time.perf_counter()

    def tick(self):
        self.frame_count += 1

        now = time.perf_counter()
        if now - self.clock >= 1.0:
            print(f"\r{self.frame_count / (now - self.clock):.0f} fps   ", end="", flush=True)
            self.clock       = now
            self.frame_count = 0


def resolve_timeout(args):
    """Spotlight defaults to a short auto-quit as the last-resort failsafe; highlight runs untimed."""
    if args.timeout is not None:
        return args.timeout

    return DEFAULT_SPOTLIGHT_TIMEOUT_SEC if args.mode == "spotlight" else 0


def connect_x(window_id, mode):
    """Open the X connection, resolve the root and target windows, and arm the panic key."""
    xlib_dpy      = xdisplay.Display()
    root          = xlib_dpy.screen().root
    target        = xlib_dpy.create_resource_object("window", window_id)
    panic_keycode = arm_panic_key(xlib_dpy, root, mode)

    return xlib_dpy, root, target, panic_keycode


def print_banner(args, window_id, src, origin, timeout):
    """Announce the session and spell out every exit: Esc, Ctrl+C, and the spotlight timeout."""
    print(f"Overlaying {args.mode} ({args.model}) on 0x{window_id:x}  {src.width}x{src.height} at {origin}")
    print("The overlay is click-through; press Esc anywhere (or Ctrl+C here) to quit.")

    if args.mode != "spotlight":
        return
    if timeout:
        print(f"Spotlight auto-quits after {timeout:.0f} s (--timeout N changes it, 0 disables it at your own risk).")
    else:
        print("Spotlight timeout disabled; Esc and Ctrl+C are the only exits.")

    print("Avoid true fullscreen targets: an active keyboard grab there can defeat the Esc panic key.")


def cleanup(xlib_dpy, panic_keycode, win, tex, cuda_tex):
    """Release the key grabs, the interop and GL resources, and the display connection."""
    release_escape_key(xlib_dpy, xlib_dpy.screen().root, panic_keycode)
    # Interop cleanup targets the current GL context; make ours current before releasing.
    glfw.make_context_current(win)
    release_size_state(tex, cuda_tex)
    glfw.destroy_window(win)
    # No glfw.terminate(): the capture context still owns a live GLFW window.
    xlib_dpy.close()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex); defaults to the active window",
    )
    p.add_argument(
        "--model", choices=MODELS, default="ema",
        help="Detector: ema differences against a moving-average background,"
             " adaptive z-scores each pixel (default: ema)",
    )
    p.add_argument(
        "--mode", choices=MODE_IDS, default="highlight",
        help="highlight glows over motion;"
             " spotlight veils the window and motion punches through (default: highlight)",
    )
    p.add_argument(
        "--fps", type=int, default=24,
        help="Frame limiter; match the source's real update rate (or lower). 0 uses vsync instead (default: 24)",
    )
    p.add_argument(
        "--strength", type=float, default=0.8,
        help="Highlight opacity, or spotlight veil darkness (default: 0.8)",
    )
    p.add_argument(
        "--timeout", type=float, default=None,
        help="Auto-quit after N seconds; 0 disables. Default: 10 in spotlight mode (safety), 0 in highlight",
    )

    return p.parse_args()


def main():
    args      = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()
    vsync     = args.fps == 0
    timeout   = resolve_timeout(args)

    xlib_dpy, root, target, panic_keycode = connect_x(window_id, args.mode)

    with X11Window(window_id) as src:
        origin           = target_origin(root, target)
        win, overlay_win = create_overlay_window(xlib_dpy, origin, (src.width, src.height), vsync)
        ctx              = mgl.create_context()
        vao              = build_pipeline(ctx, args.mode, args.strength)

        tex, cuda_tex, model, smoothed = build_size_state(ctx, args.model, src.width, src.height)
        seen_revision = src.revision

        print_banner(args, window_id, src, origin, timeout)

        fps_report = FpsReporter()
        pos_clock  = time.perf_counter()
        deadline   = time.perf_counter() + timeout if timeout else None
        try:
            while True:
                t0 = time.perf_counter()
                if deadline is not None and t0 >= deadline:
                    print("\nTimeout reached (--timeout).")
                    break
                glfw.poll_events()
                if escape_pressed(xlib_dpy, panic_keycode):
                    print("\nEsc pressed.")
                    break

                try:
                    buf = src.capture()
                except CaptureSourceLostError:
                    print("\nSource window closed.")
                    break

                glfw.make_context_current(win)
                if src.revision != seen_revision:
                    seen_revision = src.revision
                    release_size_state(tex, cuda_tex)
                    glfw.set_window_size(win, src.width, src.height)
                    tex, cuda_tex, model, smoothed = build_size_state(ctx, args.model, src.width, src.height)

                frame    = torch.as_tensor(buf, device="cuda").float() / 255.0
                smoothed = SMOOTH_ALPHA * model(frame) + (1 - SMOOTH_ALPHA) * smoothed
                cuda_tex.write(smoothed.contiguous())

                draw_overlay(ctx, vao, tex, overlay_win, win)

                origin, pos_clock = follow_target(win, root, target, origin, pos_clock)
                limit_framerate(t0, args.fps)
                fps_report.tick()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cleanup(xlib_dpy, panic_keycode, win, tex, cuda_tex)


if __name__ == "__main__":
    main()
