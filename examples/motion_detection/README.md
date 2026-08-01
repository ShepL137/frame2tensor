# Motion detection

Live frame-differencing motion filters over a captured window.

Two entry points:
- `windowed.py` shows the model output in a separate window (`X11Window` capture, torch detector, `WindowedRenderer`).
- `overlay.py` draws it over the captured window itself, as a transparent, always-on-top, click-through layer, so the filter displays as Augmented Reality while you keep using the application (`X11Window` capture, torch detector, `CUDAWritableTexture`, transparent overlay quad).

## Running

```sh
python examples/motion_detection/windowed.py [WINDOW_ID] [--model {ema,adaptive}] [--fps N]
python examples/motion_detection/overlay.py  [WINDOW_ID] [--model {ema,adaptive}] [--mode {highlight,spotlight}] [--fps N] [--strength S] [--timeout N]
```

`WINDOW_ID` is decimal or `0x`-prefixed hex; omit it to target the active window, or find one with `xwininfo`.
The windowed viewer runs until its window is closed.
The overlay is click-through and cannot be closed by clicking: quit with Esc, or Ctrl+C in the launching terminal.

## Detectors

- `ema`     : motion as absolute difference against an exponential-moving-average background.
- `adaptive`: motion as a per-pixel z-score against running mean and variance. Each pixel is compared to its own noise statistics, which places the visibility threshold exactly at that pixel's noise floor.

Configurable constants in `_detectors.py`; gains assume input normalized to 0..1.

## Sampling rate (`--fps`)

Sampling faster than the source produces new frames which feed the detectors duplicate captures.
This makes it flicker like crazy; it really hurts the eyes.
Tune `--fps` to the source's actual update rate or lower. Undershooting tends to produce better results.

## Overlay modes

- `highlight`: motion glows as a colored translucent layer; static regions stay fully transparent.
- `spotlight`: a dark veil covers the window and motion punches holes in it.

`--strength` sets the highlight opacity, or the spotlight veil darkness. For the latter, a strength of 1.0 produces complete blackness without motion.

### The Esc panic key

A high-strength spotlight can produce a black screen, and the overlay is click-through and unfocusable, so it offers no easy way out.
The overlay therefore grabs the unmodified Esc key globally (`XGrabKey` on the root window).
While it runs, Esc quits it; modified combinations (Ctrl+Esc, Alt+Esc) pass through without quiting.
The grab is registered per lock-state combination (none, Caps Lock, Num Lock, both) rather than with `AnyModifier`, which otherwise conflicts with any window-manager binding involving Escape and fails with `BadAccess`.
If the grab is still unavailable, the overlay warns at startup: highlight mode runs with Ctrl+C in the terminal as the only exit, and spotlight mode refuses to start.
To change or remove the key, edit `grab_escape_key` in the script.

#### Fullscreen warning

A true fullscreen application that holds an active keyboard grab (common for games) receives all key events directly, and passive grabs, like the panic key, never activate.
Spotlight mode therefore also auto-quits after 10 seconds by default as a last-resort failsafe.
`--timeout N` changes it, and `--timeout 0` disables it.

>WARNING: Don't get lost in the dark.

### Overlay mechanics

The overlay window is override-redirect, set before the first map, so the window manager never manages or layers it, and it is restacked to the top every frame.
Transparency, click-through, and no-focus behavior come from GLFW 3.4 hints (`TRANSPARENT_FRAMEBUFFER`, `MOUSE_PASSTHROUGH`, `FLOATING`, `FOCUS_ON_SHOW=false`).
The overlay follows the target's screen position with a 1 Hz re-query. Feel free to increase this.

### Compositor limitations and troubleshooting

- A focused client asserting `_NET_WM_STATE_FULLSCREEN` can still contest the top of the stack; some games assert it even in borderless-windowed mode.
- picom's `unredir-if-possible` disables compositing (and with it, transparency) over true exclusive fullscreen.
- If the spotlight veil stays translucent at full `--strength`, the compositor is likely applying its own opacity rule on top of the overlay's alpha; check the compositor config.
