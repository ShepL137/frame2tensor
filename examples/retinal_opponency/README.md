# Retinal opponency

Capture an X11 window, run a spatial filter on the GPU, and composite the result to a live window.
A convolutional filter (difference-of-Gaussians) projects each frame into three center-surround opponent channels (luminance, blue-yellow, red-green).

This exercises the toolkit's capture → torch model → windowed-render loop.

## Running

```
python examples/retinal_opponency/windowed.py [WINDOW_ID] [--mode M] [--fps N]
```

`WINDOW_ID` is decimal or `0x`-prefixed hex; it defaults to the active window.
Runs until the display window is closed (or Esc/Q).

## Views

| Key | Mode | Shows |
|-----|------|-------|
| `1` | `opponency` | Channel packed: red-green, luminance, blue-yellow mapped to R, G, B. |
| `2` | `greyscale` | Luminance alone: edge sharpening. |
| `3` | `by` | Blue-yellow channel on a blue-to-yellow gradient. |
| `4` | `rg` | Red-green channel on a green-to-red gradient. |

`Tab` cycles through the views; `Esc` or `Q` quits.

The single-channel views (`by`, `rg`) use a two-color gradient rather than greyscale to highlight contrast in opposite directions (e.g., red on green vs green on red).

## What each view tends to show

- `by` tends to highlight shadows, edges, and foliage. Vegetation in particular stands out strongly.
- `rg` tends to highlight human skin across complexions.
- `greyscale` is a plain contrast-sharpening pass, useful as a baseline to see what the color channels add over luminance alone.

Point the capture at a photo, a video, or a webcam feed and cycle the views to compare.
>NOTE: compression artifacts in images and video, among other things, tend to appear in the examples as blocky noise. I may need to adjust the defaults.

## Tuning

In `_filters.py`:

- `CONTRAST_GAIN` sets how hard the opponent responses saturate before display. The default assumes input normalized to 0..1.
- `CENTER_SIGMA` / `SURROUND_SIGMA` / `KERNEL_SIZE` shape the difference-of-Gaussians.
  A wider surround emphasizes larger structures; a larger kernel resolves the surround more accurately at some compute cost.

## Notes

The opponent color space is the Ruderman decomposition (Ruderman, Cronin, and Chiao, 1998, "Statistics of cone responses to natural images: implications for visual coding").
Everything runs on CUDA; the filter holds only convolution kernels, so it is size-agnostic and a source resize needs no rebuild.
