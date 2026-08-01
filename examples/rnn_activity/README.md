# RNN activity

Live visualization of a PyTorch RNN's per-layer activations, as heatmap tiles.

This is the **multi-texture, single-channel** example: several independent textures written from tensors each frame and drawn through your own shader.
A render target's `write()` takes one full-frame buffer, which a grid of separate layers cannot be expressed as, so this example goes through `CUDAWritableTexture` directly instead.

## Running

```sh
python examples/rnn_activity/windowed.py [--fps N] [--layer-size N] [--ring-speed R]
```

Runs until the window is closed.

## Write path

- Each layer's activation vector is reshaped to a square tile and written into a single-channel (`components=1`) float texture with `CUDAWritableTexture.write()`.
- The textures render as a 2x2 grid of quads through the heatmap shaders in `shaders/`, which map 0..1 onto a blue-cyan-green-yellow-red gradient.
- `TILE_GRID` drives both the grid and the layer count, so raising it to 3 gives nine layers in a 3x3 arrangement with no other changes.

## Model

A frozen stack of randomly initialized RNN-cells, 1024 units per layer by default.

The input is a Gaussian bump on a ring of 64 units, rotating at `--ring-speed`.

## Normalization

Each tile is min-max scaled per layer and per frame, to 0..1.

Activation magnitude declines with depth, so a shared scale would leave the deeper tiles nearly black.
Tiles show relative structure within a layer rather than absolute magnitude.
