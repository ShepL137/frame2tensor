#version 330
// The transparent framebuffer is composited with premultiplied alpha,
// so both modes emit (rgb * a, a) directly and no GL blending is needed.

uniform sampler2D u_motion;
uniform int u_mode;         // 0 = highlight, 1 = spotlight
uniform float u_strength;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 m          = clamp(texture(u_motion, v_uv).rgb, 0.0, 1.0);
    float magnitude = max(m.r, max(m.g, m.b));
    if (u_mode == 0) {
        frag_color = vec4(m * u_strength, magnitude * u_strength);
    } else {
        frag_color = vec4(0.0, 0.0, 0.0, u_strength * (1.0 - magnitude));
    }
}
