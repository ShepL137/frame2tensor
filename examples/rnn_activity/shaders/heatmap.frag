#version 330
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;

vec3 heatmap_color(float value) {
    value = clamp(value, 0.0, 1.0);
    if (value < 0.25) {
        float t = value / 0.25;
        return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);
    } else if (value < 0.5) {
        float t = (value - 0.25) / 0.25;
        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else if (value < 0.75) {
        float t = (value - 0.5) / 0.25;
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t);
    } else {
        float t = (value - 0.75) / 0.25;
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t);
    }
}

void main() {
    float activation = texture(u_texture, v_texcoord).r;
    vec3 color       = heatmap_color(activation);
    fragColor        = vec4(color, 1.0);
}
