#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_uv;

void main() {
    v_uv        = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
