#version 330 core

layout (location = 0) in vec2 inPos;

uniform vec2 center;
uniform float radius;
uniform float aspect;

out vec2 uv;

void main()
{
    vec2 pos = center + vec2(inPos.x * radius, inPos.y * radius * aspect);
    uv = inPos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
