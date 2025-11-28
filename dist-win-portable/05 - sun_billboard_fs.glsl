#version 330 core

in vec2 uv;
out vec4 FragColor;

uniform vec3 sunColor;

void main()
{
    // dist em UV, mas usamos inPos em [-1,1] para mask circular
    float r = length(uv * 2.0 - 1.0);      // 0 no centro, ~1.414 no canto
    float radial = clamp(r / 1.2, 0.0, 1.0); // normaliza para ~1 na borda

    float core = smoothstep(0.2, 0.0, radial);
    float glow = smoothstep(0.85, 0.15, radial);

    float mask = 1.0 - smoothstep(0.95, 1.05, radial); // zera fora do disco

    vec3 color = sunColor * (core * 1.4 + glow * 0.6) * mask;
    FragColor = vec4(color, 1.0);
}
