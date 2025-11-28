#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;
uniform float time;
uniform int useWingFlap;
uniform vec3 wingCenter;
uniform vec3 wingSize;
uniform float wingAmplitude;
uniform float wingFrequency;

vec3 applyWingFlap(vec3 pos)
{
    if (useWingFlap == 0)
        return pos;

    float halfWidth = max(wingSize.x * 0.5, 0.0001);
    float halfDepth = max(wingSize.z * 0.5, 0.0001);
    float normalizedX = (pos.x - wingCenter.x) / halfWidth;
    float edgeMask = smoothstep(0.15, 0.9, abs(normalizedX));
    float depthMask = 1.0 - smoothstep(0.0, 0.45, abs((pos.z - wingCenter.z) / halfDepth));
    float influence = edgeMask * (0.5 + 0.5 * depthMask);
    if (influence < 0.001)
        return pos;

    float flap = sin(time * wingFrequency);
    float dynAmp = wingAmplitude * max(halfWidth, wingSize.y * 0.6);
    // move ao longo de Z local para virar vertical apos o pitch de -90deg do modelo
    pos.z += flap * dynAmp * influence;
    return pos;
}

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
    vec4 FragPosLightSpace;
} vs_out;

void main()
{
    vec3 animatedPos = applyWingFlap(position);
    vec4 worldPos = model * vec4(animatedPos, 1.0);
    vs_out.FragPos = worldPos.xyz;
    vs_out.Normal = normalize(normalMatrix * normal);
    vs_out.TexCoord = texCoord;
    vs_out.FragPosLightSpace = lightSpaceMatrix * worldPos;

    gl_Position = projection * view * worldPos;
}
