#version 330 core

in VS_OUT {
    vec3 fragPos;
    vec3 fragNormal;
    vec2 uv;
    vec4 lightSpacePos;
} fs_in;

uniform vec3 viewPos;
uniform vec3 sunDirection;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 skyColor;
uniform vec3 fogColor;
uniform vec3 objectColor;
uniform float shininess;
uniform float fogDensity;
uniform float fogBaseDensity;
uniform float fogHeightFalloff;
uniform float fogAnisotropy;
uniform float lightRadiusUV;
uniform float minBias;
uniform float maxBias;
uniform int fogSteps;
uniform int blockerSamples;
uniform int pcfSamples;
uniform int useTexture;

uniform sampler2D diffuseMap;
uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

out vec4 FragColor;

const vec2 POISSON_DISK[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

float computeBias(vec3 normal, vec3 lightDir)
{
    return mix(minBias, maxBias, 1.0 - max(dot(normal, lightDir), 0.0));
}

float computeShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.z > 1.0 || projCoords.z < 0.0)
        return 0.0;

    float currentDepth = projCoords.z;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

    float bias = computeBias(normal, lightDir);

    // 1) Busca de bloqueadores
    float blockerSum = 0.0;
    float blockerCount = 0.0;
    int blockerMax = clamp(blockerSamples, 1, 16);
    for (int i = 0; i < 16; ++i) {
        if (i >= blockerMax) break;
        vec2 offset = POISSON_DISK[i] * lightRadiusUV;
        float closestDepth = texture(shadowMap, projCoords.xy + offset * texelSize).r;
        if (currentDepth - bias > closestDepth) {
            blockerSum += closestDepth;
            blockerCount += 1.0;
        }
    }
    if (blockerCount == 0.0)
        return 0.0;

    float avgBlocker = blockerSum / blockerCount;
    float penumbra = (currentDepth - avgBlocker) * lightRadiusUV / max(avgBlocker, 0.0001);

    // 2) PCF adaptativo
    float shadow = 0.0;
    float samples = 0.0;
    int pcfMax = clamp(pcfSamples, 1, 16);
    for (int i = 0; i < 16; ++i) {
        if (i >= pcfMax) break;
        vec2 offset = POISSON_DISK[i] * (penumbra + lightRadiusUV);
        float closestDepth = texture(shadowMap, projCoords.xy + offset * texelSize).r;
        shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
        samples += 1.0;
    }
    shadow = (samples > 0.0) ? shadow / samples : 0.0;
    return shadow;
}


float phaseHG(float cosTheta, float g)
{
    float denom = pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
    return (1.0 - g * g) / max(denom, 0.0001);
}


vec3 applyVolumetricFog(vec3 lighting, vec3 normal, vec3 lightDir)
{
    if (fogSteps <= 0) {
        float distanceToCamera = length(viewPos - fs_in.fragPos);
        float fogFactor = 1.0 - exp(-pow(distanceToCamera * fogDensity, 2.0));
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        return mix(lighting, fogColor, fogFactor);
    }

    const int MAX_STEPS = 16;
    float steps = float(clamp(fogSteps, 1, MAX_STEPS));
    vec3 dir = normalize(fs_in.fragPos - viewPos);
    float totalDist = length(fs_in.fragPos - viewPos);
    float stepLen = totalDist / steps;

    float transmittance = 1.0;
    vec3 scattering = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (i >= fogSteps) break;
        float t = (float(i) + 0.5) * stepLen;
        vec3 samplePos = viewPos + dir * t;
        float density = fogBaseDensity * exp(-samplePos.y * fogHeightFalloff);
        float sigma = density * stepLen;
        float stepTrans = exp(-sigma);

        vec4 lightSpace = lightSpaceMatrix * vec4(samplePos, 1.0);
        float shadow = computeShadow(lightSpace, normal, lightDir);
        float phase = phaseHG(dot(dir, -lightDir), fogAnisotropy);
        vec3 Li = lightColor * (1.0 - shadow);

        scattering += transmittance * (1.0 - stepTrans) * phase * Li;
        transmittance *= stepTrans;
    }

    return lighting * transmittance + fogColor * scattering;
}


void main()
{
    vec3 normal = normalize(fs_in.fragNormal);
    vec3 lightDir = normalize(-sunDirection);
    vec3 viewDir = normalize(viewPos - fs_in.fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);

    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    vec3 albedo = objectColor;
    if (useTexture == 1)
    {
        albedo = texture(diffuseMap, fs_in.uv).rgb;
    }

    float shadow = computeShadow(fs_in.lightSpacePos, normal, lightDir);
    vec3 ambient = ambientColor * albedo;
    vec3 diffuse = lightColor * albedo * diff;
    vec3 specular = lightColor * spec * 0.35;

    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
    vec3 finalColor = applyVolumetricFog(lighting, normal, lightDir);
    FragColor = vec4(finalColor, 1.0);
}
