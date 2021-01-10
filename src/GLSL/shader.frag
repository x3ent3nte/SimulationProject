#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragPosition;

layout(location = 0) out vec4 outColour;

float cosineSimilarity(vec3 a, vec3 b) {
    float magMult = length(a) * length(b);
    if (magMult == 0.0f) {
        return 0.0f;
    }

    return dot(a, b) / magMult;
}

void main() {
    vec3 lightVector = vec3(0.0f, 0.0f, 0.0f) - fragPosition;
    float lightCosSim = cosineSimilarity(fragNormal, lightVector);
    if (lightCosSim < 0.0f) {
        lightCosSim = 0.0f;
    }
    vec3 light = vec3(0.1, 0.0, 0.2) + (vec3(0.2, 0.0, 0.7) * lightCosSim);

    outColour = vec4(light * fragColour * texture(texSampler, fragTexCoord).rgb, 1.0f);
}
