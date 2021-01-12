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
    if (magMult <= 0.0f) {
        return 0.0f;
    }

    return dot(a, b) / magMult;
}

void main() {
    vec3 lightVector = vec3(0.0f, 0.0f, 0.0f) - fragPosition;
    float lightCosSim = max(cosineSimilarity(fragNormal, lightVector), 0.0f);
    vec3 lightColour = vec3(0.9, 0.9, 0.9);
    vec3 diffuse = lightCosSim * lightColour;

    vec3 ambient = vec3(0.04f, 0.04f, 0.04f);

    vec3 light = ambient + diffuse;

    //outColour = vec4(light * fragColour * texture(texSampler, fragTexCoord).rgb, 1.0f);
    outColour = vec4(light * fragColour * vec3(0.3, 0.0, 0.99), 1.0f);
}
