#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (binding = 2) uniform sampler2D texSampler[];

layout (location = 0) in vec3 fragColour;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in vec3 fragNormal;
layout (location = 3) in vec3 fragPosition;
layout (location = 4) in flat vec3 fragCameraPosition;
layout (location = 5) in flat uint typeId;

layout (location = 0) out vec4 outColour;

vec3 safeNormalize(vec3 v) {
    float mag = length(v);
    if (mag <= 0.0f) {
        return vec3(1.0f, 0.0f, 0.0f);
    } else {
        return v / mag;
    }
}

void main() {
    vec3 lightColour = vec3(1.0f, 1.0f, 1.0f);

    float ambientStrength = 0.001f;
    vec3 ambient = ambientStrength * lightColour;

    vec3 normalizedNormal = safeNormalize(fragNormal);
    vec3 sunPosition = vec3(10000, 300, 10000);
    vec3 lightVector = safeNormalize(sunPosition - fragPosition);
    float lightCosSim = max(dot(normalizedNormal, lightVector), 0.0f);
    vec3 diffuse = lightCosSim * lightColour;

    float specularStrength = 0.5;
    vec3 viewVector = safeNormalize(fragCameraPosition - fragPosition);
    vec3 reflectVector = reflect(-lightVector, normalizedNormal);

    float specularFocus = pow(max(dot(viewVector, reflectVector), 0.0f), 32);
    vec3 specular = specularStrength * specularFocus * lightColour;

    vec3 light = ambient + diffuse + specular;

    outColour = vec4(light * fragColour * texture(texSampler[typeId], fragTexCoord).rgb, 1.0f);
    //outColour = vec4(light * fragColour * vec3(0.35, 0.05, 0.99), 1.0f);
}
