#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColour;

void main() {
    outColour = vec4(vec3(0.4, 0.0, 0.6) * fragColour * texture(texSampler, fragTexCoord).rgb, 1.0f);
}
