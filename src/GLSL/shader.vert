#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColour;
layout(location = 1) out vec2 fragTexCoord;

vec3 instanceTrans[5] = vec3[](
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 1.0, 1.0),
    vec3(2.0, 2.0, 2.0),
    vec3(3.0, 3.0, 3.0),
    vec3(4.0, 4.0, 4.0));

void main() {
    mat4 model = ubo.model;
    model[3] = vec4(instanceTrans[gl_InstanceIndex], 1.0f);

    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
    fragColour = inColour;
    fragTexCoord = inTexCoord;
}
