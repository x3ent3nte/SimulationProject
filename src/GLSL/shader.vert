#version 460
#extension GL_ARB_separate_shader_objects : enable

layout (binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPosition;
} ubo;

struct AgentRenderInfo {
    int typeId;
    vec3 position;
    vec4 rotation;
};

readonly layout (std430, binding = 1) buffer agentsBuffer {
    AgentRenderInfo agents[];
};

layout (push_constant) uniform InfoConstant {
    uint agentsOffset;
} info;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inColour;
layout (location = 3) in vec2 inTexCoord;

layout (location = 0) out vec3 fragColour;
layout (location = 1) out vec2 fragTexCoord;
layout (location = 2) out vec3 fragNormal;
layout (location = 3) out vec3 fragPosition;
layout (location = 4) out vec3 fragCameraPosition;

vec4 hamiltonProduct(vec4 a, vec4 b) {
    const float r = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z);
    const float i = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y);
    const float j = (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x);
    const float k = (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w);

    return vec4(i, j, k, r);
}

vec4 inverseQuaternion(vec4 q) {
    return vec4(-q.x, -q.y, -q.z, q.w);
}

vec3 rotatePointByQuaternion(vec3 p, vec4 q) {
    const vec4 p4 = vec4(p, 0.0f);
    const vec4 qi = inverseQuaternion(q);

    return hamiltonProduct(hamiltonProduct(q, p4), qi).xyz;
}

void main() {
    const uint agentIndex = gl_InstanceIndex + info.agentsOffset;

    const AgentRenderInfo agent = agents[agentIndex];

    mat4 model = mat4(1.0f);
    model[3] = vec4(agent.position, 1.0f);

    const vec4 worldPosition = model * vec4(rotatePointByQuaternion(inPosition, agent.rotation), 1.0);
    gl_Position = ubo.proj * ubo.view * worldPosition;

    fragColour = inColour;
    fragTexCoord = inTexCoord;
    fragNormal = rotatePointByQuaternion(inNormal, agent.rotation);
    fragPosition = worldPosition.xyz;
    fragCameraPosition = ubo.cameraPosition;
}
