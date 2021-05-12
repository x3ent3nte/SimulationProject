#include <Renderer/MyGLM.h>

std::ostream& operator<<(std::ostream& os, const glm::uvec4& v) {
    os << v.x << " " << v.y << " " << v.z << " " << v.w;
    return os;
}
