nvcc -o game \
 \
src/*.cxx \
src/Renderer/*.cxx \
src/Simulator/*.cxx \
src/Utils/*.cxx \
src/*.cu \
src/Kernel/*.cu \
src/Test/*.cxx \
src/Test/*.cu \
 \
-I $(PWD)/src \
-I/c/Users/m202-/Desktop/Developer/glm \
-I/c/Users/m202-/Desktop/Developer/tinyobjloader \
-I/c/Users/m202-/Desktop/Developer/glfw-3.3.2.bin.WIN64/include \
-I/c/VulkanSDK/1.2.141.2/Include \
-I/c/Users/m202-/Desktop/Developer/stb \
-L/c/Users/m202-/Desktop/Developer/glfw-3.3.2.bin.WIN64/lib-vc2019 \
-L/c/VulkanSDK/1.2.141.2/Lib \
-lvulkan-1 \
-lglfw3dll
