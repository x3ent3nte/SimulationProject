nvcc -o main \
main.cu \
Timer.cxx \
Renderer.cxx \
Kernel/Reduce.cu \
Kernel/Scan.cu \
Kernel/InsertionSort.cu \
Kernel/RadixSort.cu \
Kernel/Agent.cu \
Kernel/ContinuousCollision.cu \
Kernel/MyMath.cu \
Test/TestUtils.cxx \
Test/InsertionSortTest.cu \
-I/c/Users/m202-/Desktop/Developer/glm \
-I/c/Users/m202-/Desktop/Developer/glfw-3.3.2.bin.WIN64/include \
-I/c/VulkanSDK/1.2.141.2/Include \
-L/c/Users/m202-/Desktop/Developer/glfw-3.3.2.bin.WIN64/lib-vc2019 \
-L/c/VulkanSDK/1.2.141.2/Lib \
-lvulkan-1 \
-lglfw3
