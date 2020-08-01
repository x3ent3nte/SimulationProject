nvcc -o main \
main.cu \
Timer.cxx \
Kernel/Reduce.cu \
Kernel/Scan.cu \
Kernel/InsertionSort.cu \
Kernel/RadixSort.cu \
Kernel/Agent.cu \
Kernel/ContinuousCollision.cu \
Kernel/MyMath.cu \
Test/TestUtils.cxx \
Test/InsertionSortTest.cu \
-I/c/VulkanSDK/1.2.141.2/Include \
-L/c/VulkanSDK/1.2.141.2/Lib \
-lvulkan-1
