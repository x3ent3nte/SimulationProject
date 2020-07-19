nvcc -o main \
main.cu \
Timer.cxx \
Kernel/Reduce.cu \
Kernel/Scan.cu \
Kernel/InsertionSort.cu \
Kernel/RadixSort.cu \
Kernel/Agent.cu \
Kernel/ContinuousCollision.cu \
Test/TestUtils.cxx \
Test/InsertionSortTest.cu
