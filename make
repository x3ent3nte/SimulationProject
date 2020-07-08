nvcc -o main \
main.cu \
Timer.cxx \
Kernel/Reduce.cu \
Kernel/Scan.cu \
Kernel/InsertionSort.cu \
Test/TestUtils.cxx \
Test/InsertionSortTest.cu
