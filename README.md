# Ex3_Parallel-Distributed-Computer-Systems
Implement in CUDA the evolution of an Ising model in two dimensions for a given
number of steps k.
V0. Sequential
V1. GPU with one thread per moment
V2. GPU with one thread computing a block of moments
V3. GPU with multiple thread sharing common input moments


To run the code:
module load gcc/9.4.0-eewq4j6 cuda/11.2.2-kkrwdua
nvcc "file_name" -O3 -o output
./output
