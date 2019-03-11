#include <cuda_runtime.h>
#include <stdio.h>

int main(void){
    struct cudaDeviceProp prop;
    int count, i;
    cudaError_t ret = cudaGetDeviceCount(&count);
    if(ret != cudaSuccess){
        fprintf(stderr, "cudaGetDeviceCount() == %d\n", (int)ret);
        fflush(stderr);
    }else{
        for(i=0;i<count;i++){
            if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
                fprintf(stdout, "%d.%d\n", prop.major, prop.minor);
                fflush(stdout);
            }
        }
    }
    return ret != cudaSuccess;
}
