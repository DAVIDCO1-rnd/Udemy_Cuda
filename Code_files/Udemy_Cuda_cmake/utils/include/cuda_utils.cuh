#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

class BlockAndGridDimensions {
public:
    dim3 blocksPerGrid;
    dim3 threadsPerBlock;
    BlockAndGridDimensions(dim3 block_sizes, dim3 grid_sizes) {
        blocksPerGrid = grid_sizes;
        threadsPerBlock = block_sizes;
    }
};

BlockAndGridDimensions* CalculateBlockAndGridDimensions(int channels, int width, int height);

