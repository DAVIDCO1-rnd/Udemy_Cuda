#include "cuda_utils.cuh"

BlockAndGridDimensions* CalculateBlockAndGridDimensions(int channels, int width, int height)
{
    cudaDeviceProp  prop;
    int device_index = 0; //For now I assume there's only one GPu device
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_index));
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockSize = maxThreadsPerBlock / 2;

    dim3 blockSize;
    dim3 gridSize;

    // Calculate optimal block size, depends on the number of channels in picture
    if (width * height * channels < maxBlockSize)
    {
        blockSize.x = width;
        blockSize.y = height;
    }
    else
    {
        int warpSize = prop.warpSize;
        float dWarp = warpSize / (float)channels;
        int maxSize = (int)(maxBlockSize / (float)channels);

        if (width <= maxSize)
            blockSize.x = width;
        else
        {
            float threadsX = 0.0f;
            while (threadsX < maxSize)
            {
                threadsX += dWarp;

            }
            blockSize.x = (int)threadsX;
        }
        blockSize.y = maxSize / blockSize.x;
        if (blockSize.y == 0)
        {
            blockSize.y = 1;
        }
    }

    //block size 3rd dimension is always the number of channels.
    blockSize.z = channels;

    //calculate grid size. (number of necessary blocks to cover the whole picture) 
    gridSize.x = (int)ceil((double)width / blockSize.x);
    gridSize.y = (int)ceil((double)height / blockSize.y);

    BlockAndGridDimensions* block_and_grid_dimensions = new BlockAndGridDimensions(blockSize, gridSize);
    return block_and_grid_dimensions;
}