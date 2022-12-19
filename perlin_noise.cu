#include <stdio.h>
#include <math.h>
#include <time.h>
#include "perlin_noise.cuh"
#include "bitmap.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

static int SEED = 0;

static int hash[] = {208,34,231,213,32,248,233,56,161,78,24,140,71,48,140,254,245,255,247,247,40,
                     185,248,251,245,28,124,204,204,76,36,1,107,28,234,163,202,224,245,128,167,204,
                     9,92,217,54,239,174,173,102,193,189,190,121,100,108,167,44,43,77,180,204,8,81,
                     70,223,11,38,24,254,210,210,177,32,81,195,243,125,8,169,112,32,97,53,195,13,
                     203,9,47,104,125,117,114,124,165,203,181,235,193,206,70,180,174,0,167,181,41,
                     164,30,116,127,198,245,146,87,224,149,206,57,4,192,210,65,210,129,240,178,105,
                     228,108,245,148,140,40,35,195,38,58,65,207,215,253,65,85,208,76,62,3,237,55,89,
                     232,50,217,64,244,157,199,121,252,90,17,212,203,149,152,140,187,234,177,73,174,
                     193,100,192,143,97,53,145,135,19,103,13,90,135,151,199,91,239,247,33,39,145,
                     101,120,99,3,186,86,99,41,237,203,111,79,220,135,158,42,30,154,120,67,87,167,
                     135,176,183,191,253,115,184,21,233,58,129,233,142,39,128,211,118,137,139,255,
                     114,20,218,113,154,27,127,246,250,1,8,198,250,209,92,222,173,21,88,102,219};

int noise2(int x, int y) {
    int tmp = hash[(y + SEED) % 256];
    return hash[(tmp + x) % 256];
}

// Get noise value for grid corner points
__device__
int noise2_gpu(int x, int y) {
    static int hash[] = {208,34,231,213,32,248,233,56,161,78,24,140,71,48,140,254,245,255,247,247,40,
                     185,248,251,245,28,124,204,204,76,36,1,107,28,234,163,202,224,245,128,167,204,
                     9,92,217,54,239,174,173,102,193,189,190,121,100,108,167,44,43,77,180,204,8,81,
                     70,223,11,38,24,254,210,210,177,32,81,195,243,125,8,169,112,32,97,53,195,13,
                     203,9,47,104,125,117,114,124,165,203,181,235,193,206,70,180,174,0,167,181,41,
                     164,30,116,127,198,245,146,87,224,149,206,57,4,192,210,65,210,129,240,178,105,
                     228,108,245,148,140,40,35,195,38,58,65,207,215,253,65,85,208,76,62,3,237,55,89,
                     232,50,217,64,244,157,199,121,252,90,17,212,203,149,152,140,187,234,177,73,174,
                     193,100,192,143,97,53,145,135,19,103,13,90,135,151,199,91,239,247,33,39,145,
                     101,120,99,3,186,86,99,41,237,203,111,79,220,135,158,42,30,154,120,67,87,167,
                     135,176,183,191,253,115,184,21,233,58,129,233,142,39,128,211,118,137,139,255,
                     114,20,218,113,154,27,127,246,250,1,8,198,250,209,92,222,173,21,88,102,219};

    static int SEED = 0;
    
    int tmp = hash[(y + SEED) % 256];
    return hash[(tmp + x) % 256];
}

float lin_inter(float x, float y, float s) {
    return x + s * (y-x);
}

float smooth_inter(float x, float y, float s) {
    return lin_inter(x, y, s * s * (3-2*s));
}

// Linear interpolation used in the smooth interpolation
__device__
float lin_inter_gpu(float x, float y, float s) {
    return x + s * (y-x);
}

// Smooth interpolation for smoother transition
__device__
float smooth_inter_gpu(float x, float y, float s) {
    return lin_inter_gpu(x, y, s * s * (3-2*s));
}

float noise2d(float x, float y) {
    int x_int = x;
    int y_int = y;
    float x_frac = x - x_int;
    float y_frac = y - y_int;
    int s = noise2(x_int, y_int);
    int t = noise2(x_int+1, y_int);
    int u = noise2(x_int, y_int+1);
    int v = noise2(x_int+1, y_int+1);
    float low = smooth_inter(s, t, x_frac);
    float high = smooth_inter(u, v, x_frac);
    return smooth_inter(low, high, y_frac);
}

// Calculate noise value for target point by calculating one octave on the gpu
__device__
float noise2d_gpu(float x, float y) {
    int x_int = x;
    int y_int = y;
    float x_frac = x - x_int;
    float y_frac = y - y_int;
    int s = noise2_gpu(x_int, y_int);
    int t = noise2_gpu(x_int+1, y_int);
    int u = noise2_gpu(x_int, y_int+1);
    int v = noise2_gpu(x_int+1, y_int+1);
    float low = smooth_inter_gpu(s, t, x_frac);
    float high = smooth_inter_gpu(u, v, x_frac);
    return smooth_inter_gpu(low, high, y_frac);
}

// Calculate the noise value for one point by calculating all octave values and adding them according to the amplitude
float perlin2d(float x, float y, float freq, int octaves) {
    float xa = x*freq;
    float ya = y*freq;
    float amp = 1.0;
    float fin = 0;
    float div = 0.0;

    for(int i=0; i<octaves; i++)
    {
        div += 256 * amp;
        fin += noise2d(xa, ya) * amp;
        amp /= 2;
        xa *= 2;
        ya *= 2;
    }

    return fin/div;
}

// Calculate all noise values for the texture on the cpu
float* perlin2d_cpu(int size, double frequency, int octaves) {
    float* noise = (float*)malloc(size * size * sizeof(float));

    for(int y=0; y<size; y++)
        for(int x=0; x<size; x++)
            noise[x + y * size] = perlin2d(x, y, frequency, octaves);

    return noise;
}

// Find x and y index by block and 
__global__
void perlin2d_gpu(int size, double frequency, int octaves, float* values) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float xa = x*frequency;
    float ya = y*frequency;
    float amp = 1.0;
    float fin = 0;
    float div = 0.0;

    for(int i=0; i < octaves; i++)
    {
        div += 256 * amp;
        fin += noise2d_gpu(xa, ya) * amp;
        amp /= 2;
        xa *= 2;
        ya *= 2;
    }

    values[x + y * size] = fin/div;
}

// Saves an float array as an bitmap image to disk
void save_image(char* filename, float* noise, int size) {
    BMP cpu_output = BMP(size, size, false);
    for (int i = 0; i < size * size; i++) {
        int y = i / size;
        int x = i - y * size;
        cpu_output.set_pixel(x, y, noise[i] * 255, noise[i] * 255, noise[i] * 255, 1);
    }
    cpu_output.write(filename);
}

// Random noise used as a comparison to perlin noise
float* random_noise(int size) {
    float* noise = (float*)malloc(size * size * sizeof(float));

    for (int i = 0; i < size * size; i++) {
        noise[i] = (float)rand()/(float)(RAND_MAX);
    }

    return noise;
}

// Run the cpu program for x repititions and calculate the average runtime
void testCPU(int repeats, int size) {
    clock_t start, end;
    float diff = 0;
    for (int i = 0; i < repeats; i++) {
        start = clock();
        float* noise = perlin2d_cpu(size, FREQUENCY, OCTAVES);
        end = clock();
        diff += (float)(end - start) / CLOCKS_PER_SEC;
        free(noise);
    }
    
    printf("Calculation on CPU took an average of %f seconds over %i executions\n", diff / repeats, repeats);
}

// Run the gpu program for x repititions and calculate the average runtime
void testGPU(int repeats, int size) {
    clock_t start, end;
    float diff = 0;
    for (int i = 0; i < repeats; i++) {
        start = clock();
        float* noise_gpu; 
        CHECK_CUDA_ERROR(cudaMallocManaged(&noise_gpu, size * size * sizeof(float)));
        dim3 threads_per_block (128, 128, 1);
        dim3 number_of_blocks (size / threads_per_block.x, size / threads_per_block.y, 1);

        perlin2d_gpu <<<threads_per_block, number_of_blocks>>> (size, FREQUENCY, OCTAVES, noise_gpu);
        cudaDeviceSynchronize();
        end = clock();
        diff += (float)(end - start) / CLOCKS_PER_SEC;
        cudaFree(noise_gpu);
    }
    
    printf("Calculation on GPU took an average of %f seconds over %i executions\n", diff / repeats, repeats);
}

int main(int argc, char *argv[]) {
    if (argc == 2) {
        int size = atoi(argv[1]);;

        clock_t start, end;
        start = clock();
        float* noise = perlin2d_cpu(size, FREQUENCY, OCTAVES);
        end = clock();
        printf("Calculation on CPU took %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
        save_image("output_cpu.bmp", noise, size);
        free(noise);

        start = clock();
        float* noise_gpu; 
        CHECK_CUDA_ERROR(cudaMallocManaged(&noise_gpu, size * size * sizeof(float)));
        dim3 threads_per_block (128, 128, 1);
        dim3 number_of_blocks (size / threads_per_block.x, size / threads_per_block.y, 1);

        perlin2d_gpu <<<threads_per_block, number_of_blocks>>> (size, FREQUENCY, OCTAVES, noise_gpu);
        cudaDeviceSynchronize();
        end = clock();
        CHECK_LAST_CUDA_ERROR();
        printf("Calculation on GPU took %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
        save_image("output_gpu.bmp", noise_gpu, size);

        //testCPU(50, size);
        //testGPU(50, size);
    }

    return 0;
}