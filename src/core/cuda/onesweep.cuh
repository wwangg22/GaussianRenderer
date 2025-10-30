#pragma once
#define RADIX_BITS 8
#define RADIX 256

#define TILE_SIZE 2048

extern "C" void oneSweepSort(int* input, int* output, int N, int maxVal, float* kernel_ms);

