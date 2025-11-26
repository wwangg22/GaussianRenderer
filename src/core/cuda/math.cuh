#pragma once
#include <cuda_runtime.h>

// Device-only CUDA math helpers
__device__ void normalize_cuda(float vec[3]);
__device__ void projectVecToVec_cuda(const float from[3], const float to[3], float out[3]);
__device__ void dotProduct_cuda(const float a[3], const float b[3], float& out);
__device__ void crossProduct_cuda(const float a[3], const float b[3], float out[3]);
__device__ float norm_cuda(const float vec[3]);
__device__ void subtractVec_cuda(const float a[3], const float b[3], float out[3]);

__device__ void makeRigidTransformMat_cuda(const float basisX[3], const float basisY[3], const float basisZ[3],
                                           const float translation[3], float out[16]);

__device__ void buildPerspectiveMatrix_cuda(float fovY, float aspect, float znear, float zfar, float out[16]);

__device__ void negateVec_cuda(float vec[3]);

__device__ void matMul4D_cuda(const float A[16], const float B[16], float out[16]);
__device__ void matVecMul4D_cuda(const float M[16], const float v[4], float out[4]);

__device__ void makeViewMatrix_cuda(const float basisX[3], const float basisY[3], const float basisZ[3],
                                    const float eye[3], float out[16]);

__device__ void build3x3RotationMatrix_cuda(const float basisX[3], const float basisY[3], const float basisZ[3],
                                            float R[9]);

__device__ void transpose3x3_cuda(const float A[9], float At[9]);
__device__ void buildRotMatFromQuat_cuda(const float quat[4], float R[9]);
__device__ void buildDiagonalMatrix_cuda(const float a[3], float D[9]);

__device__ void matMul3D_cuda(const float A[9], const float B[9], float out[9]);
__device__ void geMatMul_cuda(const float* A, const float* B, int M, int N, int K, float* out);
