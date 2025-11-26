#include <cuda_runtime.h>
#include <math_constants.h>  // for CUDART_PI_F
#include "math.cuh"

// Device math utility functions

__device__ void normalize_cuda(float vec[3]) {
    float n = sqrtf(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if (n > 1e-8f) {
        vec[0] /= n;
        vec[1] /= n;
        vec[2] /= n;
    } else {
        vec[0] = 0.0f;
        vec[1] = 0.0f;
        vec[2] = 0.0f;
    }
}

__device__ float norm_cuda(const float vec[3]) {
    return sqrtf(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

__device__ void negateVec_cuda(float vec[3]) {
    vec[0] = -vec[0];
    vec[1] = -vec[1];
    vec[2] = -vec[2];
}

__device__ void dotProduct_cuda(const float a[3], const float b[3], float& out) {
    out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ void projectVecToVec_cuda(const float from[3], const float to[3], float out[3]) {
    float dot = from[0]*to[0] + from[1]*to[1] + from[2]*to[2];
    float to_norm_sq = to[0]*to[0] + to[1]*to[1] + to[2]*to[2];
    if (to_norm_sq > 0.0f) {
        float scalar = dot / to_norm_sq;
        out[0] = scalar * to[0];
        out[1] = scalar * to[1];
        out[2] = scalar * to[2];
    } else {
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
    }
}

__device__ void crossProduct_cuda(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ void subtractVec_cuda(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

__device__ void makeRigidTransformMat_cuda(const float basisX[3], const float basisY[3],
                                           const float basisZ[3], const float translation[3],
                                           float out[16]) {
    out[0]  = basisX[0]; out[1]  = basisX[1]; out[2]  = basisX[2]; out[3]  = translation[0];
    out[4]  = basisY[0]; out[5]  = basisY[1]; out[6]  = basisY[2]; out[7]  = translation[1];
    out[8]  = basisZ[0]; out[9]  = basisZ[1]; out[10] = basisZ[2]; out[11] = translation[2];
    out[12] = 0.0f;      out[13] = 0.0f;      out[14] = 0.0f;      out[15] = 1.0f;
}

__device__ void makeViewMatrix_cuda(const float basisX[3], const float basisY[3],
                                    const float basisZ[3], const float eye[3],
                                    float out[16]) {
    // Row 0
    out[0] = basisX[0];
    out[1] = basisX[1];
    out[2] = basisX[2];
    out[3] = -(basisX[0]*eye[0] + basisX[1]*eye[1] + basisX[2]*eye[2]);

    // Row 1
    out[4] = basisY[0];
    out[5] = basisY[1];
    out[6] = basisY[2];
    out[7] = -(basisY[0]*eye[0] + basisY[1]*eye[1] + basisY[2]*eye[2]);

    // Row 2
    out[8]  = basisZ[0];
    out[9]  = basisZ[1];
    out[10] = basisZ[2];
    out[11] = -(basisZ[0]*eye[0] + basisZ[1]*eye[1] + basisZ[2]*eye[2]);

    // Row 3
    out[12] = 0.0f;
    out[13] = 0.0f;
    out[14] = 0.0f;
    out[15] = 1.0f;
}

__device__ void buildPerspectiveMatrix_cuda(float fovY, float aspect,
                                            float nearVal, float farVal,
                                            float out[16]) {
    float f = 1.0f / tanf(fovY * 0.5f * (CUDART_PI_F / 180.0f));
    out[0]  = f / aspect; out[1]  = 0.0f;   out[2]  = 0.0f;                             out[3]  = 0.0f;
    out[4]  = 0.0f;       out[5]  = f;      out[6]  = 0.0f;                             out[7]  = 0.0f;
    out[8]  = 0.0f;       out[9]  = 0.0f;   out[10] = (farVal + nearVal) / (nearVal - farVal);
    out[11] = (2 * farVal * nearVal) / (nearVal - farVal);
    out[12] = 0.0f;       out[13] = 0.0f;   out[14] = -1.0f;                            out[15] = 0.0f;
}

__device__ void matMul4D_cuda(const float A[16], const float B[16], float out[16]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                out[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
            }
        }
    }
}

__device__ void matMul3D_cuda(const float A[9], const float B[9], float out[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                out[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
            }
        }
    }
}

__device__ void matVecMul4D_cuda(const float M[16], const float v[4], float out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            out[i] += M[i * 4 + j] * v[j];
        }
    }
}

__device__ void build3x3RotationMatrix_cuda(const float basisX[3], const float basisY[3],
                                            const float basisZ[3], float R[9]) {
    R[0] = basisX[0]; R[1] = basisX[1]; R[2] = basisX[2];
    R[3] = basisY[0]; R[4] = basisY[1]; R[5] = basisY[2];
    R[6] = basisZ[0]; R[7] = basisZ[1]; R[8] = basisZ[2];
}

__device__ void transpose3x3_cuda(const float A[9], float At[9]) {
    At[0] = A[0]; At[1] = A[3]; At[2] = A[6];
    At[3] = A[1]; At[4] = A[4]; At[5] = A[7];
    At[6] = A[2]; At[7] = A[5]; At[8] = A[8];
}

__device__ void buildRotMatFromQuat_cuda(const float quat[4], float R[9]) {
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    float n = sqrtf(x*x + y*y + z*z + w*w);
    x /= n; y /= n; z /= n; w /= n;

    R[0] = 1 - 2*y*y - 2*z*z; R[1] = 2*x*y - 2*w*z;     R[2] = 2*x*z + 2*w*y;
    R[3] = 2*x*y + 2*w*z;     R[4] = 1 - 2*x*x - 2*z*z; R[5] = 2*y*z - 2*w*x;
    R[6] = 2*x*z - 2*w*y;     R[7] = 2*y*z + 2*w*x;     R[8] = 1 - 2*x*x - 2*y*y;
}

__device__ void buildDiagonalMatrix_cuda(const float a[3], float D[9]) {
    D[0] = a[0]; D[1] = 0.0f;  D[2] = 0.0f;
    D[3] = 0.0f; D[4] = a[1];  D[5] = 0.0f;
    D[6] = 0.0f; D[7] = 0.0f;  D[8] = a[2];
}

__device__ void geMatMul_cuda(const float* A, const float* B,
                              int M, int N, int K,
                              float* out) {
    // A is MxK
    // B is KxN
    // out is MxN
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out[i * N + j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                out[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
