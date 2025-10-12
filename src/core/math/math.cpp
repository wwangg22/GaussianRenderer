#define _USE_MATH_DEFINES
#include <cmath>
#include "math.hpp"



void normalize(float vec[3]){
    float norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if (norm > 1e-8f) {
        vec[0] /= norm;
        vec[1] /= norm;
        vec[2] /= norm;
    }
    else {
        vec[0] = 0.0f;
        vec[1] = 0.0f;
        vec[2] = 0.0f;
    }
}

float norm(const float vec[3]){
    return std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

void negateVec(float vec[3]){
    vec[0] = -vec[0];
    vec[1] = -vec[1];
    vec[2] = -vec[2];
}
void dotProduct(const float a[3], const float b[3], float& out){
    out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
void projectVectoVec(const float from[3], const float to[3], float out[3]){
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
void crossProduct(const float a[3], const float b[3], float out[3]){
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

void subtractVec(const float a[3], const float b[3], float out[3]){
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

void makeRigidTransformMat(const float basisX[3], const float basisY[3], const float basisZ[3], const float translation[3], float out[16]) {
    out[0]  = basisX[0]; out[1]  = basisX[1]; out[2]  = basisX[2]; out[3]  = translation[0];
    out[4]  = basisY[0]; out[5]  = basisY[1]; out[6]  = basisY[2]; out[7]  = translation[1];
    out[8]  = basisZ[0]; out[9]  = basisZ[1]; out[10] = basisZ[2]; out[11] = translation[2];
    out[12] = 0.0f;      out[13] = 0.0f;      out[14] = 0.0f;      out[15] = 1.0f;
}
void makeViewMatrix(const float basisX[3], const float basisY[3], const float basisZ[3], const float eye[3], float out[16])
{
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
void buildPerspectiveMatrix(float fovY, float aspect, float near, float far, float out[16]) {
    float f = 1.0f / std::tan(fovY * 0.5f * (M_PI / 180.0f));
    out[0]  = f / aspect; out[1]  = 0.0f; out[2]  = 0.0f;                          out[3]  = 0.0f;
    out[4]  = 0.0f;       out[5]  = f;    out[6]  = 0.0f;                          out[7]  = 0.0f;
    out[8]  = 0.0f;       out[9]  = 0.0f; out[10] = (far + near) / (near - far);   out[11] = (2 * far * near) / (near - far);
    out[12] = 0.0f;       out[13] = 0.0f; out[14] = -1.0f;                         out[15] = 0.0f;
}
// this is CPU for now while we build the rest of the stuff
void MatMul_4D(const float A[16], const float B[16], float out[16]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                out[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
            }
        }
    }
}
void MatMul_3D(const float A[9], const float B[9], float out[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                out[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
            }
        }
    }
}
void MatVecMul_4D(const float M[16], const float v[4], float out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            out[i] += M[i * 4 + j] * v[j];
        }
    }
}
void build3x3RotationMatrix(const float basisX[3], const float basisY[3], const float basisZ[3], float R[9]) {
    R[0] = basisX[0]; R[1] = basisX[1]; R[2] = basisX[2];
    R[3] = basisY[0]; R[4] = basisY[1]; R[5] = basisY[2];
    R[6] = basisZ[0]; R[7] = basisZ[1]; R[8] = basisZ[2];
}
void transpose3x3(const float A[9], float At[9]) {
    At[0] = A[0]; At[1] = A[3]; At[2] = A[6];
    At[3] = A[1]; At[4] = A[4]; At[5] = A[7];
    At[6] = A[2]; At[7] = A[5]; At[8] = A[8];
}

void buildRotMatFromQuat(float quat[4], float R[9]){
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];

    R[0] = 1 - 2*y*y - 2*z*z; R[1] = 2*x*y - 2*w*z;     R[2] = 2*x*z + 2*w*y;
    R[3] = 2*x*y + 2*w*z;     R[4] = 1 - 2*x*x - 2*z*z; R[5] = 2*y*z - 2*w*x;
    R[6] = 2*x*z - 2*w*y;     R[7] = 2*y*z + 2*w*x;     R[8] = 1 - 2*x*x - 2*y*y;
}

void buildDiagonalMatrix(const float a[3], float D[9]){
    D[0] = a[0]; D[1] = 0.0f;  D[2] = 0.0f;
    D[3] = 0.0f;  D[4] = a[1]; D[5] = 0.0f;
    D[6] = 0.0f;  D[7] = 0.0f;  D[8] = a[2];
}

void GeMatMul(const float* A, const float* B, int M, int N, int K, float* out) {
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