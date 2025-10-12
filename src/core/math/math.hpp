
void normalize(float vec[3]);
void projectVectoVec(const float from[3], const float to[3], float out[3]);
void dotProduct(const float a[3], const float b[3], float& out);
void crossProduct(const float a[3], const float b[3], float out[3]);
float norm(const float vec[3]);
void subtractVec(const float a[3], const float b[3], float out[3]);
void makeRigidTransformMat(const float basisX[3], const float basisY[3], const float basisZ[3], const float translation[3], float out[16]);
void buildPerspectiveMatrix(float fovY, float aspect, float near, float far, float out[16]);
void negateVec(float vec[3]);
void MatMul_4D(const float A[16], const float B[16], float out[16]);
void MatVecMul_4D(const float M[16], const float v[4], float out[4]);
void makeViewMatrix(const float basisX[3], const float basisY[3], const float basisZ[3], const float eye[3], float out[16]);
void build3x3RotationMatrix(const float basisX[3], const float basisY[3], const float basisZ[3], float R[9]);
void transpose3x3(const float A[9], float At[9]);
void buildRotMatFromQuat(float quat[4], float R[9]);
void buildDiagonalMatrix(const float a[3], float D[9]);
void MatMul_3D(const float A[9], const float B[9], float out[9]);
void GeMatMul(const float* A, const float* B, int M, int N, int K, float* out);