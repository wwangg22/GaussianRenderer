#define _USE_MATH_DEFINES

#include <cmath>
#include "camera.hpp"
#include "math.hpp"

Camera::Camera() : fovY(45.0f), aspectRatio(1.0f), nearClip(0.1f), farClip(100.0f) {
    position[0] = 0.0f; position[1] = 0.0f; position[2] = 5.0f;
    lookAt[0] = 0.0f; lookAt[1] = 0.0f; lookAt[2] = 0.0f;
    up_vec[0] = 0.0f; up_vec[1] = 1.0f; up_vec[2] = 0.0f;
    w_up[0] = 0.0f; w_up[1] = 1.0f; w_up[2] = 0.0f;
}
void Camera::setLookAt(const float coords[3]) {
    lookAt[0] = coords[0];
    lookAt[1] = coords[1];
    lookAt[2] = coords[2];
}

void Camera::setPosition(const float coords[3]) {
    position[0] = coords[0];
    position[1] = coords[1];
    position[2] = coords[2];
}
void Camera::setFovY(float fov) {
    fovY = fov;
}
void Camera::setAspectRatio(float ratio) {
    aspectRatio = ratio;
}
void Camera::setClippingPlanes(float nearC, float farC) {
    nearClip = nearC;
    farClip = farC;
}

void Camera::updateCameraMatrices() {

    subtractVec(lookAt, position, f_axis);

    normalize(f_axis);

    crossProduct(f_axis, w_up, r_axis);
    normalize(r_axis);

    // already normalized
    crossProduct(r_axis, f_axis, u_axis);

    negateVec(f_axis); // look towards -z in camera space

    makeRigidTransformMat(r_axis, u_axis, f_axis, position, V_matrix);
    buildPerspectiveMatrix(fovY, aspectRatio, nearClip, farClip, P_matrix);
}

void Camera::zoom(float delta){
    for (int i = 0; i < 3; i++) {
        position[i] += f_axis[i] * delta;
    }
    updateCameraMatrices();
}

void Camera::orbit(float azimuth, float elevation){
    // convert to radians
    azimuth = azimuth * M_PI / 180.0f;
    elevation = elevation * M_PI / 180.0f;

    float radius_vec[3];
    subtractVec(position, lookAt, radius_vec);
    float radius = norm(radius_vec);

    float theta = std::atan2(radius_vec[2], radius_vec[0]);
    float phi = std::acos(radius_vec[1] / radius);

    theta += azimuth;
    phi += elevation;

    // clamp phi to avoid gimbal lock
    const float epsilon = 0.01f;
    if (phi < epsilon) phi = epsilon;
    if (phi > M_PI - epsilon) phi = M_PI - epsilon;

    radius_vec[0] = radius * std::sin(phi) * std::cos(theta);
    radius_vec[1] = radius * std::cos(phi);
    radius_vec[2] = radius * std::sin(phi) * std::sin(theta);

    for (int i = 0; i < 3; i++) {
        position[i] = lookAt[i] + radius_vec[i];
    }
    updateCameraMatrices();
}
