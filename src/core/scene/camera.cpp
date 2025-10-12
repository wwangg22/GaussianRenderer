#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
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

    build3x3RotationMatrix(r_axis, u_axis, f_axis, r_cam);
    transpose3x3(r_cam, r_cam_T);

    makeViewMatrix(r_axis, u_axis, f_axis, position, V_matrix);
    buildPerspectiveMatrix(fovY, aspectRatio, nearClip, farClip, P_matrix);

    MatMul_4D(P_matrix, V_matrix, M_matrix);
}

void Camera::updateFrustumPlanes(){
    // compute plane normals for frustum culling
    // near plane
    for (int i = 0; i < 3; i++) {
        plane_normals[i] = f_axis[i];
    }
    //offset
    plane_normals[3] = (f_axis[0]*position[0] + f_axis[1]*position[1] + f_axis[2]*position[2] - nearClip);

    // far plane
    for (int i = 0; i < 3; i++) {
        plane_normals[4 + i] = -f_axis[i];
    }   
    //offset
    plane_normals[7] = -(f_axis[0]*position[0] + f_axis[1]*position[1] + f_axis[2]*position[2] - farClip);

    float t_y = std::tan(fovY * 0.5f * (M_PI / 180.0f));
    float t_x = t_y * aspectRatio;
    float right_norm[3];
    //right plane
    for (int i = 0; i < 3; i ++) {
        right_norm[i] = f_axis[i] * t_x - r_axis[i];
    }
    normalize(right_norm);
    for (int i = 0; i < 3; i++) {
        plane_normals[8 + i] = right_norm[i];
    }   

    //offset
    plane_normals[11] = 0.0f;

    //left plane
    float left_norm[3];
    for (int i = 0; i < 3; i ++) {
        left_norm[i] = f_axis[i] * t_x + r_axis[i];
    }
    normalize(left_norm);
    for (int i = 0; i < 3; i++) {
        plane_normals[12 + i] = left_norm[i];
    }
    plane_normals[15] = 0.0f;

    //top plane
    float top_norm[3];
    for (int i = 0; i < 3; i ++) {
        top_norm[i] = f_axis[i] * t_y - u_axis[i];
    }
    normalize(top_norm);
    for (int i = 0; i < 3; i++) {
        plane_normals[16 + i] = top_norm[i];
    }
    plane_normals[19] = 0.0f;
    //bottom plane
    float bottom_norm[3];
    for (int i = 0; i < 3; i ++) {
        bottom_norm[i] = f_axis[i] * t_y + u_axis[i];
    }
    normalize(bottom_norm);
    for (int i = 0; i < 3; i++) {
        plane_normals[20 + i] = bottom_norm[i];
    }
    plane_normals[23] = 0.0f;
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

void Camera::transformPointToCameraSpace(const float point[4], float out[4]){
    MatVecMul_4D(M_matrix, point, out);

    // float out_test[4];
    // MatVecMul_4D(V_matrix, point, out_test);
    // std::cout << "After View Matrix: (" << out_test[0] << ", " << out_test[1] << ", " << out_test[2] << ")\n";

    out[0] = out[0] / out[3];
    out[1] = out[1] / out[3];
    out[2] = out[2] / out[3];
}
