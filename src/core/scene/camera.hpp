
struct Camera {
    float position[3];
    float lookAt[3];
    float w_up[3];
    float fovY; // in degrees
    float aspectRatio;
    float nearClip;
    float farClip;

    float forward_vec[3];
    float right_vec[3];
    float up_vec[3];

    float P_matrix[16];
    float V_matrix[16];
    float M_matrix[16]; // model matrix, identity for now
    float f_axis[3];
    float r_axis[3];
    float u_axis[3];


    Camera();

    void setLookAt(const float coords[3]);
    void setPosition(const float coords[3]);
    void setFovY(float fov);
    void setAspectRatio(float ratio);
    void setClippingPlanes(float nearC, float farC);

    void updateCameraMatrices();

    void zoom(float delta); 
    void orbit(float azimuth, float elevation);

    void transformPointToCameraSpace(const float point[4], float out[4]);
};