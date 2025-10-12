#define _USE_MATH_DEFINES
#include "gaussians.hpp"
#include "camera.hpp"
#include "math.hpp"
#include "render.hpp"

void frustum_cull(std::vector<Gaussian>& g,std::vector<Gaussian>& out, float* planes, float& treshold){
    // planes contains the normal vectors for the 6 planes (24,)
    // stored as [x,y,z, offset]
    for (auto& gauss : g) {
        for (int i =0; i < 6; i++){
            float* normal = planes + i*4;
            float xyz[3] = {gauss.x, gauss.y, gauss.z};
            float dot;
            dotProduct(xyz, normal, dot);
            if (dot + planes[i*4+3] < -treshold){
                break; // outside
            }
        }
        out.push_back(gauss);
    }
}


void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam, 
                                TilingInformation& tile_info, float fovY, float aspect, float k){
    // k determins the k-sigma radi
    float* V = cam.V_matrix;
    float* P = cam.P_matrix;
    float* R_cam = cam.r_cam;
    float* R_cam_T = cam.r_cam_T;
    float pad = 1.0f; // padding factor
    const float fy = 1.0f / std::tan(fovY * 0.5f * (M_PI / 180.0f));
    const float fx = fy / aspect;
    float width_stride = tile_info.W / tile_info.num_tile_x;
    float height_stride = tile_info.H / tile_info.num_tile_y;
    float jacobian[6];
    float jacobian_T[6];
    float R[9];
    float R_T[9];
    float scale_mat[9];
    float scale[3];
    float tmp[9];
    float covar[9];
    for (int idx = 0 ; idx < g.size(); idx++) {
        Gaussian& gauss = g[idx];
        float old_xyz[4] = {gauss.x, gauss.y, gauss.z, 1.0f};
        float new_xyz[4];
        float tmp_xyz[4];
        float X, Y, Z;

        MatVecMul_4D(V, old_xyz, tmp_xyz);
        X=tmp_xyz[0];
        Y=tmp_xyz[1];
        Z=tmp_xyz[2];
        MatVecMul_4D(P, tmp_xyz, new_xyz);
        new_xyz[0] = new_xyz[0] / new_xyz[3];
        new_xyz[1] =  new_xyz[1] /  new_xyz[3];
        new_xyz[2] =  new_xyz[2] /  new_xyz[3];

        gauss.x = new_xyz[0];
        gauss.y = new_xyz[1];
        gauss.z = new_xyz[2];
        if (gauss.z < 0) {
            std::cout << "Warning: Please Cull Gaussians beforehand.\n";
            return;
        }

        // contruct Jacobian
        jacobian[0] = fx / Z; jacobian[1] = 0.0f;
        jacobian[2] = -fx *X / (Z * Z); jacobian[3] = 0.0f;
        jacobian[4] = fy / Z; jacobian[5] = -fy * Y / (Z * Z);
        // transpose
        jacobian_T[0] = jacobian[0]; jacobian_T[1] = jacobian[2]; jacobian_T[2] = jacobian[4];
        jacobian_T[3] = jacobian[1]; jacobian_T[4] = jacobian[3]; jacobian_T[5] = jacobian[5];
        //build covariance
        buildRotMatFromQuat(gauss.rot, R);
        transpose3x3(R, R_T);
        scale[0] = gauss.scale[0] * gauss.scale[0];
        scale[1] = gauss.scale[1] * gauss.scale[1];
        scale[2] = gauss.scale[2] * gauss.scale[2];
        buildDiagonalMatrix(scale, scale_mat);

        // covar = R * S * R_T
        MatMul_3D(R, scale_mat, tmp);
        MatMul_3D(tmp, R_T, covar);

        // covar transform to world
        MatMul_3D(R_cam, covar, tmp);
        MatMul_3D(tmp, R_cam_T, covar);

        // covar  = J * covar * J_T
        GeMatMul(jacobian, covar, 2, 3, 3, tmp);
        float Sigma2D[2*2];
        GeMatMul(tmp, jacobian_T, 2, 3, 2, Sigma2D);

        memcpy(gauss.covar, Sigma2D, sizeof(float)*4);

        //extract eigenvalues
        float Sxx = Sigma2D[0];
        float Sxy = Sigma2D[1];
        float Syx = Sigma2D[2];
        float Syy = Sigma2D[3];

        float sxy = 0.5f*(Sxy + Syx);


        float tr  = Sxx + Syy;
        float dif = Sxx - Syy;
        float rad = std::sqrt( std::max(0.0f, dif*dif + 4*sxy*sxy) );
        float lamb_1  = 0.5f*(tr + rad);
        float lamb_2  = 0.5f*(tr - rad);
        float theta = 0.5f * std::atan2(2*sxy, dif); // in radians

        float r1 = k * std::sqrt(lamb_1);
        float r2 = k * std::sqrt(lamb_2);

        float c = std::cos(theta);
        float s = std::sin(theta);
        float ex = std::fabs(r1*c) + std::fabs(r2*s);
        float ey = std::fabs(r1*s) + std::fabs(r2*c);

        float xmin = new_xyz[0] - ex - pad;
        float xmax = new_xyz[0] + ex + pad;
        float ymin = new_xyz[1] - ey - pad;
        float ymax = new_xyz[1] + ey + pad;

        gauss.aabb[0] = xmin;
        gauss.aabb[1] = ymin;
        gauss.aabb[2] = xmax;
        gauss.aabb[3] = ymax;

        // check which tile it belongs to
        int lowest_tile = static_cast<int> (std::floor(((xmin + 1.0f) * 0.5f) * tile_info.W / width_stride));
        int highest_tile = static_cast<int> (std::floor(((xmax + 1.0f) * 0.5f) * tile_info.W / width_stride));
        int leftmost_tile = static_cast<int> (std::floor(((ymin + 1.0f) * 0.5f) * tile_info.H / height_stride));
        int rightmost_tile = static_cast<int> (std::floor(((ymax + 1.0f) * 0.5f) * tile_info.H / height_stride));

        for (int i = lowest_tile; i <= highest_tile; i++) {
            for (int j = leftmost_tile; j <= rightmost_tile; j++) {
                uint32_t tile_id = i + j * tile_info.num_tile_x;
                lightWeightGaussian lwg;
                lwg.radix_id = (static_cast<uint64_t>(tile_id) << 32) | static_cast<uint32_t> (-Z);
                lwg.gaussian_id = static_cast<uint32_t> (idx); 
                out.push_back(lwg);
                tile_info.tile_id_counts[tile_id]++;
            }
        }
    }
}



