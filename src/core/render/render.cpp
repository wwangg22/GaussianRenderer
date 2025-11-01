#define _USE_MATH_DEFINES
#include <algorithm>
#include "gaussians.hpp"
#include "camera.hpp"
#include "math.hpp"
#include "render.hpp"

std::vector<Gaussian> frustum_cull(std::vector<Gaussian>& g, Camera& cam, const float& treshold){
    // planes contains the normal vectors for the 6 planes (24,)
    // stored as [x,y,z, offset]
    float* planes = cam.plane_normals;
    std::vector<Gaussian> out;
    for (auto& gauss : g) {
        int i;
        for (i =0; i < 6; i++){
            float* normal = planes + i*4;
            float xyz[3] = {gauss.x, gauss.y, gauss.z};
            float dot;
            dotProduct(xyz, normal, dot);
            if (dot + planes[i*4+3] < -treshold){
                break; // outside
            }
        }
        if (i == 6) out.push_back(gauss);
    }
    return out;
}

const float SH_C0 = 0.28209479177387814f;
const float SH_C1 = 0.4886025119029199f;
const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};


void transformAndTileGaussians(std::vector<Gaussian>& g, std::vector<lightWeightGaussian>& out, Camera& cam, 
                                TilingInformation& tile_info, float k){
    std::cout << "Transforming and Tiling " << g.size() << " gaussians.\n";
    float aspect = cam.aspectRatio;
    float fovY = cam.fovY;
    // k determins the k-sigma radi
    float* V = cam.V_matrix;
    float* P = cam.P_matrix;
    float* R_cam = cam.r_cam;
    float* R_cam_T = cam.r_cam_T;
    float pad = 1.0f; // padding factor
    const float fy = 1.0f / std::tan(fovY * 0.5f * (M_PI / 180.0f));
    const float fx = fy / aspect;
    int deg = 3;

    // std::cout << "Camera fx: " << fx << ", fy: " << fy << "\n";

    int width_stride  = tile_info.width_stride;
    int height_stride = tile_info.height_stride;
    // std::cout << "Tile size: " << width_stride << " x " << height_stride << "\n";
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

        // print spherical harmonics values:
        // for (int i = 0; i < 27; i++) {
        //     std::cout << "SH " << i << ": " << gauss.sh[i] << "\n";
        // }

        // grab color from SH
        float viewing_dir[3] = {gauss.x - cam.position[0],
                                gauss.y - cam.position[1],
                                gauss.z - cam.position[2]};
        normalize(viewing_dir);

        // print viewing dir
        // std::cout << "Viewing dir: ";
        // std::cout << viewing_dir[0] << ", " << viewing_dir[1] << ", " << viewing_dir[2] << "\n";
        for (int i = 0; i <3; i++) gauss.color[i] = gauss.sh[i] * SH_C0; 

        if (deg > 0)
        {
            float x = viewing_dir[0];
            float y = viewing_dir[1];
            float z = viewing_dir[2];

            for (int i = 0; i < 3; i++){
                gauss.color[i] += SH_C1 * z * gauss.sh[2*3 + i];
                gauss.color[i] -= SH_C1 * y * gauss.sh[3 + i];
                gauss.color[i] -= SH_C1 * x * gauss.sh[3*3 + i];
            }

            if (deg > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                for (int i = 0; i < 3; i++) {
                    gauss.color[i] += SH_C2[0] * xy * gauss.sh[4*3 + i];
                    gauss.color[i] += SH_C2[1] * yz * gauss.sh[5*3 + i];
                    gauss.color[i] += SH_C2[2] * (2.0f * zz - xx - yy) * gauss.sh[6*3 + i];
                    gauss.color[i] += SH_C2[3] * xz * gauss.sh[7*3 + i];
                    gauss.color[i] += SH_C2[4] * (xx - yy) * gauss.sh[8*3 + i];
                }
                // if (deg > 2)
                // {
                //     for(int i = 0; i < 3; i++) {
                //         gauss.color[i] += SH_C3[0] * y * (3.0f * xx - yy) * gauss.sh[9*3 + i];
                //         gauss.color[i] += SH_C3[1] * xy * z * gauss.sh[10*3 + i];
                //         gauss.color[i] += SH_C3[2] * y * (4.0f * zz - xx - yy) * gauss.sh[11*3 + i];
                //         gauss.color[i] += SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * gauss.sh[12*3 + i];
                //         gauss.color[i] += SH_C3[4] * x * (4.0f * zz - xx - yy) * gauss.sh[13*3 + i];
                //         gauss.color[i] += SH_C3[5] * z * (xx - yy) * gauss.sh[14*3 + i];
                //         gauss.color[i] += SH_C3[6] * x * (xx - 3.0f * yy) * gauss.sh[15*3 + i];
                //     }
                // }
            }
        }
        for (int i = 0; i <3; i++){
            gauss.color[i] += 0.5f;
            gauss.color[i] = std::min(std::max(gauss.color[i], 0.0f), 1.0f);
        }
        // print color of gaussian
        // std::cout << "Gaussian " << idx << " color: ";
        // std::cout << gauss.color[0] << ", " << gauss.color[1] << ", " << gauss.color[2] << "\n";
        float old_xyz[4] = {gauss.x, gauss.y, gauss.z, 1.0f};
        float new_xyz[4];
        float tmp_xyz[4];
        float X, Y, Z;

        MatVecMul_4D(V, old_xyz, tmp_xyz);
        X=tmp_xyz[0];
        Y=tmp_xyz[1];
        Z=tmp_xyz[2];
        // print xyz
        // std::cout << "Gaussian " << idx << " camera space coords: ";
        // std::cout << X << ", " << Y << ", " << Z << "\n";
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
        jacobian[0] = - fx / Z; jacobian[1] = 0.0f;
        jacobian[2] = fx *X / (Z * Z); jacobian[3] = 0.0f;
        jacobian[4] = - fy / Z; jacobian[5] = fy * Y / (Z * Z);
        // transpose
        jacobian_T[0] = jacobian[0]; jacobian_T[1] = jacobian[3]; jacobian_T[2] = jacobian[1];
        jacobian_T[3] = jacobian[4]; jacobian_T[4] = jacobian[2]; jacobian_T[5] = jacobian[5];

        // print Jacobian:
        // std::cout << "Jacobian for gaussian " << idx << ": \n";
        // std::cout << jacobian[0] << " " << jacobian[1] << "\n";
        // std::cout << jacobian[2] << " " << jacobian[3] << "\n";
        // std::cout << jacobian[4] << " " << jacobian[5] << "\n";
        //build covariance
        buildRotMatFromQuat(gauss.rot, R);
        transpose3x3(R, R_T);
        scale[0] = gauss.scale[0] * gauss.scale[0];
        scale[1] = gauss.scale[1] * gauss.scale[1];
        scale[2] = gauss.scale[2] * gauss.scale[2];
        buildDiagonalMatrix(scale, scale_mat);

        // print scale matrix 
        // std::cout << "Scale matrix for gaussian " << idx << ": \n";
        // std::cout << scale_mat[0] << " " << scale_mat[1] << " " << scale_mat[2] << "\n";
        // std::cout << scale_mat[3] << " " << scale_mat[4] << " " << scale_mat[5] << "\n";
        // std::cout << scale_mat[6] << " " << scale_mat[7] << " " << scale_mat[8] << "\n";  

        // covar = R * S * R_T
        MatMul_3D(R, scale_mat, tmp);
        MatMul_3D(tmp, R_T, covar);

        // print R matrix
        // std::cout << "Rotation matrix R for gaussian " << idx << ": \n";
        // std::cout << R[0] << " " << R[1] << " " << R[2] << "\n";
        // std::cout << R[3] << " " << R[4] << " " << R[5] << "\n";
        // std::cout << R[6] << " " << R[7] << " " << R[8] << "\n";

        // covar transform to world
        MatMul_3D(R_cam, covar, tmp);
        MatMul_3D(tmp, R_cam_T, covar);
        
        //print Rcam matrix
        // std::cout << "R_cam matrix: \n";
        // std::cout << R_cam[0] << " " << R_cam[1] << " " << R_cam[2] << "\n";
        // std::cout << R_cam[3] << " " << R_cam[4] << " " << R_cam[5] << "\n";
        // std::cout << R_cam[6] << " " << R_cam[7] << " " << R_cam[8] << "\n";

        // covar  = J * covar * J_T
        GeMatMul(jacobian, covar, 2, 3, 3, tmp);
        float Sigma2D[2*2];
        GeMatMul(tmp, jacobian_T, 2, 2, 3, Sigma2D);

        // print sigma2D
        // std::cout << "Projected 2D covariance Sigma2D for gaussian " << idx << ": \n";
        // std::cout << Sigma2D[0] << " " << Sigma2D[1] << "\n";
        // std::cout << Sigma2D[2] << " " << Sigma2D[3] << "\n";

        float invSigma2D[4];
        float det = Sigma2D[0]*Sigma2D[3] - Sigma2D[1]*Sigma2D[2];
        if (det < 1e-8f) {
            std::cout << "Warning: Singular Covariance Matrix for gaussian " << idx << ". Skipping.\n";
            continue;
        }
        float invDet = 1.0f / det;
        invSigma2D[0] =  Sigma2D[3] * invDet; invSigma2D[1] = -Sigma2D[1] * invDet;
        invSigma2D[2] = -Sigma2D[2] * invDet; invSigma2D[3] =  Sigma2D[0] * invDet;

        memcpy(gauss.inv_covar, invSigma2D, sizeof(float)*4);

        //extract eigenvalues
        float Sxx = Sigma2D[0];
        float Sxy = Sigma2D[1];
        float Syx = Sigma2D[2];
        float Syy = Sigma2D[3];

        float sxy = 0.5f*(Sxy + Syx);

        // std::cout << "eigenvalues for gaussian " << idx << ": ";
        // std::cout << Sxx << ", " << Syy << ", " << sxy << "\n";


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

        // std::cout << "Gaussian " << idx << " projected radii: ex = " << ex << ", ey = " << ey << "\n";

        float xmin = std::clamp(new_xyz[0] - ex, -1.0f, 1.0f);
        float xmax = std::clamp(new_xyz[0] + ex, -1.0f, 1.0f);
        float ymin = std::clamp(new_xyz[1] - ey, -1.0f, 1.0f);
        float ymax = std::clamp(new_xyz[1] + ey, -1.0f, 1.0f);

        // std::cout << "Gaussian " << idx << " projected aabb in NDC: (" << xmin << ", " << ymin << ") to (" << xmax << ", " << ymax << ")\n";

        int xmin_px = static_cast<int> (std::floor(((xmin + 1.0f) * 0.5f) * tile_info.W));
        int xmax_px = static_cast<int> (std::ceil(((xmax + 1.0f) * 0.5f) * tile_info.W));
        int ymin_px = static_cast<int> (std::floor(((ymin + 1.0f) * 0.5f) * tile_info.H));
        int ymax_px = static_cast<int> (std::ceil(((ymax + 1.0f) * 0.5f) * tile_info.H));

        gauss.px_x = static_cast<int> (std::round(((new_xyz[0] + 1.0f) * 0.5f) * tile_info.W));
        gauss.px_y = static_cast<int> (std::round(((new_xyz[1] + 1.0f) * 0.5f) * tile_info.H));

        gauss.aabb[0] = xmin_px;
        gauss.aabb[1] = ymin_px;
        gauss.aabb[2] = xmax_px;
        gauss.aabb[3] = ymax_px;

        // check which tile it belongs to
        int min_x = std::max(0, xmin_px / width_stride);
        int max_x = std::min(tile_info.num_tile_x-1, xmax_px / width_stride);
        int min_y = std::max(0, ymin_px / height_stride);
        int max_y = std::min(tile_info.num_tile_y-1, ymax_px / height_stride);
        // std::cout << "Gaussian " << idx << " in tiles x: " << min_x << " to " << max_x << ", y: " << min_y << " to " << max_y << "\n";

        for (int i = min_x; i <= max_x; i++) {
            for (int j = min_y; j <= max_y; j++) {
                uint32_t tile_id = i + j * tile_info.num_tile_x;
                lightWeightGaussian lwg;
                lwg.radix_id = (static_cast<uint64_t>(tile_id) << 32) | static_cast<uint32_t> (-Z);
                lwg.gaussian_id = static_cast<uint32_t> (idx); 
                out.push_back(lwg);
                tile_info.tile_id_offset[tile_id]++;
            }
        }
    }
    // inclusive scan to get offsets
    for (int i = 1; i < tile_info.num_tile_x * tile_info.num_tile_y; i++) {
        tile_info.tile_id_offset[i] += tile_info.tile_id_offset[i - 1];
    }
}



