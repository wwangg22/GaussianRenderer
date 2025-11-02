#define _USE_MATH_DEFINES
#include <algorithm>
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
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
        jacobian[0] = fx / Z; jacobian[1] = 0.0f;
        jacobian[2] = - fx *X / (Z * Z); jacobian[3] = 0.0f;
        jacobian[4] = fy / Z; jacobian[5] = -fy * Y / (Z * Z);

        // jacobian[0] = -fx / Z; jacobian[1] = 0.0f;
        // jacobian[2] = fx *X / (Z * Z); jacobian[3] = 0.0f;
        // jacobian[4] = -fy / Z; jacobian[5] = fy * Y / (Z * Z);
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

        // // print scale matrix 
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

        //print aabb in pixels
        // std::cout << "Gaussian " << idx << " projected aabb in pixels: (" << xmin_px << ", " << ymin_px << ") to (" << xmax_px << ", " << ymax_px << ")\n";

        gauss.px_x = static_cast<int> (std::round(((new_xyz[0] + 1.0f) * 0.5f) * tile_info.W));
        gauss.px_y = static_cast<int> (std::round(((new_xyz[1] + 1.0f) * 0.5f) * tile_info.H));

        // print mean in pixels
        // std::cout << "Gaussian " << idx << " mean in pixels: (" << gauss.px_x << ", " << gauss.px_y << ")\n";

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
        // exit(1);
        // for (int i = min_x; i <= max_x; i++) {
        //     for (int j = min_y; j <= max_y; j++) {
        //         uint32_t tile_id = i + j * tile_info.num_tile_x;
        //         lightWeightGaussian lwg;
                // lwg.radix_id = (static_cast<uint64_t>(tile_id) << 32) | static_cast<uint32_t> (gauss.z);
        //         lwg.gaussian_id = static_cast<uint32_t> (idx); 
        //         out.push_back(lwg);
        //         tile_info.tile_id_offset[tile_id]++;
        //     }
        // }
       
        lightWeightGaussian lwg;
        // lwg.radix_id = (static_cast<uint64_t>(tile_id) << 32) | static_cast<uint32_t> (gauss.z);
        lwg.radix_id = static_cast<uint64_t> (gauss.z * 1e6f);
        lwg.gaussian_id = static_cast<uint32_t> (idx); 
        out.push_back(lwg);
          
    }
    // inclusive scan to get offsets
    for (int i = 1; i < tile_info.num_tile_x * tile_info.num_tile_y; i++) {
        tile_info.tile_id_offset[i] += tile_info.tile_id_offset[i - 1];
    }
}

void drawScreen(float* pixel_out){

    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // only for Mac OS X
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const int W = 800, H = 600;
    GLFWwindow* win = glfwCreateWindow(W, H, "My Blank Viewer", nullptr, nullptr);
    if (!win) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // vsync
     if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return;
    }  
    glViewport(0, 0, W, H);

    // initialize vertex BUFFER OBJECT (VBO)
    unsigned int VBO;
    // first arg is number of buffers we want
    // second arg is the address of the buffer we want to initialize
    // so if we pass >1 we need to pass an array of unsigned ints
    glGenBuffers(1, &VBO);  
    // we just specify to openGL this is a vertex buffer (GL_ARRAY_BUFFER)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    // 1. bind Vertex Array Object
    glBindVertexArray(VAO);

    // 1) Create a fullscreen QUAD with positions + texcoords + color
    struct Vertex {
        float pos[4];     // in_vertex
        float uv[4];      // in_texcoord (z,w unused)
        float col[4];     // in_color
    };

    // Two triangles covering NDC [-1,1], UV [0,1]
    const float U1 = 0.999999f, V1 = 0.999999f; // or std::nextafter(1.0f, 0.0f)
    Vertex quad[6] = {
    {{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}},
    {{ 1,-1,0,1}, {U1,0,0,0}, {1,1,1,1}},
    {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}},
    {{-1,-1,0,1}, {0,0,0,0}, {1,1,1,1}},
    {{ 1, 1,0,1}, {U1,V1,0,0}, {1,1,1,1}},
    {{-1, 1,0,1}, {0,V1,0,0}, {1,1,1,1}},
    };


    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    GLsizei stride = sizeof(Vertex);
    std::size_t off_pos = offsetof(Vertex, pos);
    std::size_t off_uv  = offsetof(Vertex, uv);
    std::size_t off_col = offsetof(Vertex, col);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_pos);

    // attrib 1: in_texcoord (vec4) - we use uv in xy
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_uv);

    // attrib 2: in_color (vec4) - constant white here
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, (void*)off_col);

    const char* vertexShaderSource = "#version 450\n"
    "layout(location = 0) in vec4 in_vertex;   /**< Input vertex coordinates */\n"
    "layout(location = 1) in vec4 in_texcoord; /**< Input texture coordinates */\n"
    "layout(location = 2) in vec4 in_color;    /**< Input colour value */\n"
    "\n"
    "out vec4 texcoord;                        /**< Output texture coordinates */\n"
    "out vec4 color;                           /**< Output color value */\n"

    "void main(void) {\n"
    "    gl_Position = in_vertex;\n"
    "    texcoord    = in_texcoord;\n"
    "    color       = in_color;\n"
    "}\0";

    const char* fragmentShaderSource = "#version 450\n"
    "\n"
    "layout(location = 0) out vec4 out_color;\n"
    "\n"
    "layout(std430, binding = 0) buffer colorLayout\n"
    "{\n"
    "    float data[];\n"
    "} source;\n"
    "\n"
    "uniform bool flip = false;\n"
    "uniform int width = 1000;\n"
    "uniform int height = 800;\n"
    "\n"
    "in vec4 texcoord;\n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    int x = int(texcoord.x * width);\n"
    "    int y;\n"
    "\n"
    "    if(flip)\n"
    "        y = height - 1 - int(texcoord.y * height);\n"
    "    else\n"
    "        y = int(texcoord.y * height);\n"
    "\n"
    "    float r = source.data[0 * width * height + (y * width + x)];\n"
    "    float g = source.data[1 * width * height + (y * width + x)];\n"
    "    float b = source.data[2 * width * height + (y * width + x)];\n"
    "    vec4 color   = vec4(r, g, b, 1);\n"
    "    out_color    = color;\n"
    "}\n";
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glUseProgram(shaderProgram);


    GLuint ssbo=0;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, W*H*3*sizeof(float), pixel_out, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo); // <-- binding = 0 matches the shader

    // 5) Set required uniforms (don’t rely on defaults)
    glUniform1i(glGetUniformLocation(shaderProgram, "width"),  W);
    glUniform1i(glGetUniformLocation(shaderProgram, "height"), H);
    glUniform1i(glGetUniformLocation(shaderProgram, "flip"),   GL_FALSE); // or GL_TRUE



    while (!glfwWindowShouldClose(win)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(win);
        glfwPollEvents();

    }
}

