# Project Goals

## Overview
This project focuses on building a high-performance Gaussian Renderer with support for both standard 3D and extended 4D (spacetime) Gaussians. The emphasis is on efficiency and speed beyond existing baseline renderers.

---

## "Small Steps"
- [X] Create Datastructures and Parsing code for 3D Gaussians (finished 9/28/2025)
- [X] Implement Interactive Viewer (be able to move around in 3D space, while providing data on view) (finished 10/1/2025)
- [X] Implement Pre-Processing on GPU (culling gaussians, tiling, sorting) (finished on 10/30/2025) Busy semester :(
- [X] Finish 3D Gaussian Rendering Pipeline (finished 11/28/2025)
- [X] Add dynamic resizing (finished 12/31/2025)
- [ ] benchmark performance -> add improvements like arranging gaussians in 3D morton order (done in og paper as well)
- [ ] Add interactive features like drag and drop to load
- [ ] Add 4D gaussian support
- [ ] Add streaming training process

---
## DEMO
<video src="demos/demo_1_2.mp4" controls muted playsinline style="max-width: 100%;"></video>

## OVERALL Objectives
- [ ] Build a 3D Gaussian Renderer  
- [ ] Support 4D Gaussians (Spacetime Gaussians)  
- [ ] Optimize performance to be more efficient/faster than the default 3D Gaussian Renderer  

---

## Deliverables
- Working 3D Gaussian Renderer implementation  
- Extended 4D Gaussian rendering support  
- Benchmarking results showing performance gains  

---

## Success Criteria
- 3D renderer matches or exceeds baseline quality  
- 4D spacetime rendering works correctly with test cases  
- Benchmarks demonstrate improved efficiency over the default implementation  

---

## Stretch Goals
- Real-time rendering support  
- Integration with interactive viewers (VR/AR/WebGL)  
- Memory-efficient Gaussian data storage  

---

## Notes
- Dependencies: CUDA / OpenGL / Vulkan (TBD)  
- Risks: Optimization complexity, GPU compatibility  
- Future Work: Multi-GPU scaling, neural field integration  


"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
