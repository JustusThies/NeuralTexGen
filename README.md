# NeuralTexGen

This code is meant for simple image-space texture optimization.
As input you have to provide uv-renderings of the target scene.
Based on these 'UV-maps' and the color images one can use PyTorch with the differentiable bilinear sampling to 'train' a color texture under a specific error metric.
You can have arbitrary image losses like L1, L2 or style losses (like VGG content & style loss).

## Training Data
### 3D Reconstruction
You need to have a reconstructed mesh of the target object/scene along with the camera parameters for each image.
Some approaches that I used in the past:
- KinectFusion with MarchingCubes
- VoxelHashing (Github repo)[https://github.com/niessner/VoxelHashing]
- BundleFusion (Github repo)[https://github.com/niessner/BundleFusion]
- Colmap (Project Page)[https://colmap.github.io/]

### Parametrization
As we want to optimize for a texture, we have to define a texture space including the mapping from the vertices of the mesh to the texture space (Parametrization).
To this end you can use trivial per triangle parametrization (not recommended) or some more advanced techniques to compute the parametrization.
You can use MeshLab for trivial parametrizations, Blender or the UV parametrization tool from Microsoft (Github repo)[https://github.com/microsoft/UVAtlas].

### Rendering of the UV maps
Use a renderer of your choice to render the per frame uvs using the mesh and the camera parameters.
For example, you can use an headless 'EGL-based' OpenGL renderer on your server.

### Summary
In the end you should have training pairs consisting of a uv map and a original color image which serves as target image.

## Optimization

### Define a loss function
First of all, you have to choose a loss function (or you write your own).

### Optimize aka 'Train'
Start optimization over the entire training data corpus.


