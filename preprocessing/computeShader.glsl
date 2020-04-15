#version 430 compatibility

#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable


/*struct VtxData {
   vec3  position;
   float pad0;
   vec3  normal;
   float pad1;
   vec4  color;
   vec2  uv;
   vec2  pad2;
}; // ^^ 16 * sizeof (GLfloat) per-vtx

layout (std140, binding = 3) buffer VertexBuffer {
   VtxData verts [];
};*/
struct VtxData {
   vec3  position;
   vec3  normal;
   vec4  color;
   vec2  uv;
   float mask;
}; // ^^ 16 * sizeof (GLfloat) per-vtx

layout (binding = 3) buffer VertexBuffer {
   VtxData verts [];
};

layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

void main()
{
    uint globalId = gl_GlobalInvocationID.x;


    //vec4 c = verts[ globalId ].color;

    //verts[ globalId ].color = vec4(1.0, 0.0, 0.0, 1.0);

}