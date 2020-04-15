#version 430

#define PER_FACE_NORMAL

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

// input from vertex shader
in VertexData
{
  vec3 position;
  vec3 normal;
  vec4 color;
  vec2 uv;
  float mask;
  uint id;
} inData[];


// output to fragment shader
out FragmentData
{
    vec3 position;
    vec3 normal;
    vec4 color;
    vec2 uv;
    float mask;
    vec3 baryCoord;
    flat uvec3 vertexIds;
} fragData;

void main()
{
    uvec3 vertexIds = uvec3(inData[0].id, inData[1].id, inData[2].id);

    vec3 bary[3];
    bary[0] = vec3(1.0, 0.0, 0.0);
    bary[1] = vec3(0.0, 1.0, 0.0);
    bary[2] = vec3(0.0, 0.0, 1.0);

#ifdef PER_FACE_NORMAL
    vec3 normal;
    normal = cross(inData[1].position - inData[0].position, inData[2].position - inData[0].position);
    normal = normalize(normal);
#endif

    int i;
    for(i = 0;i < gl_in.length();i++)
    {      
      fragData.position = inData[i].position;
      #ifdef PER_FACE_NORMAL
      fragData.normal   = normal; // per face normal
      #else
      fragData.normal   = inData[i].normal;
      #endif
      fragData.color    = inData[i].color;
      fragData.uv       = inData[i].uv;
      fragData.mask     = inData[i].mask;
      fragData.baryCoord = bary[i];
      fragData.vertexIds = vertexIds;
      gl_Position = gl_in[i].gl_Position;
      EmitVertex();
    }

    EndPrimitive();
}


/*
https://www.khronos.org/opengl/wiki/Geometry_Shader

- There are some GS input values that are based on primitives, not vertices. These are not aggregated into arrays. These are:

in int gl_PrimitiveIDIn;
in int gl_InvocationID; // Requires GLSL 4.0 or ARB_gpu_shader5

*/