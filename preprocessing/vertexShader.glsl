#version 430

// input uniforms
uniform mat4 modelview;
uniform mat4 projection;


// input mesh data
layout(location = 0) in vec4  in_position;
layout(location = 1) in vec4  in_normal;
layout(location = 2) in vec4  in_color;
layout(location = 3) in vec2  in_uv;
layout(location = 4) in float  in_mask;

// output to geometry shader
out VertexData
{
  vec3 position;
  vec3 normal;
  vec4 color;
  vec2 uv;
  float mask;
  uint id;
} outData;

void main()
{
  vec4 pos = vec4(in_position.xyz, 1.0);
  pos = modelview * pos;

  outData.position = pos.xyz;
  outData.normal = in_normal.xyz;
  outData.color = in_color;
  outData.uv = in_uv;
  outData.mask = in_mask;
  outData.id = gl_VertexID;

  gl_Position = projection * pos;
}




/*
https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)

Vertex Shaders have the following built-in input variables.

in int gl_VertexID;
in int gl_InstanceID;
in int gl_DrawID; // Requires GLSL 4.60 or ARB_shader_draw_parameters
in int gl_BaseVertex; // Requires GLSL 4.60 or ARB_shader_draw_parameters
in int gl_BaseInstance; // Requires GLSL 4.60 or ARB_shader_draw_parameters

Vertex Shaders have the following predefined outputs.

out gl_PerVertex
{
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
};
*/