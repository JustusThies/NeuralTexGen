The input data needs to have a structure like this (filenames match the current settings of the scripts. Of course you can adapt the structure and filenames in the scripts):

room1/
	room_uv.off
	ZMojNkEp431_room0__0__.txt
	frames/
		ZMojNkEp431/
			camera/
				0.txt
				1.txt
				...
			color/
				0.jpg
				1.jpg
				...


Below you find information about the content of the files you have to provide:

#####################################
Mesh file containing vertex positions, vertex uvs and the triangle indices
room1/room_uv.off:
------------------------------------
STOFF
61103 100000 0
26.5143 -4.11894 -0.866639 0.249326 0.665014 
.....
3 34639 34638 59419
.....
------------------------------------

######################################
List of images to use (starts with 0)
room1/ZMojNkEp431_room0__0__.txt
------------------------------------
1
458
.....
------------------------------------

######################################
Camera extrinsics and intrinsics per frame (note that the uv_renderer in this repo uses the intrinsics of the first frame)
room1/frames/ZMojNkEp431/camera/0.txt
------------------------------------
0.952432 -0.209116 -0.221682 17.2694 
0.304553 0.626945 0.717069 3.20383 
-0.0109679 -0.750473 0.66081 1.46273 
0 0 0 1 
268.652 0 158.548 0 
0 268.86 125.538 0 
0 0 1 0 
0 0 0 1 
------------------------------------