set -ex
# . prepare_data.sh &
export DISPLAY=:0


################
## input data ##
################
# sequence name
SEQ=room1_us
#SEQ=room1_lqlq
# filename of the input mesh that also has uv coordinates (*.off is supported)
MESH_FILENAME=../datasets/$SEQ/room_uv.off
# directory of the camera information (per frame extrinsics and intrinsics, both stored as 4x4 matrices)
CAMERA_DIR=../datasets/$SEQ/frames/ZMojNkEp431/camera/
# directory of the original images (filenames are assumed to be 0.jpg, 1.jpg, ....)
IMG_DIR=../datasets/$SEQ/frames/ZMojNkEp431/color/
# file list containing the frame ids (per line one id)
FILE_LIST=../datasets/$SEQ/ZMojNkEp431_room0__0__.txt
# image resolution of the original images
WIDTH_ORIG=320
HEIGHT_ORIG=256



#################
## output data ##
#################

# note that the output dimensions can be a multiple of the input dimensions (e.g. to increase sample rate)
WIDTH_RENDERING=1280
HEIGHT_RENDERING=1024

# output folders
OUTPUT_DIR_UV=../datasets/$SEQ/uvs/
OUTPUT_DIR_IMG=../datasets/$SEQ/images/
mkdir -p $OUTPUT_DIR_UV
mkdir -p $OUTPUT_DIR_IMG



###############
## copy data ##
###############

# copy original images and camera files to dataset folder
while read -r file_name; do cp $IMG_DIR${file_name%?}.jpg $OUTPUT_DIR_IMG ; done < $FILE_LIST
while read -r file_name; do cp $CAMERA_DIR${file_name%?}.txt $OUTPUT_DIR_IMG ; done < $FILE_LIST
#cp $IMG_DIR/* $OUTPUT_DIR_IMG/



########################
## render the uv maps ##
########################
CAMERA_DIR_NEW=$OUTPUT_DIR_IMG
cd preprocessing 
./uv_renderer $MESH_FILENAME $CAMERA_DIR_NEW $OUTPUT_DIR_UV $WIDTH_ORIG $HEIGHT_ORIG $WIDTH_RENDERING $HEIGHT_RENDERING
cd ..