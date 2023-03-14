import os, sys

import bpy

CORNER_LU = 0
CORNER_LD = 1
CORNER_RD = 2
CORNER_RU = 3

# set the path to the FBX file
fbx_file = "/home/mias/Datasets/CarlaRoads/Town01_Road_Road.fbx"

# set the path to the output directory
output_dir = "/home/mias/Projects/pothole_project/SeparateRoadSeg"

# load the FBX file
bpy.ops.import_scene.fbx(filepath=fbx_file)

# loop through each object in the scene
for obj in bpy.context.scene.objects:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    obj_name = obj.name
    obj_file = os.path.join(output_dir, obj_name + ".obj")
    bpy.ops.export_scene.obj(filepath=obj_file, use_selection=True)
