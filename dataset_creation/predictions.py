# -----------------------------
# change paths in line 9-11
# -----------------------------

import bpy
import os
import numpy as np 

root = 'MeshNet/diamond/diamond1/train/'
out_root = 'MeshNet/diamond/diamond1/train/'
out_save_root = 'MeshNet/diamond/diamond1/train/'

files = os.listdir(root)
files = [x for x in files if 'automated_without_' in x]
for file in files:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("path exists: ",os.getcwd())
    file_loc = os.path.join(root,file)
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.selected_objects[0] ####<--Fix

    #load output from file
    index = file.split(".")[0].split("_")[-1]
    output_file = os.path.join(out_root,'test'+str(index)+'.npy')
    params = np.load(output_file)
    location = (params[0],params[1],params[2])
    rotation = (params[3],params[4],params[5])
    scale = (params[6],params[6],params[6])
    bpy.ops.mesh.primitive_brilliant_add(align='WORLD', location=location, rotation=rotation, change=False)
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    
    target_file = os.path.join(out_save_root, "predicted_diamond_" + str(index) + ".obj")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
    print('Processed name: ', output_file)
