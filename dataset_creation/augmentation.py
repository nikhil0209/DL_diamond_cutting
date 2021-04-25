import bpy
import os
import math
import numpy as np

def rotate_all_objects(angle, axis):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.rotate(value=angle, orient_axis=axis, orient_type='VIEW', orient_matrix=((1, -0, -0), (-0, 1, -0), (0, -0, 1)), orient_matrix_type='VIEW', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

root = '/Users/Shan/Documents/UMass/DL_diamond_cutting/diamond_dataset'
new_root = '/Users/Shan/Documents/UMass/DL_diamond_cutting/augmented_diamond_dataset'
files = os.listdir(root)
files = [x for x in files if '_with_diamond' in x and '.obj' in x]
last_index = [int(x.split(".")[0].split("_")[-1]) for x in files]
file_count = last_index[-1]+1
for file in files:
    file_loc = os.path.join(root,file)
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)    
    blend_file_path = new_root

    #data agumentation and writin
    for i in range(4):
        r_x = np.random.uniform(0, 2*math.pi)
        r_y = np.random.uniform(0, 2*math.pi)
        r_z = np.random.uniform(0, 2*math.pi)
        #s = np.random.uniform(1, 2)
        objects = bpy.context.selectable_objects
        cubes = []
        for object in objects:
            if object.name.startswith('dobj'):
                diamond = object
            elif object.name.startswith('Icosphere'):
                outer_sphere = object
            elif object.name.startswith('Cube'):
                cubes.append(object)
        rotate_all_objects(r_x, 'X')
        rotate_all_objects(r_y, 'Y')
        rotate_all_objects(r_z, 'Z')

        #save target parameters
        location, rotation, scale = diamond.location, diamond.rotation_euler, diamond.scale
        file_name = os.path.join(blend_file_path,"box_label_" + str(file_count) + ".txt")
        np.savetxt(file_name,np.array([location, rotation, scale]))
            
        #save expected output
        bpy.ops.object.select_all(action='SELECT')
        target_file = os.path.join(blend_file_path, "automated_with_diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        
        #save only diamond
        bpy.ops.object.select_all(action='DESELECT')
        diamond.select_set(True)
        diamond_file = os.path.join(blend_file_path, "diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=diamond_file, use_selection=True)
        
        #save without diamond
        bpy.ops.object.select_all(action='SELECT')
        diamond.select_set(False)
        target_file = os.path.join(blend_file_path, "automated_without_diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        
        #save all cubes
        bpy.ops.object.select_all(action='SELECT')
        diamond.select_set(False)
        outer_sphere.select_set(False)
        cubes_file = os.path.join(blend_file_path, "rocks_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=cubes_file, use_selection=True)
        
        #save outer_sphere
        bpy.ops.object.select_all(action='DESELECT')
        outer_sphere.select_set(True)
        outer_sphere_file = os.path.join(blend_file_path, "outer_sphere_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=outer_sphere_file, use_selection=True)
        
        file_count += 1
        #scale_all_objects(1/s)
        rotate_all_objects(-r_z, 'Z')
        rotate_all_objects(-r_y, 'Y')
        rotate_all_objects(-r_x, 'X')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
