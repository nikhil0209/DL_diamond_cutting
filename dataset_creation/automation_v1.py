
import bpy
import numpy as np
import math
import time
import os
import bmesh

def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    # V2.79 : bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    
def distance_between(object_one, object_two):
    x1 = object_one.location[0]
    y1 = object_one.location[1]
    z1 = object_one.location[2]
    x2 = object_two.location[0]
    y2 = object_two.location[1]
    z2 = object_two.location[2]
    return ((x1 - x2)**2 + (y1 -y2)**2 + (z1 - z2)**2)**0.5 

def distance_between_vectors(a, b):
    c = a - b
    return (c[0]**2 + c[1]**2 + c[2]**2)**0.5

def make_irregular(obj, irregular_min_val, irregular_max_val):
    vertex_set = obj.data.vertices
    center = obj.location
    for vertex in vertex_set:
        vertex.co = center + np.random.uniform(irregular_min_val, irregular_max_val)*(vertex.co - center)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
file_count = 13
for j in range(0, 247):
    bpy.ops.mesh.primitive_ico_sphere_add(
    enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    so = bpy.context.active_object
    #so.active_material = bpy.data.materials['Transparent']
    s_x = np.random.uniform(1.1, 1.15)
    s_y = np.random.uniform(1.1, 1.15)
    s_z = np.random.uniform(1.1, 1.15)
    bpy.ops.transform.resize(value=(s_x, s_y, s_z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    make_irregular(so, 1, 1.1)
    bpy.ops.mesh.primitive_brilliant_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), change=False)
    #scaling the diamond
    diamond_scale = np.random.uniform(0.4, 1)
    so = bpy.context.active_object
    bpy.ops.transform.resize(value=(diamond_scale, diamond_scale, diamond_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    #translating the diamond
    critical_points = [[0, 0, -diamond_scale*0.9], [0, diamond_scale, 0], [diamond_scale, 0, 0], 
    [0, -diamond_scale, 0], [-diamond_scale, 0, 0], [0, diamond_scale*0.6, 0.4*diamond_scale], [diamond_scale*0.6, 0, 0.4*diamond_scale], 
    [0, -diamond_scale*0.6, 0.4*diamond_scale], [-diamond_scale*0.6, 0, 0.4*diamond_scale]]
    translation_x_neg = -float('inf')
    translation_x_pos = float('inf')
    for i in range(0, 9):
        a = (1 - (critical_points[i][1])**2 -(critical_points[i][2])**2)**0.5 - critical_points[i][0]
        translation_x_pos = min(translation_x_pos, a)
        a = -1*((1 - (critical_points[i][1])**2 -(critical_points[i][2])**2)**0.5) - critical_points[i][0]
        translation_x_neg = max(translation_x_neg, a)
    translation_x = np.random.uniform(translation_x_neg, translation_x_pos)
    translation_y_neg = -float('inf')
    translation_y_pos = float('inf')
    for i in range(0, 9):
        a = (1 - (critical_points[i][0] + translation_x)**2 -(critical_points[i][2])**2)**0.5 - critical_points[i][1]
        translation_y_pos = min(translation_y_pos, a)
        a = -1*((1 - (critical_points[i][0] + translation_x)**2 -(critical_points[i][2])**2)**0.5) - critical_points[i][1]
        translation_y_neg = max(translation_y_neg, a)
    translation_y = np.random.uniform(translation_y_neg, translation_y_pos)
    translation_z_neg = -float('inf')
    translation_z_pos = float('inf')
    for i in range(0, 9):
        a = (1 - (critical_points[i][0] + translation_x)**2 -(critical_points[i][1] + translation_y)**2)**0.5 - critical_points[i][2]
        translation_z_pos = min(translation_z_pos, a)
        a = -1*((1 - (critical_points[i][0] + translation_x)**2 -(critical_points[i][1] + translation_y)**2)**0.5) - critical_points[i][2]
        translation_z_neg = max(translation_z_neg, a)
    translation_z = np.random.uniform(translation_z_neg, translation_z_pos)
    bpy.ops.transform.translate(value=(translation_x, translation_y, translation_z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

    diamond = so
    total_rocks = 0
    rocks_buffer = []
    for i in range(1, 19):
        total_blobs = 0
        start_time = time.time()
        while True:
            if (time.time() - start_time) > 10 or total_blobs > 20:
                break
            print("Total rocks created are " + str(total_rocks))
            blob_x = np.random.uniform(-1 + i*0.1, -0.9 + i*0.1)
            blob_y = np.random.uniform(-1*(1 - blob_x**2)**0.5 + 0.1, (1 - blob_x**2)**0.5 - 0.1)
            blob_z = np.random.uniform(-1*(1 - blob_x**2 - blob_y**2)**0.5 + 0.1, (1 - blob_x**2 - blob_y**2)**0.5 - 0.1)
            bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(blob_z, blob_y, blob_x), scale=(1, 1, 1))
            so = bpy.context.active_object
            rock_scale = np.random.uniform(0.01414, 0.035)
            so.scale[0] = rock_scale
            so.scale[1] = rock_scale
            so.scale[2] = rock_scale
            make_irregular(so, 0.8, 1.1)
            so.rotation_euler[0] = np.random.uniform(0, 2*math.pi)
            so.rotation_euler[1] = np.random.uniform(0, 2*math.pi)
            so.rotation_euler[2] = np.random.uniform(0, 2*math.pi)
            is_valid_blob = True
            for object in rocks_buffer:
                if distance_between(object, so) < (object.scale[0] + rock_scale):
                    is_valid_blob = False
                    break
            _, nearest_point, normal, _ = diamond.closest_point_on_mesh(so.location)
            print("Nearest point", nearest_point)
            print("Normal", normal)
            is_valid_blob = is_valid_blob and (np.dot(normal, so.location - nearest_point) > 0.02)
            relative_dis = distance_between_vectors(so.location, diamond.location) - distance_between_vectors(nearest_point, diamond.location)
            print("Relative distance is", relative_dis)
            is_valid_blob = is_valid_blob and (relative_dis > 0.1)
            print("Rock center is at", so.location)
            if is_valid_blob == False:
                bpy.ops.object.select_all(action='DESELECT')
                so.select_set(True)
                bpy.ops.object.delete(use_global=False, confirm=False)
                continue
            total_blobs += 1
            total_rocks += 1
            rocks_buffer.append(so)
            if len(rocks_buffer) > 20:
                rocks_buffer.pop(0)
                
    #data agumentation and writing    
    all_angles = [[0, 0, 0], [math.pi/2, 0, 0], [0, math.pi/2, 0], [0, 0, math.pi/2]]
    l = len(all_angles)
    for i in range(0, l):
        objects = bpy.context.selectable_objects
        cubes = []
        for object in objects:
            triangulate_object(object)
            if object.name.startswith('dobj'):
                diamond = object
            elif object.name.startswith('Icosphere'):
                outer_sphere = object
            elif object.name.startswith('Cube'):
                cubes.append(object)
            object.rotation_euler = all_angles[i]
        blend_file_path = bpy.data.filepath
        data_path = "data_6"
        directory = os.path.dirname(blend_file_path)
        f = open(directory + "\\" + data_path + "\\" + "box_label_" + str(file_count) + ".txt", "w")
        f.write(str(translation_x) + "," + str(translation_y) + "," + str(translation_z) + "\n")
        f.write(str(all_angles[i][0]) + "," + str(all_angles[i][1]) + "," + str(all_angles[i][2]) + "\n")
        f.write(str(diamond_scale) + "," + str(diamond_scale) + "," + str(diamond_scale) + "\n")
        f.close()
        target_file = os.path.join(directory + "\\data_6", "automated_with_diamond_" + str(file_count) + ".obj")
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
        diamond.select_set(True)
        diamond_file = os.path.join(directory + "\\data_6", "diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=diamond_file, use_selection=True)
        bpy.ops.object.select_all(action='SELECT')
        diamond.select_set(False)
        target_file = os.path.join(directory + "\\data_6", "automated_without_diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
        for cube in cubes:
            cube.select_set(True)
        cubes_file = os.path.join(directory + "\\data_6", "rocks_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=cubes_file, use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
        outer_sphere.select_set(True)
        outer_sphere_file = os.path.join(directory + "\\data_6", "outer_sphere_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=outer_sphere_file, use_selection=True)
        file_count += 1
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    

 
            
        
    




