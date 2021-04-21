
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
file_count = 1
for j in range(0, 25):
    bpy.ops.mesh.primitive_ico_sphere_add(
    enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    so = bpy.context.active_object
    so.active_material = bpy.data.materials['Transparent']
    so.scale[0] = np.random.uniform(1.15, 1.3)
    so.scale[1] = np.random.uniform(1.15, 1.3)
    so.scale[2] = np.random.uniform(1.15, 1.3)
    make_irregular(so, 1, 1.2)
    bpy.ops.mesh.primitive_brilliant_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), change=False)
    #scaling the diamond
    diamond_scale = np.random.uniform(0.4, 1)
    so = bpy.context.active_object
    so.scale[0] = diamond_scale
    so.scale[1] = diamond_scale
    so.scale[2] = diamond_scale
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
    so.location[0] = translation_x
    so.location[1] = translation_y
    so.location[2] = translation_z

    diamond = so
    total_rocks = 0
    rocks_buffer = []
    for i in range(1, 19):
        total_blobs = 0
        start_time = time.time()
        while True:
            if (time.time() - start_time) > 4 or total_blobs > 9:
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
                if distance_between(object, so) < (object.scale[0] + so.scale[0]):
                    is_valid_blob = False
                    break
            nearest_face_distance = float('inf')
            nearest_face = diamond.data.polygons[0]
            for face in diamond.data.polygons:
                face_distance = distance_between_vectors(so.location, face.center)
                if face_distance < nearest_face_distance:
                    nearest_face_distance = face_distance
                    nearest_face = face
            dp = np.dot(nearest_face.normal, so.location - nearest_face.center)
            print("Dot product is ", dp)
            print("Distance is ", nearest_face_distance)
            if dp <= 0.013:
                is_valid_blob = False
            if is_valid_blob == False:
                bpy.ops.object.delete(use_global=False, confirm=False)
                continue
            total_blobs += 1
            total_rocks += 1
            rocks_buffer.append(so)
            if len(rocks_buffer) > 9:
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
        data_path = "data_4"
        directory = os.path.dirname(blend_file_path)
        f = open(directory + "\\" + data_path + "\\" + "box_label_" + str(file_count) + ".txt", "w")
        f.write(str(diamond.location[0]) + "," + str(diamond.location[1]) + "," + str(diamond.location[2]) + "\n")
        f.write(str(diamond.rotation_euler[0]) + "," + str(diamond.rotation_euler[1]) + "," + str(diamond.rotation_euler[2]) + "\n")
        f.write(str(diamond.scale[0]) + "," + str(diamond.scale[1]) + "," + str(diamond.scale[2]) + "\n")
        f.close()
        target_file = os.path.join(directory + "\\data_4", "automated_with_diamond_" + str(file_count) + ".obj")
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
        diamond.select_set(True)
        diamond_file = os.path.join(directory + "\\data_4", "diamond_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=diamond_file, use_selection=True)
        locations = diamond.location
        scales = diamond.scale
        bpy.ops.object.select_all(action='DESELECT')
        diamond.select_set(True)
        bpy.ops.object.delete()
        target_file = os.path.join(directory + "\\data_4", "automated_without_diamond_" + str(file_count) + ".obj")
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True)
        bpy.ops.mesh.primitive_brilliant_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), change=False)
        bpy.context.active_object.location = locations
        bpy.context.active_object.scale = scales
        bpy.ops.object.select_all(action='DESELECT')
        for cube in cubes:
            cube.select_set(True)
        cubes_file = os.path.join(directory + "\\data_4", "rocks_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=cubes_file, use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
        outer_sphere.select_set(True)
        outer_sphere_file = os.path.join(directory + "\\data_4", "outer_sphere_" + str(file_count) + ".obj")
        bpy.ops.export_scene.obj(filepath=outer_sphere_file, use_selection=True)
        file_count += 1
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    

 
            
        
    




