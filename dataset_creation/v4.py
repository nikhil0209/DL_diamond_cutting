import bpy
import numpy as np
import math
import time
import os
import bmesh


from mathutils import Matrix


'''
# Snippet install scipy in blender
import subprocess
py_exec = bpy.app.binary_path_python
# ensure pip is installed & update
subprocess.call([str(py_exec), "-m", "ensurepip", "--user"])
subprocess.call([str(py_exec), "-m", "pip", "install", "--upgrade", "pip"])
# install dependencies using pip
# dependencies such as 'numpy' could be added to the end of this command's list
subprocess.call([str(py_exec),"-m", "pip", "install", "--user", "scipy"])
#/Applications/Blender.app/Contents/Resources/2.92/python/lib/python3.7/
'''
#import scipy

def make_irregular(obj, irregular_min_val, irregular_max_val):
    vertex_set = obj.data.vertices
    center = obj.location
    for vertex in vertex_set:
        vertex.co = center + np.random.uniform(irregular_min_val, irregular_max_val)*(vertex.co - center)

def translate_diamond():
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


def get_diamond_corner_points(diamond):
    vertices = np.array([x.co for x in diamond.data.vertices])
    zmin = 1e5
    xmax = -1e5
    zmax = -1e5
    zmin = np.min(vertices[:,2])
    zmax = np.max(vertices[:,2])
    xmax = np.max(vertices[:,0])

    xmax_z_lower = np.min(vertices[vertices[:,0]==xmax][:,2])
    xmax_z_upper = np.max(vertices[vertices[:,0]==xmax][:,2])
    zmax_x = vertices[vertices[:,2]==zmax][0][0]
    zmin_x = vertices[vertices[:,2]==zmin][0][0]
    return zmin,zmax,xmax,xmax_z_lower,xmax_z_upper,zmax_x,zmin_x

def sample_points(zmin,zmax,xmax,xmax_z_lower,xmax_z_upper,zmax_x, zmin_x, margin):
    def get_line_margin(margin,x1,x2,y1,y2):
        abs_slope = abs((y2-y1)/(x2-x1))
        multiplier_x = math.sin(math.atan(abs_slope))
        multiplier_y = math.cos(math.atan(abs_slope))
        max_translate = max(margin/multiplier_x,margin/multiplier_y)
        return max_translate
    
    def get_from_line(z,x1,x2,y1,y2):
        slope = (y2-y1)/(x2-x1)
        x = (z-y1)/slope + x1
        return x

    z = np.random.uniform(-1,1)
    #print(z)
    origin = [zmin_x, zmin_x, xmax_z_lower]
    max_x_radius = (1.1-z**2)**0.5
    pts = np.random.uniform(-1,1,size=2)
    pts /= np.linalg.norm(pts)
    if z<zmin-margin/2:
        print("sample anywhere")
        sample_radius = np.random.uniform(0,max_x_radius)
    elif z<=xmax_z_lower:
        print("lower part of diamond")
        max_translate = get_line_margin(margin,zmin_x,xmax,zmin,xmax_z_lower)
        smpl_xmin = get_from_line(z,zmin_x,xmax+max_translate,zmin-max_translate,xmax_z_lower)
        sample_radius = np.random.uniform(smpl_xmin,max_x_radius)
    elif z<= zmax+margin/2:
        print("crown")
        max_translate = get_line_margin(margin,zmax_x,xmax,zmax,xmax_z_upper)
        smpl_xmin = get_from_line(z,zmax_x,xmax+max_translate,zmax+max_translate,xmax_z_upper)
        sample_radius = np.random.uniform(smpl_xmin,max_x_radius)
        x=y=0
    else:
        print("sample anywhere")
        sample_radius = np.random.uniform(0,max_x_radius)
        
    pts *= sample_radius 
    x, y = pts[0], pts[1] 
    return (x,y,z)

def manage_rock_distances(centers, rock_margin):
    distances = np.sqrt(np.sum((centers[None, :] - centers[:, None])**2, -1))
    distances = distances > rock_margin
    triu = np.triu(distances)
    idxs = [i for i,x in enumerate(np.sum(triu,axis=0)) if x==i]
    centers = centers[idxs]
    return centers

file_count = 1
for j in range(0, 1):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_ico_sphere_add(
    enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    so = bpy.context.active_object
    #so.active_material = bpy.data.materials['Transparent']
    s_x = np.random.uniform(1.17, 1.3)
    s_y = np.random.uniform(1.17, 1.3)
    s_z = np.random.uniform(1.17, 1.3)
    bpy.ops.transform.resize(value=(s_x, s_y, s_z), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    make_irregular(so, 1, 1.1)
    bpy.ops.object.modifier_add(type='TRIANGULATE')



    bpy.ops.mesh.primitive_brilliant_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), change=False)
    #scaling the diamond
    diamond_scale = np.random.uniform(0.4, 1)
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    so = bpy.context.active_object
    #bpy.ops.transform.resize(value=(diamond_scale, diamond_scale, diamond_scale), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    #translating the diamond
    #translate_diamond()
    diamond = so
    
    rock_scale = 0.035
    margin = rock_scale * 2 * math.sqrt(2)
    zmin,zmax,xmax,xmax_z_lower,xmax_z_upper,zmax_x,zmin_x = get_diamond_corner_points(diamond)
    centers = []
    for i in range(200):
        location = sample_points(zmin,zmax,xmax,xmax_z_lower,xmax_z_upper,zmax_x, zmin_x, margin)
        centers.append(location)
    
    centers = manage_rock_distances(np.array(centers), margin)
    print("N CENTERS ", len(centers))
    #manage_rock_distances(centers, rock_margin)
    for location in centers:
        bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=location)
        rock_scale_ = np.random.uniform(0.014, rock_scale)
        bpy.ops.transform.resize(value=(rock_scale_,rock_scale_, rock_scale_), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        so = bpy.context.active_object
        make_irregular(so, 0.8, 1)
        so.rotation_euler.rotate_axis("Z", np.random.uniform(0, 2*math.pi))
        so.rotation_euler.rotate_axis("X", np.random.uniform(0, 2*math.pi))
        so.rotation_euler.rotate_axis("Y", np.random.uniform(0, 2*math.pi))
    