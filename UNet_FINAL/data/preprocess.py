import glob as glob
import numpy as np
import os
import trimesh
import math


if __name__ == '__main__':

    root = '../split_dataset/'
    new_root = '../UNet/diamond_voxelized/'
    max_faces = 0
    resolution = 62
    radius = math.ceil(resolution/2)-1
    center = [0,0,0]
    for type in os.listdir(root):
        if '.DS_Store' in type:
            continue
        if '.' in type or 'test_voxelization' in type or '__pycache' in type:
          continue
        for phrase in ['train', 'test','val']:
            type_path = os.path.join(root, type)
            phrase_path = os.path.join(type_path, phrase)
            if not os.path.join(new_root, type):
              os.mkdir(os.path.join(new_root, type))
            if not os.path.join(new_root, type,phrase):
              os.mkdir(os.path.join(new_root, type,phrase))
            files_outer = glob.glob(os.path.join(phrase_path, 'outer_sphere*.obj'))
            files_rocks = glob.glob(os.path.join(phrase_path, 'rocks*.obj'))
            files_target = glob.glob(os.path.join(phrase_path, 'box_label*.txt'))
            files_outer_ids = [x.split(".obj")[0].split("_")[-1] for x in files_outer]
            files_rocks_ids_map = {x.split(".obj")[0].split("_")[-1]:i for i, x in enumerate(files_rocks)}
            files_target_ids_map = {x.split(".txt")[0].split("_")[-1]:i for i, x in enumerate(files_target)}
            #assert len(files_outer) == len(files_rocks) == len(files_target)
            for i, file in enumerate(files_outer):
                outer_mesh = trimesh.load_mesh(file)
                rock_file = files_rocks[files_rocks_ids_map[files_outer_ids[i]]]
                rocks_mesh = trimesh.load_mesh(rock_file)
                target_file = files_target[files_target_ids_map[files_outer_ids[i]]]
                diamond_labels = np.loadtxt(target_file).reshape(-1)
                targets = diamond_labels[:-2]#since scale is same for x,y,z - pick only one
                targets[3:6]/=math.pi

                pitch = math.ceil((np.max(outer_mesh.bounds[1]-outer_mesh.bounds[0]))*10)/10
                pitch = pitch/(radius*2+1)

                outer = trimesh.voxel.creation.local_voxelize(outer_mesh,center,pitch,radius+1,fill=True)
                rocks = trimesh.voxel.creation.local_voxelize(rocks_mesh,center,pitch,radius+1,fill=False)

                diamond_center = diamond_labels[0]
                grid_vals = (diamond_center - center+0.5*pitch)/pitch
                diamoind_center_grid_point = np.floor(grid_vals)+radius

                _, filename = os.path.split(file)
                filename = 'Voxel_Diamond_'+str(files_outer_ids[i])
                np.savez(new_root + '/' + type + '/' + phrase + '/' + filename + '.npz',
                         outer=outer.matrix, rocks=rocks.matrix, diamoind_center_grid_point=diamoind_center_grid_point, targets = targets, pitch=pitch,radius=radius)

                print(file)
