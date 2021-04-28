import glob as glob
import numpy as np
import os
import trimesh
import math

if __name__ == '__main__':

    root = '/content/drive/MyDrive/DiamondPrediction/object_files'
    new_root = '/content/drive/MyDrive/DiamondPrediction/PointNet/processed_data/'
    max_faces = 0

    for phrase in os.listdir(root):
        phrase_path = os.path.join(root, phrase)
        if not os.path.join(new_root, phrase):
            os.mkdir(os.path.join(new_root, phrase))
        files_rocks = glob.glob(os.path.join(phrase_path, 'rocks*.obj'))
        files_target = glob.glob(os.path.join(phrase_path, 'box_label*.txt'))
        files_rock_ids = [x.split(".")[0].split("_")[-1] for x in files_rocks]
        files_target_ids_map = {x.split(".")[0].split("_")[-1]:i for i, x in enumerate(files_target)}

        for i, file in enumerate(files_rocks):
            # load mesh
            mesh = trimesh.load_mesh(file)
            rock_vertices = mesh.vertices.copy()
                 
            #get corresponding targets
            #target = (center cordinates, rotation x, rotation y, rotation z, scale x, scale y, scale z)
            target_file = files_target[files_target_ids_map[files_rock_ids[i]]]
            targets = np.genfromtxt(target_file,delimiter=' ').reshape(-1)[:-2]#since scale is same for x,y,z - pick only one
            filename = 'processed_data_'+str(files_rock_ids[i])
            np.savez(new_root + '/' + phrase + '/' + filename + '.npz',
                         vertices=rock_vertices, targets=targets)

            print(file)
