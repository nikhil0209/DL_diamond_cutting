import glob as glob
import numpy as np
import os
import trimesh


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def process_mesh(mesh):

    # clean up
    #mesh, _ = pymesh.remove_isolated_vertices(mesh)
    #mesh, _ = pymesh.remove_duplicated_vertices(mesh)

    #get elements
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center

    # normalize
    max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
    vertices /= np.sqrt(max_len)

    # get normal vector
    mesh = trimesh.Trimesh(vertices, faces)
    #mesh.add_attribute('face_normal')
    #face_normal = mesh.get_face_attribute('face_normal')
    face_normal = mesh.face_normals

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(vertices)):
        faces_contain_this_vertex.append(set([]))
    centers = []
    corners = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]



        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces = np.concatenate([centers, corners, face_normal], axis=1)
    neighbors = np.array(neighbors)
    return faces, neighbors

if __name__ == '__main__':

    root = 'diamond'
    new_root = 'diamond_simplified/'
    max_faces = 0

    for type in os.listdir(root):
        if '.DS_Store' in type:
            continue
        for phrase in ['train', 'test']:
            type_path = os.path.join(root, type)
            phrase_path = os.path.join(type_path, phrase)
            if not os.path.exists(type_path):
                os.mkdir(os.path.join(new_root, type))
            if not os.path.exists(phrase_path):
                os.mkdir(phrase_path)

            files_outer = glob.glob(os.path.join(phrase_path, 'outer_sphere*.obj'))
            files_rocks = glob.glob(os.path.join(phrase_path, 'rocks*.obj'))
            files_target = glob.glob(os.path.join(phrase_path, 'box_label*.txt'))
            files_outer_ids = [x.split(".")[0].split("_")[-1] for x in files_outer]
            files_rocks_ids_map = {x.split(".")[0].split("_")[-1]:i for i, x in enumerate(files_rocks)}
            files_target_ids_map = {x.split(".")[0].split("_")[-1]:i for i, x in enumerate(files_target)}

            #assert len(files_outer) == len(files_rocks) == len(files_target)
            for i, file in enumerate(files_outer):
                # load mesh
                mesh = trimesh.load_mesh(file)
                faces, neighbors = process_mesh(mesh)

                #get corresponding rocks
                rock_file = files_rocks[files_rocks_ids_map[files_outer_ids[i]]]
                mesh = trimesh.load_mesh(rock_file)
                rock_faces, rock_neighbors = process_mesh(mesh)
                
                #get corresponding targets
                target_file = files_target[files_target_ids_map[files_outer_ids[i]]]
                targets = np.genfromtxt(target_file,delimiter=',').reshape(-1)

                #combine faces
                outer_labels = np.zeros((faces.shape[0],1))
                rock_labels = np.ones((rock_faces.shape[0],1))
                impurity_label = np.concatenate([outer_labels, rock_labels])
                faces = np.concatenate([faces,rock_faces],axis=0)
                neighbors = np.concatenate([neighbors,rock_neighbors],axis=0)

                max_faces = max(max_faces, faces.shape[0])
                _, filename = os.path.split(file)
                filename = 'Diamond_'+str(files_outer_ids[i])
                np.savez(new_root + '/' + type + '/' + phrase + '/' + filename + '.npz',
                         faces=faces, neighbors=neighbors, impurity_label=impurity_label, targets=targets)

                print(file)
    print("Maximum Number of Faces Found = ", max_faces)
