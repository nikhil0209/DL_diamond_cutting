import os
from shutil import copyfile

dataset_root = "diamond_dataset"
new_root = "split_dataset/"
test = 0.15
val = 0.15

if not os.path.exists(new_root):
    os.mkdir(new_root)

files = os.listdir(dataset_root)
n_files = len([x for x in files if '_with_diamond' in x and '.obj' in x])
n_test = int(n_files*test)
n_val = int(n_files*val)
n_train = n_files - n_test - n_val
splits = {"train":n_train,"val":n_val,"test":n_test}

file_count = 0
for k, v in splits.items():
    k_root = os.path.join(new_root,k)
    if not os.path.exists(k_root):
        os.mkdir(k_root)
    for i in range(v):
        suffix = "_"+str(file_count)+".obj"
        data_files = [x for x in files if x.endswith(suffix)]
        for f in data_files:
            src = os.path.join(dataset_root,f)
            dst = os.path.join(k_root,f)
            copyfile(src, dst)

        file_count+=1

if file_count == n_files:
    print("All files processed")
else:
    print("Check Code")