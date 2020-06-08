import os
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import sys
sys.path.append("/eva_data/psa/code/data_utils")
from utils import *

warnings.filterwarnings('ignore')
root_dirs = {"ModelNet40": "/eva_data/psa/datasets/PointNet/ModelNet40_pcd",
             "ModelNet10": "/eva_data/psa/datasets/PointNet/ModelNet10_pcd"}
obj_name_files = {"ModelNet40": "modelnet40_shape_names.txt",
                  "ModelNet10": "modelnet10_shape_names.txt"}

class ModelNetDataLoader(Dataset):
    def __init__(self,  num_in_points=1024, num_out_points=1024, split="train", dataset_mode="ModelNet40", sparsify_mode="zorder"):
        self.num_in_points = num_in_points
        self.num_out_points = num_out_points
        self.sparsify_mode = sparsify_mode

        assert (dataset_mode == "ModelNet40" or dataset_mode == "ModelNet10"), "PLZ verify dataset_mode should be [ModelNet40, ModelNet10]"
        root_dir = root_dirs[dataset_mode]
        obj_name_file = obj_name_files[dataset_mode]

        # Load class name file and Create target table
        self.class_file = os.path.join(root_dir, obj_name_file)
        self.class_names = [line.rstrip() for line in open(self.class_file)]
        self.target_table = dict(zip(self.class_names, range(len(self.class_names))))

        # Create datapath -> (shape_name, shape_pcd_file_pat)
        assert (split == "train" or split == "test"), "PLZ verify split should be [train, test]"
        self.datapath = [] # list of (shape_name, shape_pcd_file_path) tuple
        for class_name in self.class_names:
            file_dir = os.path.join(root_dir, class_name, split)
            filenames = os.listdir(file_dir)
            for filename in filenames:
                file_path = os.path.join(file_dir, filename)
                self.datapath.append((class_name, file_path))
        
        print("The size of %s data is %d" % (split, len(self.datapath)))

    def __getitem__(self, index):
        classname_target_pair = self.datapath[index]

        # Get target
        target = self.target_table[classname_target_pair[0]]
        target = np.array([target]).astype(np.int32)
        
        # Get points
        points = read_pcd(classname_target_pair[1])
        points = pcd_normalize(points) # Normalize points
        complete = random_sample(points, num_points=self.num_in_points)

        # Various sparsify mode
        points_zorder = get_zorder_sequence(points) # Sort input points with z values

        if self.sparsify_mode == "zorder":
            partial = keep_zorder(points_zorder, num_points=self.num_in_points)
        elif self.sparsify_mode == "multizorder":
            partial = keep_multizorder(points_zorder, num_points=self.num_in_points)
        
        return partial, target, complete

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader("/eva_data/psa/datasets/PointNet/ModelNet40_pcd", split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
