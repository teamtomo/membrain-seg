import os

import numpy as np
from membrain_seg.segmentation.dataloading.data_utils import (
    get_csv_data,
    read_nifti,
    write_nifti,
)
from membrain_seg.segmentation.membrane_normals.mesh_utils import convert_file_to_mesh
from sklearn.neighbors import NearestNeighbors

task_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBra\
    in-seg-normalAugs/Task143_cryoET7"
task_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBra\
    in-seg-normalAugs/Task529_ChlamyV3_HDCR"

tr_val = "Val"

if tr_val == "Tr":
    gt_dir_nifti = os.path.join(task_dir, "labelsTr")
    gt_dir = os.path.join(task_dir, "labelsTr_as_MRC")
    mesh_dir = os.path.join(task_dir, "labelsTR_as_meshes")
    vecs_dir = os.path.join(task_dir, "labelsTr_vecs")
elif tr_val == "Val":
    gt_dir_nifti = os.path.join(task_dir, "labelsVal")
    gt_dir = os.path.join(task_dir, "labelsVal_as_MRC")
    mesh_dir = os.path.join(task_dir, "labelsVal_as_meshes")
    vecs_dir = os.path.join(task_dir, "labelsVal_vecs")

for directory in [gt_dir, mesh_dir, vecs_dir]:
    os.makedirs(directory, exist_ok=True)

min_dist_thres = 3.0


for filetoken in os.listdir(gt_dir_nifti):
    # convert_nifti_to_mrc(os.path.join(gt_dir_nifti, filetoken),
    #   os.path.join(gt_dir, filetoken[:-6] + 'mrc'))
    filetoken = filetoken[:-7]
    print("Processing", filetoken)
    # if not "tomo17_patch001" in filetoken: continue
    # mrc_file = os.path.join(gt_dir, filetoken)
    nifti_file = os.path.join(gt_dir_nifti, filetoken + ".nii.gz")
    out_mesh = os.path.join(mesh_dir, filetoken + ".vtp")
    normal_vtp = os.path.join(mesh_dir, filetoken + "2.vtp")
    normal_csv = os.path.join(mesh_dir, filetoken + "2.csv")
    out_vec_file1 = os.path.join(vecs_dir, filetoken + "_norm1.nii.gz")
    out_vec_file2 = os.path.join(vecs_dir, filetoken + "_norm2.nii.gz")
    out_vec_file3 = os.path.join(vecs_dir, filetoken + "_norm3.nii.gz")
    out = convert_file_to_mesh(
        nifti_file,
        out_mesh,
        degr=0.8,
        store_also_as_obj=True,
        dilation_and_erosion=False,
        compute_normals=True,
        normal_vtp=normal_vtp,
    )
    if out is False:
        print("Did not find any membrane in", nifti_file, ". Setting normals to 0.")
        seg = read_nifti(nifti_file)
        normal_labels = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3))
    else:
        coords_and_vecs = np.array(get_csv_data(normal_csv), dtype=float)
        coords = coords_and_vecs[:, :3]
        normals = coords_and_vecs[:, 3:]
        seg = read_nifti(nifti_file)
        ones = np.argwhere(seg == 1.0)
        Vnx = np.arange(seg.shape[0])
        Vny = np.arange(seg.shape[1])
        Vnz = np.arange(seg.shape[2])
        [X, Y, Z] = np.meshgrid(np.array(Vnx), np.array(Vny), np.array(Vnz))
        X = np.expand_dims(X, -1)
        Y = np.expand_dims(Y, -1)
        Z = np.expand_dims(Z, -1)
        ones = np.concatenate((X, Y, Z), axis=-1)
        ones = ones.reshape(-1, 3)
        nn_entity = NearestNeighbors(n_neighbors=1).fit(coords)
        min_args = nn_entity.kneighbors(ones)
        corr_normals = normals[min_args[1]]
        corr_normals = corr_normals.reshape(seg.shape[0], seg.shape[1], seg.shape[2], 3)
        min_dists = min_args[0].reshape(seg.shape[0], seg.shape[1], seg.shape[2])
        mask = min_dists < min_dist_thres
        normal_labels = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3))
        normal_labels[mask] = corr_normals[mask]

    for k in range(3):
        normal_labels[:, :, :, k] = np.transpose(normal_labels[:, :, :, k], (2, 1, 0))
    write_nifti(out_vec_file2, normal_labels[:, :, :, 0])
    write_nifti(out_vec_file1, normal_labels[:, :, :, 1])
    write_nifti(out_vec_file3, normal_labels[:, :, :, 2])
