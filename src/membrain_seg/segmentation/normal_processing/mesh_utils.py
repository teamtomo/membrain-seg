import numpy as np
import pyvista as pv
from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    read_nifti,
    store_array_in_csv,
    store_np_in_vtp,
)
from membrain_seg.segmentation.normal_processing.membrane_normals.mesh_class import Mesh
from skimage import measure


def mesh_to_obj_file(vertices, triangle_combos, out_path):
    """Store vertices and triangle combos into an .obj file."""
    out_array = np.array(vertices)
    v_column = np.tile(np.array(["v"]), out_array.shape[0])
    out_array = np.concatenate((np.expand_dims(v_column, 1), out_array), 1)
    add_array = np.array(triangle_combos) + 1
    f_column = np.tile(np.array(["f"]), add_array.shape[0])
    add_array = np.concatenate((np.expand_dims(f_column, 1), add_array), 1)
    out_array = np.concatenate((out_array, add_array), 0)
    store_array_in_csv(out_path, out_array, out_del=" ")


def smoothen_file(out_file, seg_file, smoothen, degree, store_as_obj=False):
    """Smoothen a mesh file."""
    seg = load_tomogram(seg_file)
    verts, faces, t1, t2 = measure.marching_cubes(
        seg, 0.5, step_size=1.5, method="lewiner"
    )
    all_col = np.expand_dims(np.ones(faces.shape[0]), 1) * 3
    all_col = np.array(all_col, dtype=np.int)
    faces = np.concatenate((all_col, faces), 1)
    faces = np.array(faces, dtype=int)
    surf = pv.PolyData(verts, faces)
    surf = surf.smooth(n_iter=smoothen)
    surf = surf.decimate(degree)
    if store_as_obj:
        verts = surf.points
        faces = surf.faces + 1
        faces = np.reshape(faces, (-1, 4))
        faces = faces[:, 1:]
        mesh = Mesh(verts, faces)
        mesh.store_in_file(out_file)
    else:
        surf.save(out_file)


def get_mesh_class_from_pyvista_mesh(pv_mesh):
    """Convert a pyvista mesh to a Mesh object."""
    verts = pv_mesh.points
    faces = pv_mesh.faces + 1
    faces = np.reshape(faces, (-1, 4))
    faces = faces[:, 1:]
    mesh = Mesh(verts, faces)
    return mesh


def convert_file_to_mesh(
    seg_file,
    out_file,
    smoothing=2000,
    degr=0.90,
    store_also_as_obj=False,
    compute_normals=False,
    normal_vtp=None,
):
    """Convert an .nii.gz segmentation to a mesh file."""
    seg = read_nifti(seg_file)
    seg = np.transpose(seg, (1, 2, 0))

    seg = (seg == 1.0) * 1.0

    if 1.0 not in np.unique(seg):
        print(
            "Aborting. Could not find any 1.0 values in segmentation file. \
            Is there no membrane in"
        )
        print(seg_file + " ?")
        return False

    verts, faces, t1, t2 = measure.marching_cubes(
        seg, 0.5, step_size=1.5, method="lewiner"
    )
    all_col = np.expand_dims(np.ones(faces.shape[0]), 1) * 3
    all_col = np.array(all_col, dtype=int)
    faces = np.concatenate((all_col, faces), 1)
    faces = np.array(faces, dtype=int)
    surf = pv.PolyData(verts, faces)
    surf = surf.smooth(n_iter=smoothing)
    if degr is not None:
        surf_coarse = surf.decimate(degr)
        if compute_normals:
            surf_coarse = surf_coarse.compute_normals(flip_normals=True)
            normals = surf_coarse.get_array("Normals")
            centers = surf_coarse.cell_centers()
            center_coords = centers.points.copy()
            print(center_coords.shape, normals.shape)
            center_coords = np.concatenate((center_coords, normals), axis=1)
            store_np_in_vtp(center_coords, normal_vtp)
            store_array_in_csv(normal_vtp[:-3] + "csv", center_coords)
        surf_coarse.save(out_file)
        if store_also_as_obj:
            mesh = get_mesh_class_from_pyvista_mesh(surf_coarse)
            mesh.store_in_file(out_file[:-3] + "obj")
            print("Saving", out_file[:-3] + "obj")
    else:
        surf.save(out_file)
        if store_also_as_obj:
            mesh = get_mesh_class_from_pyvista_mesh(surf)
            mesh.store_in_file(out_file[:-3] + "obj")
            print("Saving", out_file[:-3] + "obj")
