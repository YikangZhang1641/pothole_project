import open3d as o3d
import pymesh
import numpy as np
import scipy.io as scio
import cv2
import math
import os, sys
from collections import deque
import triangle as tr
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Lock, Pool, Manager
import shutil
import trimesh
from tqdm import tqdm
import noise
import threading

try:
    multiprocessing.set_start_method('forkserver')
except RuntimeError:  # context has already been set
    pass


CPU_CORES = 23
BORDER_MARKER = -9.0
print_progress = False


CORNER_LU = 0
CORNER_LD = 1
CORNER_RD = 2
CORNER_RU = 3

CUR_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
OUTPUT_PATH = os.path.join(CUR_PATH, "RoadPitMesh")

PIT_LIST = os.listdir(os.path.join(CUR_PATH, "PitMeshNoCorner")) 
ROADSEG_FOLDER = os.path.join(CUR_PATH, "SeparateRoadSeg")


PITS_TO_EMBED = 800
MAX_PIT_ONE_SEG = 3
THRESHOLD_SPLIT = 0.01

def plot_obj(road, mesh=None, border_edges=None, opt=None):
    to_show = []
    if road:
        triangles = np.asarray(road.triangles)
        road_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))

        road_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(road.vertices),
            lines=o3d.utility.Vector2iVector(road_edges),
        )

        to_show = [road, road_lines]
    if mesh:
        triangles = np.asarray(mesh.triangles)
        mesh_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        if border_edges is not None:
            mesh_edges = border_edges
        mesh_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(mesh.vertices),
            lines=o3d.utility.Vector2iVector(mesh_edges),
        )
        
        to_show = to_show + [mesh, mesh_lines]
    if opt:
        to_show = to_show + opt
        
    o3d.visualization.draw_geometries(to_show)


def task_func(args):
    _name, Perlin_seed, pits_in_seg = args
    if print_progress:
        print(_name)
    # _name = "Road_Road_Town01_624.obj"
    road_seg = o3d.io.read_triangle_mesh(os.path.join(ROADSEG_FOLDER, _name))
    road_seg.remove_duplicated_vertices()
    road_seg.vertices = o3d.utility.Vector3dVector(np.array(road_seg.vertices)[:, [2,0,1]])


    Xs = np.array(road_seg.vertices)[:,0]
    Ys = np.array(road_seg.vertices)[:,1]
    RoadBoundary = [Xs.min(), Xs.max(), Ys.min(), Ys.max()]

    collision_manager = trimesh.collision.CollisionManager()

    ################ Generate Collision Testing Scene ################
    scene = o3d.t.geometry.RaycastingScene()
    road_tri = o3d.t.geometry.TriangleMesh.from_legacy(road_seg)
    road_id = scene.add_triangles(road_tri)

    collision_manager = trimesh.collision.CollisionManager()
    new_road = o3d.geometry.TriangleMesh()
    mesh = o3d.geometry.TriangleMesh()

    pits_to_embed = []
    centers = []
    faces_to_delete = set()

    generated = 0
    


    while generated < pits_in_seg:
    #     new_road.clear()
    #     mesh.clear()
        
        pit_name = PIT_LIST[np.random.randint(len(PIT_LIST))]
        x = np.random.uniform(RoadBoundary[0], RoadBoundary[1])
        y = np.random.uniform(RoadBoundary[2], RoadBoundary[3])
        yaw = np.random.uniform() * np.pi

        mesh = o3d.io.read_triangle_mesh(os.path.join(CUR_PATH, "PitMeshNoCorner", pit_name))
        
        tmp_v = np.array(mesh.vertices)
        border_vertex_id = np.where(tmp_v[:,2] < BORDER_MARKER)[0]
        tmp_v[border_vertex_id, 2] = 0
        mesh.vertices = o3d.utility.Vector3dVector(tmp_v)
        
        mesh_triangles = np.array(mesh.triangles)
        border_edges = set()
        for c in range(mesh_triangles.shape[0]):
            for i,j in [[0,1], [1,2], [2,0]]:
                s = min(mesh_triangles[c][i], mesh_triangles[c][j])
                e = max(mesh_triangles[c][i], mesh_triangles[c][j])
                if s in border_vertex_id and e in border_vertex_id:           
                    border_edges.add((s, e))
        
        
        log_prefix = "( " + str(generated) + " / " + str(PITS_TO_EMBED) + ") embedding " + pit_name

        mesh.translate((x, y, 0))
        Rot = mesh.get_rotation_matrix_from_xyz((0, 0, yaw))
        mesh.rotate(Rot, center=mesh.get_center())

        cl_test0 = trimesh.base.Trimesh(vertices=np.array(mesh.vertices)[:4], faces=np.array([[CORNER_LU, CORNER_LD, CORNER_RD], [CORNER_LU, CORNER_RD, CORNER_RU]]))

        collision_manager.add_object(str(generated), cl_test0)
        if collision_manager.in_collision_internal():
            collision_manager.remove_object(str(generated))
            if print_progress:
                print(log_prefix, "pits collision")
            continue
            
    #     ################ Generate Collision Testing Scene ################
    #     scene = o3d.t.geometry.RaycastingScene()
    #     road_tri = o3d.t.geometry.TriangleMesh.from_legacy(road_seg)
    #     road_id = scene.add_triangles(road_tri)

        ################ Collision Detection Level 0, corners ################
        coarse_quires = []
        for i in [CORNER_LU, CORNER_RU, CORNER_LD, CORNER_RD]:
            v = mesh.vertices[i]
            coarse_quires.append([v[0], v[1], v[2] + 1000, 0, 0, -1])
        rays = o3d.core.Tensor(coarse_quires, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)

        collide = list(set(ans['primitive_ids'].numpy()))
        if scene.INVALID_ID in collide:
            if print_progress:
                print(log_prefix, "Corner out of map!")
            continue
            
        ############### Collision Detection Level 1, all vertices ##############
        quires = []
        for v in np.array(mesh.vertices):
            quires.append([v[0], v[1], v[2] + 1000, 0, 0, -1])
        rays = o3d.core.Tensor(quires, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)

        collide = set(ans['primitive_ids'].numpy())
        if scene.INVALID_ID in collide:
            if print_progress:
                print(log_prefix, "Vertex out of map!")
            continue
        
        faces_to_delete = faces_to_delete.union(collide)
        centers.append(mesh.get_center()[:2])
        
        pits_to_embed.append({"mesh":mesh, "center":mesh.get_center(), "border_vertex_id":border_vertex_id, "border_edges":border_edges})
        generated += 1
        
    # pits_to_embed

    borders_to_cut_hole = []
    for c in faces_to_delete:
        for i,j in [[0,1], [1,2], [2,0]]:
            s = min(road_seg.triangles[c][i], road_seg.triangles[c][j])
            e = max(road_seg.triangles[c][i], road_seg.triangles[c][j])
            if (s, e) in borders_to_cut_hole:
                borders_to_cut_hole.remove((s,e))
            else:
                borders_to_cut_hole.append((s,e))

    border_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(road_seg.vertices),
        lines=o3d.utility.Vector2iVector(np.array(borders_to_cut_hole)),
    )

    new_road = o3d.geometry.TriangleMesh(road_seg)
    new_road.triangles = o3d.utility.Vector3iVector(np.delete(np.asarray(new_road.triangles), list(faces_to_delete), axis=0))


    #### triangles solution: Delaunary triangulation
    vertices = np.array(new_road.vertices)
    segments = borders_to_cut_hole

    road_triangles = np.array(new_road.triangles)
    for c in range(road_triangles.shape[0]):
        for i,j in [[0,1], [1,2], [2,0]]:
            s = min(road_triangles[c][i], road_triangles[c][j])
            e = max(road_triangles[c][i], road_triangles[c][j])
            segments.append((s,e))

    for p in pits_to_embed:
        offset = vertices.shape[0]
        mesh_triangles = np.array(mesh.triangles)
        
        border_set = set(p["border_vertex_id"])
        vertices = np.vstack([vertices, p["mesh"].vertices[:len(border_set)]])
        
        for s, e in p['border_edges']:
            if s in border_set and e in border_set:
                segments.append((offset + s, offset + e))
                
    #     segments.append((offset + 0, offset + 1))
        
    #     segments.append((offset + 0, offset + 1))
    #     segments.append((offset + 0, offset + 1))
    #     segments.append((offset + 1, offset + 2))
    #     segments.append((offset + 2, offset + 3))
    #     segments.append((offset + 0, offset + 3))
        
    segments = list(set(segments))
    if pits_in_seg > 0:
        A = dict(vertices=vertices[:,:2], segments=segments, holes=centers) ## [0,0] should work
    else:
        A = dict(vertices=vertices[:,:2], segments=segments) ## [0,0] should work

    # B = tr.triangulate(A, 'pFC')
    B = tr.triangulate(A, 'qpa0.001')

    # tr.compare(plt, A, B)


    combined_vertices = np.hstack([B["vertices"] ,np.zeros([B["vertices"].shape[0], 1])])
    combined_triangles = B["triangles"]

    # 生成Perlin噪声
    scale = 1  # 尺度参数
    height_range = 0.1

    octaves = 6  # Octave数
    persistence = 0.5  # Persistence值
    lacunarity = 2.0  # Lacunarity值

    # 补坑
    for p in pits_to_embed:
        offset = combined_vertices.shape[0]
        combined_vertices = np.vstack([combined_vertices, p["mesh"].vertices])
        combined_triangles = np.vstack([combined_triangles, np.array(p["mesh"].triangles) + offset])

    for r in range(combined_vertices.shape[0]):
        point = combined_vertices[r]
        i = point[0]
        j = point[1]
        combined_vertices[r, 2] += noise.snoise2(i/scale,
                                                j/scale,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=102400,
                                                repeaty=102400,
                                                base=Perlin_seed) / 2 * height_range


    splited_road_seg = o3d.geometry.TriangleMesh()
    splited_road_seg.vertices = o3d.utility.Vector3dVector(combined_vertices[:, [1,2,0]])
    splited_road_seg.triangles = o3d.utility.Vector3iVector(combined_triangles)
    # plot_obj(splited_road_seg)

    o3d.io.write_triangle_mesh(os.path.join(OUTPUT_PATH, _name), splited_road_seg, print_progress=print_progress)




if __name__ == '__main__':
    MultiProcessing = True

    if os.path.isdir(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)
    ROADSEG_LIST = [file for file in os.listdir(ROADSEG_FOLDER) if file.endswith("obj")]

    ######### seed list
    seed = np.random.randint(0, 100)  # 种子值
    seed_list = [seed for i in range(len(ROADSEG_LIST))]

    ######### pothole number for each segment list
    pit_list = [0 for i in range(len(ROADSEG_LIST))]
    pit_remained = PITS_TO_EMBED
    while pit_remained > 0:
        i = np.random.randint(len(ROADSEG_LIST))
        if pit_list[i] <= MAX_PIT_ONE_SEG:
            pit_list[i] += 1
            pit_remained -= 1
        
    ######## whole argument list
    arg_list = [(ROADSEG_LIST[i], seed_list[i], pit_list[i]) for i in range(len(ROADSEG_LIST))]

    if MultiProcessing:
        with Pool(processes=CPU_CORES) as pool:
            with tqdm(total=len(ROADSEG_LIST)) as pbar:
                for _ in pool.imap_unordered(task_func, arg_list):
                    pbar.update(1)
        
    else:
        for f in arg_list:
            task_func(f)

