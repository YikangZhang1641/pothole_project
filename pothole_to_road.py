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
import shutil
import trimesh
from tqdm import tqdm


CORNER_LU = 0
CORNER_LD = 1
CORNER_RD = 2
CORNER_RU = 3


CUR_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
FOLDER_PATH = os.path.join(CUR_PATH, "RoadPitMesh")
if os.path.isdir(FOLDER_PATH):
    shutil.rmtree(FOLDER_PATH)
os.makedirs(FOLDER_PATH)
PIT_LIST = os.listdir(os.path.join(CUR_PATH, "PitMesh")) 




pits_to_embed = 200

# road = o3d.io.read_triangle_mesh("/home/mias/Downloads/part_of_road.ply")
road = o3d.io.read_triangle_mesh("/home/mias/Datasets/CarlaRoads/Town01_Road_Road.obj")
road.remove_duplicated_vertices()

road.vertices = o3d.utility.Vector3dVector(np.array(road.vertices)[:, [2,0,1]] / 100)

# triangles = np.asarray(road.triangles)
# road_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))

# road_lines = o3d.geometry.LineSecdt(
#     points=o3d.utility.Vector3dVector(road.vertices),
#     lines=o3d.utility.Vector2iVector(road_edges),
# )

# o3d.visualization.draw_geometries([mesh, coord, mesh_lines, road, road_lines])


Xs = np.array(road.vertices)[:,0]
Ys = np.array(road.vertices)[:,1]
RoadBoundary = [Xs.min(), Xs.max(), Ys.min(), Ys.max()]

collision_manager = trimesh.collision.CollisionManager()
new_road = o3d.geometry.TriangleMesh()
mesh = o3d.geometry.TriangleMesh()

# pbar = tqdm(total=pits_to_embed)
generated = 0
while generated < pits_to_embed:
    new_road.clear()
    mesh.clear()

    pit_name = PIT_LIST[np.random.randint(len(PIT_LIST))]

    new_road = o3d.geometry.TriangleMesh(road)


    x = np.random.uniform(RoadBoundary[0], RoadBoundary[1])
    y = np.random.uniform(RoadBoundary[2], RoadBoundary[3])
    yaw = np.random.uniform() * np.pi

    mesh = o3d.io.read_triangle_mesh(os.path.join(CUR_PATH, "PitMesh", pit_name))
    log_prefix = "( " + str(generated) + " / " + str(pits_to_embed) + ") embedding " + pit_name

    mesh.translate((x, y, 0))
    Rot = mesh.get_rotation_matrix_from_xyz((0, 0, yaw))
    mesh.rotate(Rot, center=mesh.get_center())

    test0 = trimesh.base.Trimesh(vertices=np.array(mesh.vertices)[:4], faces=np.array([[CORNER_LU, CORNER_LD, CORNER_RD], [CORNER_LU, CORNER_RD, CORNER_RU]]))

    collision_manager.add_object(str(generated), test0)
    if collision_manager.in_collision_internal():
        collision_manager.remove_object(str(generated))
        print(log_prefix, "pits collision")
        continue

    triangles = np.asarray(mesh.triangles)
    mesh_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))

    # mesh_lines = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(mesh.vertices),
    #     lines=o3d.utility.Vector2iVector(mesh_edges),
    # )

    quires = []
    for v in np.array(mesh.vertices):
        quires.append([v[0], v[1], v[2] + 1000, 0, 0, -1])
    rays = o3d.core.Tensor(quires, dtype=o3d.core.Dtype.Float32)

    scene = o3d.t.geometry.RaycastingScene()
    road_tri = o3d.t.geometry.TriangleMesh.from_legacy(new_road)
    road_id = scene.add_triangles(road_tri)

    ans = scene.cast_rays(rays)

    collide = list(set(ans['primitive_ids'].numpy()))
    if scene.INVALID_ID in collide:
        print(log_prefix, "outside of map!")
        continue

    border = []
    for c in collide:
        for i,j in [[0,1], [1,2], [2,0]]:
            s = min(new_road.triangles[c][i], new_road.triangles[c][j])
            e = max(new_road.triangles[c][i], new_road.triangles[c][j])
            if (s, e) in border:
                border.remove((s,e))
            else:
                border.append((s,e))

    # border_lines = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(new_road.vertices),
    #     lines=o3d.utility.Vector2iVector(np.array(border)),
    # )
    # print("triangles before", new_road.triangles )
    new_road.triangles = o3d.utility.Vector3iVector(np.delete(np.asarray(new_road.triangles), collide, axis=0))
    # print("triangle after", new_road.triangles )

    all_verts = np.vstack([new_road.vertices, mesh.vertices])

    offset = np.array(new_road.vertices).shape[0]
    tmp_id = np.array([ [CORNER_LU, CORNER_LD], 
                        [CORNER_LD, CORNER_RD],
                        [CORNER_RD, CORNER_RU], 
                        [CORNER_RU, CORNER_LU]
                    ])
    border_edge_id = np.vstack([border, tmp_id+offset])


    #### triangles solution: Delaunary triangulation
    origin_idx = list(set(border_edge_id.flatten().tolist()))
    query_idx = {origin_idx[i]:i for i in range(len(origin_idx))}

    q_pts = []
    q_seg = []

    for i in origin_idx:
        q_pts.append(all_verts[i][:2])
    q_pts = np.array(q_pts)

    for bd in border_edge_id:
        q_seg.append([query_idx[bd[0]], query_idx[bd[1]]])
    q_seg = np.array(q_seg)

    A = dict(vertices=q_pts, segments=q_seg, holes=[mesh.get_center()[:2]]) ## [0,0] should work
    B = tr.triangulate(A, 'pFC')
    tr.compare(plt, A, B)


    tmp_id = np.vstack((B['triangles'][:, [0, 1]], B['triangles'][:, [1, 2]], B['triangles'][:, [2, 0]]))
    # print("origin idx:", origin_idx)
    # print("A: \n", A)
    # print("B: \n", B)
    if A['vertices'].shape[0] != B['vertices'].shape[0]:
        print(log_prefix, "Vertices number mismatch")
        continue


    delaunay_edges = np.array(origin_idx)[tmp_id]

    delaunay_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector( all_verts ),
        lines=o3d.utility.Vector2iVector( delaunay_edges ),
    )


    # new_road = o3d.geometry.TriangleMesh()
    new_road.vertices = o3d.utility.Vector3dVector(all_verts)

    new_tris = np.vstack([new_road.triangles, np.array(mesh.triangles) + offset, np.array(origin_idx)[np.array(B['triangles'])]])
    new_road.triangles = o3d.utility.Vector3iVector(new_tris)

    triangles = np.asarray(new_road.triangles)
    new_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    new_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(new_road.vertices),
        lines=o3d.utility.Vector2iVector(new_edges),
    )
    # o3d.visualization.draw_geometries([road, new_lines])
    road.clear()
    road = o3d.geometry.TriangleMesh(new_road)
    generated += 1
    # pbar.update(1)
    print(log_prefix, "Successfully generated", pit_name, "at", x, y, yaw)

print("TO WRITE")





road.vertices = o3d.utility.Vector3dVector(np.array(road.vertices) * 100)
o3d.io.write_triangle_mesh(os.path.join(FOLDER_PATH, "pothole_road.obj"), road, print_progress=True)
# o3d.visualization.draw_geometries([road, new_lines])



