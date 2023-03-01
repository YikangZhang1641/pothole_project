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
from multiprocessing import Process, Lock, Pool
import shutil
import trimesh
from tqdm import tqdm
import noise


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

def tri_area(v1, v2, v3):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    a = np.linalg.norm(v2 - v1)
    b = np.linalg.norm(v3 - v2)
    c = np.linalg.norm(v1 - v3)
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

###################################################################

pits_to_embed = 100
threshold_split = 0.1


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


    ################ Generate Collision Testing Scene ################
    scene = o3d.t.geometry.RaycastingScene()
    road_tri = o3d.t.geometry.TriangleMesh.from_legacy(new_road)
    road_id = scene.add_triangles(road_tri)

    ################ Collision Detection Level 0, corners ################
    coarse_quires = []
    for i in [CORNER_LU, CORNER_RU, CORNER_LD, CORNER_RD]:
        v = mesh.vertices[i]
        coarse_quires.append([v[0], v[1], v[2] + 1000, 0, 0, -1])
    rays = o3d.core.Tensor(coarse_quires, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)

    collide = list(set(ans['primitive_ids'].numpy()))
    if scene.INVALID_ID in collide:
        print(log_prefix, "Corner out of map!")
        continue

    ############### Collision Detection Level 1, all vertices ##############
    quires = []
    for v in np.array(mesh.vertices):
        quires.append([v[0], v[1], v[2] + 1000, 0, 0, -1])
    rays = o3d.core.Tensor(quires, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)

    collide = list(set(ans['primitive_ids'].numpy()))
    if scene.INVALID_ID in collide:
        print(log_prefix, "Vertex out of map!")
        continue

    ######################################################

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


verts = np.array(road.vertices)
tris = np.array(road.triangles)
print("verts ", verts.shape[0], " tris ", tris.shape[0])

tri_coord = verts[tris]
small_pieces = []
large_pieces = []
for i in tqdm(range(tri_coord.shape[0])):
    area = tri_area(tri_coord[i][0], tri_coord[i][1], tri_coord[i][2])
    if area < threshold_split:
        small_pieces.append(tris[i])
    else:
        large_pieces.append(tris[i])

def pit_remapping(origin_verts, small_pieces):
    mapping = dict()
    new_t = []
    new_v = []
    for p in small_pieces:
        for i in range(3):
            if p[i] not in mapping:
                mapping[p[i]] = len(new_v)
                new_v.append(origin_verts[p[i]])

        tmp = []
        for i in range(3):
            tmp.append(mapping[p[i]])
        new_t.append(tmp)
    return np.array(new_v), np.array(new_t)

_v, _t = pit_remapping(verts, small_pieces)
p_vertices = [_v]
p_triangles = [_t]

pbar = tqdm(total=len(large_pieces))
lock = Lock()
def f(q_pts):
    global lock, p_vertices, triangles, pbar
    
    q_seg = np.array([[0,1],[1,2],[2,0]])
    A = dict(vertices=q_pts, segments=q_seg)
    s = 'qpa'+str(threshold_split)
    B = tr.triangulate(A, opts=s)
    # B = tr.triangulate(A, 'qpa0.1')
    
    lock.acquire()
    try:
        offset = 0
        if len(p_vertices) != 0:
            offset = np.vstack(p_vertices).shape[0]

        new_verts = np.concatenate([np.array(B['vertices']), np.zeros([B['vertices'].shape[0], 1])], axis=1)
        p_vertices.append(new_verts)

        new_faces = offset + B['triangles']
        p_triangles.append(new_faces)
        
    finally:
        # print("verts add", new_verts.shape[0], "/", np.array(p_vertices).shape[0])
        pbar.update(1)
        lock.release()

data = []
for _i in range(len(large_pieces)):
    tri_to_split_id = large_pieces[_i]
    q_pts = verts[tri_to_split_id][:,:2]
    f(q_pts)
#     data.append(q_pts)

# pool = Pool(20)
# res = pool.map(f, data)
# pool.close()
# pool.join()



p_vertices = np.vstack(p_vertices)
p_triangles = np.vstack(p_triangles)

################ Height Noise modelization ####################
def height_modulization(p_vertices):
    x_max = p_vertices[:,0].max()
    x_min = p_vertices[:,0].min()
    y_max = p_vertices[:,1].max()
    y_min = p_vertices[:,1].min()
    print(x_max, x_min, y_max, y_min)

    GRID_SIZE = 0.3
    HEIGHT_RANGE = 0.015

    width = int((x_max - x_min) // GRID_SIZE) + 2
    height = int((y_max - y_min) // GRID_SIZE) + 2
    # scale = 50.0

    # Generate a 2D grid of Perlin noise values
    perlin_grid = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            perlin_grid[x][y] = noise.snoise2(x, y)
    perlin_grid = perlin_grid * HEIGHT_RANGE

    for i in range(p_vertices.shape[0]):
        x = p_vertices[i][0]
        y = p_vertices[i][1]
        # z = p_vertices[i][2]

        l = int((x-x_min) // GRID_SIZE)
        r = int((x-x_min) // GRID_SIZE) + 1
        u = int((y-y_min) // GRID_SIZE)
        d = int((y-y_min) // GRID_SIZE) + 1

        dhx = (perlin_grid[r][u] - perlin_grid[l][u]) * ((x-x_min) / GRID_SIZE - l) 
        dhy = (perlin_grid[l][d] - perlin_grid[l][u]) * ((y-y_min) / GRID_SIZE - u) 
        
        p_vertices[i][2] = p_vertices[i][2] + perlin_grid[l][d] + dhx + dhy
    return p_vertices

p_vertices = height_modulization(p_vertices)   
road_save = o3d.geometry.TriangleMesh()
road_save.vertices = o3d.utility.Vector3dVector(p_vertices)
road_save.triangles = o3d.utility.Vector3iVector(p_triangles)


print("TO WRITE")
print("verts ", np.array(road_save.vertices).shape[0], " tris ", np.array(road_save.triangles).shape[0])

# o3d.visualization.draw_geometries([road, new_lines])

# road.vertices = o3d.utility.Vector3dVector(np.array(road.vertices) * 100)
o3d.io.write_triangle_mesh(os.path.join(FOLDER_PATH, "pothole_road.obj"), road_save, print_progress=True)



