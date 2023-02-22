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

CORNER_LU = 0
CORNER_LD = 1
CORNER_RD = 2
CORNER_RU = 3


FOLDER_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "PitMesh")
if os.path.isdir(FOLDER_PATH):
    shutil.rmtree(FOLDER_PATH)
os.makedirs(FOLDER_PATH)

GAP = 30
GRID_PIXEL = 10
stereo_dataset_path = "/media/mias/UGREEN0/stereo_pothole_datasets/dataset2"

def bfs(img, visited, pos, new_label, label_cnt):
    x, y = pos
    if visited[x][y]:
        return False
    
    visited[x][y] = True
    if img[x][y] == 0:
        return False
    
    dq = deque([])
    dq.appendleft((x,y))
    
    L,R,U,D = visited.shape[0], 0, visited.shape[1], 0
    while len(dq) > 0:
        x, y = dq.pop()
        new_label[x][y] = label_cnt * 50
        L = min(L, x)
        R = max(R, x)
        U = min(U, y)
        D = max(D, y)
        
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x+dx, y+dy
            if nx < 0 or nx >= visited.shape[0] or ny < 0 or ny >= visited.shape[1]:
                continue
            if visited[nx][ny]:
                continue
            visited[nx][ny] = True
            
            if img[nx][ny] != 0:
                dq.appendleft((nx, ny))
            
    return (L,R,U,D)

def pushback_point(vertices, indices, base, cnt, data_points, i, j, border=False):
    p = data_points[i*GRID_PIXEL][j*GRID_PIXEL]
    
    ###### have to filter unknown bad points###########
    if np.any(np.isinf(p)):
        if i > 1:
            p = data_points[(i-1)*GRID_PIXEL][j*GRID_PIXEL] * 2 - data_points[(i-2)*GRID_PIXEL][j*GRID_PIXEL]
    if np.any(np.isinf(p)):
        if j > 1:
            p = data_points[i*GRID_PIXEL][(j-1)*GRID_PIXEL] * 2 - data_points[i*GRID_PIXEL][(j-2)*GRID_PIXEL]
    if np.any(np.isinf(p)):
        if i > 1 and j > 1:
            p = data_points[(i-1)*GRID_PIXEL][(j+1)*GRID_PIXEL] * 2 - data_points[(i-2)*GRID_PIXEL][(j+2)*GRID_PIXEL]
    if np.any(np.isinf(p)):
        if i > 1 and j < data_points.shape[1] // GRID_PIXEL - 1:
            p = data_points[(i-1)*GRID_PIXEL][(j-1)*GRID_PIXEL] * 2 - data_points[(i-2)*GRID_PIXEL][(j-2)*GRID_PIXEL]
#    if np.any(np.isinf(p)):
##        if i > 3 and j < data_points.shape[1] // GRID_PIXEL - 3:
#        p = data_points[(i-2)*GRID_PIXEL][(j-2)*GRID_PIXEL] * 2 - data_points[(i-4)*GRID_PIXEL][(j-4)*GRID_PIXEL]
    if np.any(np.isinf(p)):
        print("bad point inf detected at: ", i*GRID_PIXEL, j*GRID_PIXEL)
        return
    ###################################################       
    vertices.append([p[0]-base[0], p[1]-base[1], -p[2]+base[2]])

    if i not in indices:
        indices[i] = {}
    indices[i][j] = cnt


def func(file):
    print(file)
    filepath = os.path.join(stereo_dataset_path, "ptcloud", file) 
    labelpath = os.path.join(stereo_dataset_path, "label", file.split('.')[0]+".png")

    data_points = scio.loadmat(filepath)['xyzPoints']

    img = cv2.imread(labelpath)
    label = img[:,:,2]

    visited = np.zeros(label.shape)
    new_label = np.zeros(label.shape)
    label_cnt = 0
    for pi in range(label.shape[0]):
        for pj in range(label.shape[1]):
            result = bfs(label, visited, (pi,pj), new_label, label_cnt)
            if not result:
                continue
        
            L = max(result[0] - GAP, 0)//GRID_PIXEL
            R = min(result[1] + GAP, label.shape[0]-1)//GRID_PIXEL
            U = max(result[2] - GAP, 0)//GRID_PIXEL
            D = min(result[3] + GAP, label.shape[1]-1)//GRID_PIXEL        
                        
            name = stereo_dataset_path[-1] + '_' + file.split('.')[0]+"_"+str(label_cnt)
        
            base = data_points[(L+1)*GRID_PIXEL][(U+1)*GRID_PIXEL]

            ### vertices
            vertices = []
            indices = {}

            #### construct corners.
            pushback_point(vertices, indices, base, CORNER_LU, data_points, L, U)
            pushback_point(vertices, indices, base, CORNER_LD, data_points, L, D)
            pushback_point(vertices, indices, base, CORNER_RD, data_points, R, D)
            pushback_point(vertices, indices, base, CORNER_RU, data_points, R, U)

            #### construct vertices
            cnt = len(vertices)
            for i in range(L+1, R):
                for j in range(U+1, D):
                    pushback_point(vertices, indices, base, cnt, data_points, i, j)
                    cnt += 1

            #### construct faces
            faces = []
            for i in range(L+1, R-1):
                for j in range(U+1, D-1):
                    if j not in indices[i]:
                        continue

                    ax, ay = i+1, j+1
                    bx, by = i+1, j
                    cx, cy = i, j+1

                    if ay in indices[ax] and by in indices[bx]:
                        faces.append( [ indices[i][j], indices[ax][ay], indices[bx][by] ] )
                    if ay in indices[ax] and cy in indices[cx]:
                        faces.append( [ indices[i][j], indices[cx][cy], indices[ax][ay] ] )

            faces.append([indices[L][U], indices[L][D], indices[L+1][D-1]])
            faces.append([indices[L][U], indices[L+1][D-1], indices[L+1][U+1]])
            faces.append([indices[L][U], indices[L+1][U+1], indices[R-1][U+1]])
            faces.append([indices[L][U], indices[R-1][U+1], indices[R][U]])

            faces.append([indices[R][D], indices[R][U], indices[R-1][U+1]])
            faces.append([indices[R][D], indices[R-1][U+1], indices[R-1][D-1]])
            faces.append([indices[R][D], indices[R-1][D-1], indices[L+1][D-1]])
            faces.append([indices[R][D], indices[L+1][D-1], indices[L][D]])

            vertices = np.array(vertices) / 100
            faces = np.array(faces)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

            # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0,0,0])
            mesh_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))

            corner0 = mesh.vertices[indices[R][U]] - mesh.vertices[indices[L][U]]
            roll = np.arctan2(corner0[2], corner0[1])

            corner1 = mesh.vertices[indices[L][D]] - mesh.vertices[indices[L][U]]
            pitch = np.arctan2(corner1[2], corner1[0])

            Rot = mesh.get_rotation_matrix_from_xyz((-roll, pitch, 0))
            mesh.rotate(Rot, center=(0, 0, 0))


            ### z -> 0
            for i in range(L+1, R):
                a = mesh.vertices[indices[ i ][ U+1 ]][2]
                b = mesh.vertices[indices[ i ][ D-1 ]][2]
                for j in range(U+1, D):
                    mesh.vertices[indices[ i ][ j ]][2] -= (a + (b-a) / (D-U-1) * (j-U-1))
                    
            for j in range(U+1, D):
                c = mesh.vertices[indices[ L+1 ][ j ]][2]
                d = mesh.vertices[indices[ R-1 ][ j ]][2]    
                for i in range(L+1, R):
                    mesh.vertices[indices[ i ][ j ]][2] -= (c + (d-c) / (R-L-1) * (i-L-1))

            mesh.vertices[CORNER_LU][2] = 0
            mesh.vertices[CORNER_LD][2] = 0
            mesh.vertices[CORNER_RU][2] = 0
            mesh.vertices[CORNER_RD][2] = 0

            # ### xy -> line    
            for i in range(L+1, R): #U
                start = mesh.vertices[indices[ L+1 ][ U+1 ]][:2]
                end   = mesh.vertices[indices[ R-1 ][ U+1 ]][:2]       
                n = (end - start) / np.linalg.norm(end-start)
                mesh.vertices[indices[ i ][ U+1 ]][:2] = start + np.dot((mesh.vertices[indices[ i ][ U+1 ]][:2] - start), n) * n

                
            for i in range(L+1, R): #D
                start = mesh.vertices[indices[ L+1 ][ D-1 ]][:2]
                end   = mesh.vertices[indices[ R-1 ][ D-1 ]][:2]
                n = (end - start) / np.linalg.norm(end-start)
                mesh.vertices[indices[ i ][ D-1 ]][:2] = start + np.dot((mesh.vertices[indices[ i ][ D-1 ]][:2] - start), n) * n
                    
            for j in range(U+1, D): #L
                start = mesh.vertices[indices[ L+1 ][ U+1 ]][:2]
                end   = mesh.vertices[indices[ L+1 ][ D-1 ]][:2]
                n = (end - start) / np.linalg.norm(end-start)
                mesh.vertices[indices[ L+1 ][ j ]][:2] = start + np.dot((mesh.vertices[indices[ L+1 ][ j ]][:2] - start), n) * n

            for j in range(U+1, D): #R
                start = mesh.vertices[indices[ R-1 ][ U+1 ]][:2]
                end   = mesh.vertices[indices[ R-1 ][ D-1 ]][:2]
                n = (end - start) / np.linalg.norm(end-start)
                mesh.vertices[indices[ R-1 ][ j ]][:2] = start + np.dot((mesh.vertices[indices[ R-1 ][ j ]][:2] - start), n) * n
                

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

            triangles = np.asarray(mesh.triangles)
            mesh_edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))

            mesh_lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(mesh.vertices),
                lines=o3d.utility.Vector2iVector(mesh_edges),
            )
            # o3d.visualization.draw_geometries([mesh, coord, mesh_lines])
            o3d.io.write_triangle_mesh(os.path.join(FOLDER_PATH, name+".ply"), mesh)



if __name__ == '__main__':
    MultiProcessing = True

    if MultiProcessing:
        pool = multiprocessing.Pool()
        pool.map(func, os.listdir(os.path.join(stereo_dataset_path, "ptcloud")))
    else:
        for f in os.listdir(os.path.join(stereo_dataset_path, "ptcloud")):
            func(f)