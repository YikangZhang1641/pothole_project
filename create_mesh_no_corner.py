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
from multiprocessing import Pool
import shutil
from tqdm import tqdm

CORNER_LU = 0
CORNER_LD = 1
CORNER_RD = 2
CORNER_RU = 3


OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "PitMeshNoCorner")


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

def plot_obj(road, mesh=None, opt=None):
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
        mesh_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(mesh.vertices),
            lines=o3d.utility.Vector2iVector(mesh_edges),
        )
        
        to_show = to_show + [mesh, mesh_lines]
    if opt:
        to_show = to_show + opt
        
    o3d.visualization.draw_geometries(to_show)

class PointSet:
    def __init__(self, data_points, L, R, U, D):
        self.vertices = []
        self.indices = {}
        self.faces = []

        base = data_points[L*GRID_PIXEL][U*GRID_PIXEL]
        self.base = np.array(base)

        self.data_points = np.array(data_points)
        self.L, self.R, self.U, self.D = L, R, U, D

        self.cnt = 0
        self.mesh = None

    def pushback_point(self, i, j, border=False):
        if i in self.indices and j in self.indices[i]:
            print("point already pushed back", i, j)
            return 

        p = self.data_points[i*GRID_PIXEL][j*GRID_PIXEL]
         
        ###### have to filter unknown bad points###########
        if np.any(np.isinf(p)):
            if i > 1:
                p = self.data_points[(i-1)*GRID_PIXEL][j*GRID_PIXEL] * 2 - self.data_points[(i-2)*GRID_PIXEL][j*GRID_PIXEL]
        if np.any(np.isinf(p)):
            if j > 1:
                p = self.data_points[i*GRID_PIXEL][(j-1)*GRID_PIXEL] * 2 - self.data_points[i*GRID_PIXEL][(j-2)*GRID_PIXEL]
        if np.any(np.isinf(p)):
            if i > 1 and j > 1:
                p = self.data_points[(i-1)*GRID_PIXEL][(j+1)*GRID_PIXEL] * 2 - self.data_points[(i-2)*GRID_PIXEL][(j+2)*GRID_PIXEL]
        if np.any(np.isinf(p)):
            if i > 1 and j < self.data_points.shape[1] // GRID_PIXEL - 1:
                p = self.data_points[(i-1)*GRID_PIXEL][(j-1)*GRID_PIXEL] * 2 - self.data_points[(i-2)*GRID_PIXEL][(j-2)*GRID_PIXEL]
    #    if np.any(np.isinf(p)):
    ##        if i > 3 and j < data_points.shape[1] // GRID_PIXEL - 3:
    #        p = data_points[(i-2)*GRID_PIXEL][(j-2)*GRID_PIXEL] * 2 - data_points[(i-4)*GRID_PIXEL][(j-4)*GRID_PIXEL]
        if np.any(np.isinf(p)):
            print("bad point inf detected at: ", i*GRID_PIXEL, j*GRID_PIXEL)
            return
        
        ###################################################       
        self.vertices.append([p[0]-self.base[0], p[1]-self.base[1], -p[2]+self.base[2]])

        if i not in self.indices:
            self.indices[i] = {}
        self.indices[i][j] = self.cnt
        self.cnt += 1

    def generate_faces(self):
        for i in range(self.L, self.R):
            for j in range(self.U, self.D):
                if j not in self.indices[i]:
                    continue

                ax, ay = i+1, j+1
                bx, by = i+1, j
                cx, cy = i, j+1

                if ay in self.indices[ax] and by in self.indices[bx]:
                    self.faces.append( [ self.indices[i][j], self.indices[ax][ay], self.indices[bx][by] ] )
                if ay in self.indices[ax] and cy in self.indices[cx]:
                    self.faces.append( [ self.indices[i][j], self.indices[cx][cy], self.indices[ax][ay] ] )

        # faces.append([indices[L][U], indices[L][D], indices[L+1][D-1]])
        # faces.append([indices[L][U], indices[L+1][D-1], indices[L+1][U+1]])
        # faces.append([indices[L][U], indices[L+1][U+1], indices[R-1][U+1]])
        # faces.append([indices[L][U], indices[R-1][U+1], indices[R][U]])

        # faces.append([indices[R][D], indices[R][U], indices[R-1][U+1]])
        # faces.append([indices[R][D], indices[R-1][U+1], indices[R-1][D-1]])
        # faces.append([indices[R][D], indices[R-1][D-1], indices[L+1][D-1]])
        # faces.append([indices[R][D], indices[L+1][D-1], indices[L][D]])

    def generate_mesh(self):
        mesh = o3d.geometry.TriangleMesh()

        vertices = np.array(self.vertices) / 100
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        faces = np.array(self.faces)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0,0,0])
        mesh_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))

        corner0 = mesh.vertices[self.indices[self.R][self.U]] - mesh.vertices[self.indices[self.L][self.U]]
        roll = np.arctan2(corner0[2], corner0[1])

        corner1 = mesh.vertices[self.indices[self.L][self.D]] - mesh.vertices[self.indices[self.L][self.U]]
        pitch = np.arctan2(corner1[2], corner1[0])

        Rot = mesh.get_rotation_matrix_from_xyz((-roll, pitch, 0))
        mesh.rotate(Rot, center=(0, 0, 0))

        self.mesh = mesh

    def smooth1_flat(self):
        ### z -> 0
        for i in range(self.L + 1, self.R):
            a = self.mesh.vertices[self.indices[ i ][ self.U + 1 ]][2]
            b = self.mesh.vertices[self.indices[ i ][ self.D - 1 ]][2]
            for j in range(self.U, self.D + 1):
                self.mesh.vertices[self.indices[ i ][ j ]][2] -= (a + (b-a) / (self.D - self.U - 1) * (j - self.U - 1))
                
        for j in range(self.U + 1, self.D):
            c = self.mesh.vertices[self.indices[ self.L + 1 ][ j ]][2]
            d = self.mesh.vertices[self.indices[ self.R - 1 ][ j ]][2]    
            for i in range(self.L, self.R+1):
                self.mesh.vertices[self.indices[ i ][ j ]][2] -= (c + (d-c) / (self.R - self.L - 1) * (i - self.L - 1))

        # mesh.vertices[CORNER_LU][2] = 0
        # mesh.vertices[CORNER_LD][2] = 0
        # mesh.vertices[CORNER_RU][2] = 0
        # mesh.vertices[CORNER_RD][2] = 0

    def smooth2_align(self):
        # ### xy -> line    
        for i in range(self.L, self.R+1): #U
            start = self.mesh.vertices[self.indices[ self.L+1 ][ self.U+1 ]][:2]
            end   = self.mesh.vertices[self.indices[ self.R-1 ][ self.U+1 ]][:2]       
            n = (end - start) / np.linalg.norm(end-start)
            self.mesh.vertices[self.indices[ i ][ self.U+1 ]][:2] = start + np.dot((self.mesh.vertices[self.indices[ i ][ self.U+1 ]][:2] - start), n) * n

            
        for i in range(self.L, self.R+1): #D
            start = self.mesh.vertices[self.indices[ self.L+1 ][ self.D-1 ]][:2]
            end   = self.mesh.vertices[self.indices[ self.R-1 ][ self.D-1 ]][:2]
            n = (end - start) / np.linalg.norm(end-start)
            self.mesh.vertices[self.indices[ i ][ self.D-1 ]][:2] = start + np.dot((self.mesh.vertices[self.indices[ i ][ self.D-1 ]][:2] - start), n) * n
                
        for j in range(self.U, self.D+1): #L
            start = self.mesh.vertices[self.indices[ self.L+1 ][ self.U+1 ]][:2]
            end   = self.mesh.vertices[self.indices[ self.L+1 ][ self.D-1 ]][:2]
            n = (end - start) / np.linalg.norm(end-start)
            self.mesh.vertices[self.indices[ self.L+1 ][ j ]][:2] = start + np.dot((self.mesh.vertices[self.indices[ self.L+1 ][ j ]][:2] - start), n) * n

        for j in range(self.U, self.D+1): #R
            start = self.mesh.vertices[self.indices[ self.R-1 ][ self.U+1 ]][:2]
            end   = self.mesh.vertices[self.indices[ self.R-1 ][ self.D-1 ]][:2]
            n = (end - start) / np.linalg.norm(end-start)
            self.mesh.vertices[self.indices[ self.R-1 ][ j ]][:2] = start + np.dot((self.mesh.vertices[self.indices[ self.R-1 ][ j ]][:2] - start), n) * n
            

def func(file):
    # print(file)
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
            # print("L", L, "R", R, "U", U, "D", D)

            name = stereo_dataset_path[-1] + '_' + file.split('.')[0]+"_"+str(label_cnt)
        
            point_set = PointSet(data_points, L, R, U, D)

            #### construct BORDERS, instead of CORNERS.
            point_set.pushback_point(L, U)
            point_set.pushback_point(L, D)
            point_set.pushback_point(R, D)
            point_set.pushback_point(R, U)

            for i in range(L+1, R):
                point_set.pushback_point(i, U)
            for j in range(U+1, D):
                point_set.pushback_point(R, j)
            for i in range(R-1, L, -1):
                point_set.pushback_point(i, D)
            for j in range(D-1, U, -1):
                point_set.pushback_point(L, j)

            #### construct vertices inside
            for i in range(L+1, R):
                for j in range(U+1, D):
                    point_set.pushback_point(i, j)

            #### construct faces
            point_set.generate_faces()
            point_set.generate_mesh()
            point_set.smooth1_flat()
            # point_set.smooth2_align()

            #### mark borders: drift to 10.0 height
            for i in range(L, R+1):
                point_set.mesh.vertices[point_set.indices[i][U]][2] = -10
                point_set.mesh.vertices[point_set.indices[i][D]][2] = -10

            for j in range(U, D+1):
                point_set.mesh.vertices[point_set.indices[L][j]][2] = -10    
                point_set.mesh.vertices[point_set.indices[R][j]][2] = -10

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

            # plot_obj(point_set.mesh, None, [coord])
            o3d.io.write_triangle_mesh(os.path.join(OUTPUT_PATH, name+".ply"), point_set.mesh)



if __name__ == '__main__':
    if os.path.isdir(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    MultiProcessing = True

    data = os.listdir(os.path.join(stereo_dataset_path, "ptcloud"))
    if MultiProcessing:
        CPU_CORES = 23
        with Pool(processes=CPU_CORES) as pool:
            with tqdm(total=len(data)) as pbar:
                for _ in pool.map(func, data):
                    pbar.update()
    else:
        for f in data:
            func(f)

