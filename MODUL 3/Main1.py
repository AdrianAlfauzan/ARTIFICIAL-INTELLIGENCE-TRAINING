import matplotlib.pyplot as plt
import numpy as np
import random

import os

os.system("cls")

# Membuat matriks jarak antar kota secara acak
def create_distance_matrix(n):
    np.random.seed(1)
    dist_matrix = np.random.randint(10,100,size=(n,n))
    np.fill_diagonal(dist_matrix,0)
    return dist_matrix

# Fungsi untuk menampilkan rute yang telah dihasilkan
def plot_route(route,cities):
    x = cities[:,0]
    y = cities[:,1]
    plt.plot(x[route],y[route],r'r-')
    plt.plot(x,y,'bo')
    plt.show()

# Fungsi untuk melakukan pencarian rute terpendek menggunakan DFS
def tsp_dfs(n,dist_matrix,start=0):
    visited = [False] * n
    visited[start] = True
    stack = [(start,[start],0)]
    shortest_path = None
    while stack:
        current,path,cost = stack.pop()
        if False not in visited:
            path.append(start)
            if shortest_path is None or cost < shortest_path[1]:
                shortest_path = (path,cost)
        for neighbor in range(n):
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append((neighbor,path + [neighbor], cost + dist_matrix[current][neighbor]))
        visited = [False if i not in path else True for i in range(n)]
    
    return shortest_path

# Membuat matriks jarak antar kota sebanyak 5 kota secara acak
n = 5
distance_matrix = create_distance_matrix(n)

# Menampilkan matriks jarak antar kota 
print("Matriks Jarak AntarKota : ")
print(distance_matrix)

# Menyelasaikan TSP dengan DFS 
shortest_path , cost = tsp_dfs(n,distance_matrix,start=0)

# Menampilkan rute terpendek yang dihasilkan
print("Rute terpendek : ", shortest_path)
print("Total jarak : ", cost)

# Menampilkan rute terpendek dalam bentuk grafik
cities = np.random.randint(10,100,size=(n,2))
plot_route(shortest_path,cities)

