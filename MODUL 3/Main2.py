import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

os.system("cls")

# Fungsi untuk menghitung jarak antara dua kota
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Fungsi untuk menghitung total jarak tur 
def tour_length(tour,cities):
    length = 0
    for i in range(len(tour)):
        length += distance(cities[tour[i-1]], cities[tour[i]])
    return length

# Fungsi membuat solusi awal secara acak
def initial_solution(num_cities):
    return np.random.permutation(num_cities)

# Fungsi untuk membuat langkah perubahan solusi
def move(tour):
    new_tour = tour.copy()
    i, j = np.random.randint(len(tour),size=2)
    new_tour[i], new_tour[j] = new_tour[j],new_tour[i]
    return new_tour

# Fungsi hill climbing search

def hill_climbing_tsp(cities, max_iterations=1):
    num_cities = len(cities) 
    current_tour = initial_solution(num_cities)
    current_length = tour_length(current_tour, cities)
    iterations = 0
    lengths = [current_length]
    tours = [current_tour]

    while iterations < max_iterations: 
        new_tour = move(current_tour)
        new_length = tour_length(new_tour, cities)
        if new_length < current_length:
            current_tour = new_tour
            current_length = new_length
            lengths.append(current_length)
            tours.append(current_tour)
            iterations += 1

    return current_tour, current_length, lengths, tours

# Pengaturan random seed untuk hasil yang konsisten
np.random.seed(2)

# Membuat 10 kata secara acak
num_cities = 10
cities = np.random.rand(num_cities,2)
# print(cities)

# Menjalankan hill climbing search
best_tour,best_length, length,tours = hill_climbing_tsp(cities)
print("Best tour:", best_tour)
print("Best length:", best_length)

# Visualisasi hasil
fig, ax = plt.subplots()
def update(frame) :
    ax.clear()
    ax.set_title("Iteration {}".format(frame))
    ax.scatter(cities[:,0],cities[:,1])
    ax.plot(cities[tours[frame - 1], 0], cities[tours[frame - 1], 1])
    ax.set_xticks([])
    ax.set_yticks([])

ani = FuncAnimation(fig, update , frames=len(tours) + 1, interval=500, repeat=False)
plt.show()