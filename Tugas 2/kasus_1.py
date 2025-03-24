import sys
from queue import PriorityQueue
from tabulate import tabulate

# Representasi graf dengan jarak antar kota
graf = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 147, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Craiova': 147, 'Sibiu': 80, 'Pitesti': 97},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucarest': 101},
    'Fagaras': {'Sibiu': 99, 'Bucarest': 211},
    'Bucarest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucarest': 90},
    'Urziceni': {'Bucarest': 85, 'Vaslui': 142, 'Hirsova': 98},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86}
}

# Heuristic untuk Best First Search (jarak langsung ke Bucharest)
heuristic = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Sibiu': 253, 'Timisoara': 329,
    'Lugoj': 244, 'Mehadia': 241, 'Drobeta': 242, 'Craiova': 160, 'Rimnicu Vilcea': 193,
    'Pitesti': 100, 'Fagaras': 176, 'Bucarest': 0, 'Giurgiu': 77, 'Urziceni': 80,
    'Vaslui': 199, 'Iasi': 226, 'Neamt': 234, 'Hirsova': 151, 'Eforie': 161
}

# Algoritma Dijkstra
def dijkstra(graf, awal, tujuan):
    jarak = {node: sys.maxsize for node in graf}
    jarak[awal] = 0
    sebelum = {node: None for node in graf}
    queue = list(graf.keys())

    while queue:
        queue.sort(key=lambda node: jarak[node])
        node_terdekat = queue.pop(0)

        if node_terdekat == tujuan:
            break

        for tetangga in graf[node_terdekat]:
            jarak_baru = jarak[node_terdekat] + graf[node_terdekat][tetangga]
            if jarak_baru < jarak[tetangga]:
                jarak[tetangga] = jarak_baru
                sebelum[tetangga] = node_terdekat

    jalur = []
    node = tujuan
    while node is not None:
        jalur.insert(0, node)
        node = sebelum[node]

    return jalur

# Algoritma Depth First Search (DFS)
def dfs(graf, awal, tujuan, jalur=[]):
    jalur = jalur + [awal]
    if awal == tujuan:
        return [jalur]

    rute = []
    for tetangga in graf[awal]:
        if tetangga not in jalur:
            new_rute = dfs(graf, tetangga, tujuan, jalur)
            rute.extend(new_rute)

    return rute

# Algoritma Breadth First Search (BFS)
def bfs(graf, awal, tujuan):
    queue = [[awal]]

    while queue:
        jalur = queue.pop(0)
        node = jalur[-1]

        if node == tujuan:
            return jalur

        for tetangga in graf[node]:
            if tetangga not in jalur:
                queue.append(jalur + [tetangga])

    return None

# Algoritma Best First Search (Greedy BFS)
def best_first_search(graf, awal, tujuan, heuristic):
    queue = PriorityQueue()
    queue.put((heuristic[awal], [awal]))

    while not queue.empty():
        _, jalur = queue.get()
        node = jalur[-1]

        if node == tujuan:
            return jalur

        for tetangga in graf[node]:
            if tetangga not in jalur:
                queue.put((heuristic[tetangga], jalur + [tetangga]))

    return None

# Menjalankan algoritma
awal, tujuan = 'Arad', 'Bucarest'

jalur_dijkstra = dijkstra(graf, awal, tujuan)
jalur_bfs = bfs(graf, awal, tujuan)
jalur_bestfs = best_first_search(graf, awal, tujuan, heuristic)
rute_dfs = dfs(graf, awal, tujuan)

# Menampilkan hasil
print("\nHasil Pencarian Rute:")
print("-------------------------")

# Dijkstra
print("\n🔹 Rute Terpendek (Dijkstra):")
print(" → ".join(jalur_dijkstra))

# BFS
print("\n🔹 Rute dengan Node Paling Sedikit (BFS):")
print(" → ".join(jalur_bfs))

# Best First Search
print("\n🔹 Rute dengan Perkiraan Heuristic (Best First Search):")
print(" → ".join(jalur_bestfs))

# DFS
print("\n🔹 Semua Rute (DFS):")
table = [[f"Rute {i+1}", " → ".join(jalur)] for i, jalur in enumerate(rute_dfs)]
print(tabulate(table, headers=["No", "Jalur"], tablefmt="grid"))
