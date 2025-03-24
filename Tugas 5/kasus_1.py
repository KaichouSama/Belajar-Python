import sys
from queue import PriorityQueue
from tabulate import tabulate

# Representasi graf dengan jarak antar kota
graf = {
    'Brebes': {'Tegal': 40.9, 'Slawi': 28.6},
    'Tegal': {'Brebes': 40.9, 'Slawi': 1.3, 'Pemalang': 32.4},
    'Pemalang': {'Tegal': 32.4, 'Purbalingga': 68.8, 'Pekalongan': 47.2},
    'Pekalongan': {'Pemalang': 47.2, 'Kendal': 74.7},
    'Kendal': {'Pekalongan': 74.7, 'Temanggung': 71.2, 'Semarang': 31.7},
    'Semarang': {'Kendal': 31.7, 'Salatiga': 60.9, 'Demak': 28.8},
    'Demak': {'Semarang': 28.8, 'Purwodadi': 39, 'Kudus': 50.9},
    'Kudus': {'Demak': 50.9, 'Purwodadi': 47.9, 'Rembang': 60.7},
    'Rembang': {'Kudus': 60.7, 'Blora': 36.1},
    'Slawi': {'Brebes': 28.6, 'Tegal': 1.3, 'Purwokerto': 88.7},
    'Purwodadi': {'Demak': 39, 'Kudus': 47.9, 'Blora': 65.3, 'Solo': 67.4},
    'Blora': {'Purwodadi': 65.3, 'Rembang': 36.1, 'Sragen': 83.7},
    'Purwokerto': {'Slawi': 88.7, 'Purbalingga': 16.8, 'Cilacap': 49.7, 'Kroya': 30.5, 'Kebumen': 71.2},
    'Purbalingga': {'Purwokerto': 16.8, 'Pemalang': 68.8, 'Banjarnegara': 33.4},
    'Banjarnegara': {'Purbalingga': 33.4, 'Wonosobo': 42.5},
    'Wonosobo': {'Banjarnegara': 42.5, 'Temanggung': 40, 'Magelang': 60.4},
    'Temanggung': {'Wonosobo': 40, 'Magelang': 22.8, 'Salatiga': 55.6, 'Kendal': 71.2},
    'Salatiga': {'Temanggung': 55.6, 'Semarang': 60.9, 'Boyolali': 24.4},
    'Cilacap': {'Purwokerto': 49.7, 'Kroya': 47.6},
    'Kroya': {'Cilacap': 47.6, 'Purwokerto': 30.5, 'Kebumen': 53.6},
    'Kebumen': {'Kroya': 53.6, 'Purwokerto': 71.2, 'Purworejo': 42},
    'Purworejo': {'Kebumen': 42, 'Magelang': 8.6},
    'Magelang': {'Purworejo': 8.6, 'Wonosobo': 60.4, 'Temanggung': 22.8, 'Boyolali': 62.4},
    'Boyolali': {'Magelang': 62.4, 'Salatiga': 24.4, 'Solo': 28.6, 'Klaten': 35.3},
    'Solo': {'Boyolali': 28.6, 'Purwodadi': 67.4, 'Sragen': 34.8, 'Sukoharjo': 12.6},
    'Sragen': {'Solo': 34.8, 'Blora': 83.7},
    'Sukoharjo': {'Solo': 12.6, 'Wonogiri': 39.7},
    'Wonogiri': {'Sukoharjo': 39.7},
    'Klaten': {'Boyolali': 35.3}
}

# Heuristic untuk Best First Search (jarak langsung ke Bucharest)
heuristic = {
    'Brebes' : 94,
    'Tegal' : 92,
    'Pemalang' : 95,
    'Pekalongan' : 115,
    'Kendal' : 156,
    'Semarang' : 174,
    'Demak' : 199,
    'Kudus' : 224,
    'Rembang' : 278,
    'Slawi' : 80,
    'Purwodadi' : 219,
    'Blora' : 276,
    'Purwokerto' : 39,
    'Purbalingga' : 51,
    'Banjarnegara' : 65,
    'Wonosobo' : 105,
    'Temanggung' : 134,
    'Salatiga' : 169,
    'Cilacap' : 0,
    'Kroya' : 26,
    'Kebumen' : 71,
    'Purworejo' : 109,
    'Magelang' : 134,
    'Boyolali' : 174,
    'Solo' : 199,
    'Sragen' : 223,
    'Sukoharjo' : 197,
    'Wonogiri' : 211,
    'Klaten' : 181
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
awal, tujuan = 'Semarang', 'Cilacap'

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
# print("\n🔹 Semua Rute (DFS):")
# table = [[f"Rute {i+1}", " → ".join(jalur)] for i, jalur in enumerate(rute_dfs)]
# print(tabulate(table, headers=["No", "Jalur"], tablefmt="grid"))
