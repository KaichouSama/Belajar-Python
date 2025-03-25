import sys

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

# Algoritma Dijkstra
def dijkstra(graf, awal, tujuan):
    jarak = {node: float('inf') for node in graf}
    jarak[awal] = 0
    sebelumnya = {node: None for node in graf}
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
                sebelumnya[tetangga] = node_terdekat

    jalur = []
    node = tujuan
    while node is not None:
        jalur.insert(0, node)
        node = sebelumnya[node]

    return " → ".join(jalur)

# Algoritma BFS
def bfs(graf, awal, tujuan):
    queue = [[awal]]

    while queue:
        jalur = queue.pop(0)
        node = jalur[-1]

        if node == tujuan:
            return " → ".join(jalur)

        for tetangga in graf[node]:
            if tetangga not in jalur:
                queue.append(jalur + [tetangga])

    return "Tidak ditemukan jalur"

# Eksekusi program berdasarkan input dari Node.js
if __name__ == "__main__":
    start = sys.argv[1]
    goal = sys.argv[2]
    algorithm = sys.argv[3]

    if algorithm == "dijkstra":
        print(dijkstra(graf, start, goal))
    elif algorithm == "bfs":
        print(bfs(graf, start, goal))
    else:
        print("Algoritma tidak valid!")
