

import heapq

def best_first_search(puzzle, goal):
    # Definisi fungsi evaluasi
    def f(n):
        return n['kedalaman'] + n['jumlah_salah']

    # Fungsi untuk menghitung jumlah angka yang salah posisi
    def hitung_jumlah_salah(puzzle, goal):
        jumlah_salah = 0
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] != goal[i][j]:
                    jumlah_salah += 1
        return jumlah_salah

    # Buat node awal
    node_awal = {
        'puzzle': puzzle,
        'kedalaman': 0,
        'jumlah_salah': hitung_jumlah_salah(puzzle, goal),
        'langkah': []
    }

    # Buat antrian
    antrian = []

    # Tambahkan node awal ke dalam antrian
    antrian.append(node_awal)

    # Ulangi langkah sampai antrian kosong
    while antrian:
        # Ambil node dengan nilai f(n) terendah
        node = min(antrian, key=lambda n: f(n))
        antrian.remove(node)

        # Jika node tersebut merupakan solusi yang benar
        if node['jumlah_salah'] == 0:
            return node['langkah']

        # Buat node-node baru yang merepresentasikan kemungkinan langkah selanjutnya
        for i in range(3):
            for j in range(3):
                # Jika angka di posisi (i, j) bukan angka yang benar
                if puzzle[i][j] != goal[i][j]:
                    # Buat node baru
                    node_baru = {
                        'puzzle': [row[:] for row in node['puzzle']],
                        'kedalaman': node['kedalaman'] + 1,
                        'jumlah_salah': 0,
                        'langkah': node['langkah'] + [(i, j)]
                    }

                    # Tukar angka di posisi (i, j) dengan angka yang benar
                    node_baru['puzzle'][i][j] = goal[i][j]

                    # Hitung jumlah angka yang salah posisi
                    node_baru['jumlah_salah'] = hitung_jumlah_salah(node_baru['puzzle'], goal)

                    # Tambahkan node baru ke dalam antrian
                    antrian.append(node_baru)

    # Jika tidak ada solusi
    return None

# Contoh puzzle
puzzle = [
    [7, 2, 4],
    [5, 0, 6],
    [8, 3, 1]
]

goal = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Cari solusi
solusi = best_first_search(puzzle, goal)

# Tampilkan solusi
if solusi:
    print("Solusi:")
    for langkah in solusi:
        print(f"Langkah {langkah}: Tukar angka di posisi ({langkah[0]}, {langkah[1]})")
else:
    print("Tidak ada solusi")


