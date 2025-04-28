import random

# Fungsi untuk mengubah biner ke desimal
def decode(chromosome):
    return int(chromosome, 2)

# Fungsi fitness
def fitness(x):
    return -(x ** 2) + 10

# Inisialisasi populasi
def generate_population(size, chrom_length):
    return [''.join(random.choice('01') for _ in range(chrom_length)) for _ in range(size)]

# Seleksi induk berdasarkan fitness
def select_parents(population):
    sorted_population = sorted(population, key=lambda chrom: fitness(decode(chrom)), reverse=True)
    return sorted_population[:2]

# Crossover satu titik
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutasi dengan peluang tertentu
def mutate(chromosome, mutation_rate=0.01):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(chromosome)

# Algoritma Genetika
def genetic_algorithm(pop_size=6, chrom_length=5, generations=10):
    population = generate_population(pop_size, chrom_length)
    
    for gen in range(generations):
        print(f"Generasi {gen}:")
        for chrom in population:
            print(f"x = {decode(chrom)}, fitness = {fitness(decode(chrom))}")
        
        parents = select_parents(population)
        print(f"Orangtua terpilih: {decode(parents[0])} dan {decode(parents[1])}")
        
        next_generation = []
        
        while len(next_generation) < pop_size:
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])
        
        population = next_generation[:pop_size]
        print('-' * 30)
    
    # Ambil solusi terbaik
    best = max(population, key=lambda chrom: fitness(decode(chrom)))
    return decode(best), fitness(decode(best))

# Jalankan
best_x, best_fitness = genetic_algorithm()
print(f"Solusi terbaik: x = {best_x} dengan fitness = {best_fitness}")
    