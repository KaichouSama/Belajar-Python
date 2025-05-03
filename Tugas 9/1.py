# Sistem Pakar Hewan - Forward Chaining sesuai Tabel 8.1

# Basis Pengetahuan
rules = [
    (["has hair"], "mammal"),
    (["gives milk"], "mammal"),
    (["has feathers"], "bird"),
    (["lays eggs"], "bird"),
    (["flies"], "bird"),
    (["eats meat"], "carnivore"),
    (["has pointed teeth"], "carnivore"),
    (["has claws"], "carnivore"),
    (["has forward eyes"], "carnivore"),
    (["mammal", "carnivore", "tawny color", "has dark spots"], "cheetah"),
    (["mammal", "carnivore", "has mane"], "lion"),
    (["bird", "flies", "lays eggs"], "eagle"),
    (["mammal", "has black stripes"], "tiger"),              # Rule 11
    (["mammal", "is black and white"], "zebra"),             # Rule 12
    (["mammal", "has long neck", "has long legs"], "giraffe"), # Rule 13
    (["mammal", "has trunk", "has tusks"], "elephant"),      # Rule 14
    (["bird", "swims", "is black and white"], "penguin")     # Rule 15
]

# Fungsi forward chaining
def forward_chaining(facts):
    inferred = set(facts)
    added = True
    while added:
        added = False
        for conditions, conclusion in rules:
            if all(condition in inferred for condition in conditions) and conclusion not in inferred:
                inferred.add(conclusion)
                added = True
    return inferred

# Input pengguna
print("=== Sistem Pakar Identifikasi Hewan ===")
print("Masukkan fakta hewan (pisahkan dengan koma):")
print("Contoh: has hair, has claws, has mane")
user_input = input("Fakta: ").lower()

# Proses
facts = [fact.strip() for fact in user_input.split(",") if fact.strip()]
result = forward_chaining(facts)

# Output
print("\nFakta yang diperoleh:")
for item in result:
    print("-", item)

# Deteksi hewan
hewan = [i for i in result if i in ["cheetah", "lion", "eagle", "tiger", "zebra", "giraffe", "elephant", "penguin"]]
if hewan:
    print("\nHewan teridentifikasi:")
    for h in hewan:
        print(f"=> {h}")
else:
    print("\nTidak ditemukan hewan spesifik berdasarkan fakta yang diberikan.")
