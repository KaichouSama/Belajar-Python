const express = require("express");
const cors = require("cors");

const app = express(); // ✅ Deklarasi app sebelum digunakan
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Route utama (opsional, biar tidak error "Cannot GET /")
app.get("/", (req, res) => {
    res.send("Server berjalan dengan baik!");
});

// Route pencarian rute
app.get("/cari-rute", (req, res) => {
    const { asal, tujuan } = req.query;

    if (!asal || !tujuan) {
        return res.status(400).json({ error: "Asal dan tujuan harus dipilih!" });
    }

    // Contoh rute dummy (harus diganti dengan algoritma pencarian rute)
    const rute = [asal, "Sibiu", "Fagaras", tujuan];

    res.json({ rute });  // 🔹 Kirim rute ke frontend
});



// Menjalankan server
app.listen(PORT, () => {
    console.log(`✅ Server berjalan di http://localhost:${PORT}`);
});

