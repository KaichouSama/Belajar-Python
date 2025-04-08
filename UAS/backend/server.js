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

// const express = require('express');
// const cors = require('cors');
// const { spawn } = require('child_process');

// const app = express();
// const PORT = 5000;

// app.use(cors());
// app.use(express.json());

// // API untuk menerima kota awal dan tujuan, lalu memanggil script Python
// app.post('/cari-rute', (req, res) => {
//     const { start, goal, algorithm } = req.body;

//     if (!start || !goal || !algorithm) {
//         return res.status(400).json({ error: "Pastikan semua data dikirim!" });
//     }

//     // Panggil script Python dengan parameter (start, goal, algorithm)
//     const pythonProcess = spawn('python', ['backend/script.py', start, goal, algorithm]);

//     let resultData = '';

//     pythonProcess.stdout.on('data', (data) => {
//         resultData += data.toString();
//     });

//     pythonProcess.stderr.on('data', (data) => {
//         console.error(`Error: ${data}`);
//     });

//     pythonProcess.on('close', (code) => {
//         if (code === 0) {
//             res.json({ route: resultData.trim().split(' → ') });
//         } else {
//             res.status(500).json({ error: "Gagal menjalankan script Python." });
//         }
//     });
// });

// // Jalankan server
// app.listen(PORT, () => {
//     console.log(`✅ Server berjalan di http://localhost:${PORT}`);
// });
