import React, { useState } from "react";

const App = () => {
  const [asal, setAsal] = useState("");
  const [tujuan, setTujuan] = useState("");
  const [hasil, setHasil] = useState([]);

  const kota = [
    "Arad", "Zerind", "Oradea", "Sibiu", "Timisoara", "Lugoj",
    "Mehadia", "Drobeta", "Craiova", "Rimnicu Vilcea", "Pitesti",
    "Fagaras", "Bucarest", "Giurgiu", "Urziceni", "Vaslui", "Iasi",
    "Neamt", "Hirsova", "Eforie"
  ];

  const cariRute = async () => {
    if (!asal || !tujuan) {
      alert("Pilih kota asal dan tujuan!");
      return;
    }
    
    try {
      const res = await fetch(`http://localhost:5000/cari-rute?asal=${asal}&tujuan=${tujuan}`);
      const data = await res.json();
      setHasil(data.rute || []);
    } catch (error) {
      console.error("Gagal mengambil data:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>🔍 Cari Rute Terpendek</h1>
      
      <select value={asal} onChange={(e) => setAsal(e.target.value)}>
        <option value="">Pilih Kota Asal</option>
        {kota.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>

      <select value={tujuan} onChange={(e) => setTujuan(e.target.value)}>
        <option value="">Pilih Kota Tujuan</option>
        {kota.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>

      <button onClick={cariRute}>Cari</button>

      {hasil.length > 0 && (
        <div>
          <h2>Rute Terpendek:</h2>
          <p>{hasil.join(" → ")}</p>
        </div>
      )}
    </div>
  );
};

export default App;
