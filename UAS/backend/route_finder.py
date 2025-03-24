const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

app.post("/find-route", (req, res) => {
    const { start, end, algorithm } = req.body;
    
    const pythonProcess = spawn("python3", ["route_finder.py", start, end, algorithm]);
    
    let result = "";
    pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
    });
    
    pythonProcess.stderr.on("data", (data) => {
        console.error(`Error: ${data.toString()}`);
    });
    
    pythonProcess.on("close", (code) => {
        if (code === 0) {
            res.json({ route: result.trim().split(" -> ") });
        } else {
            res.status(500).json({ error: "Failed to find route." });
        }
    });
});

app.get("/test", (req, res) => {
    res.json({ message: "Server is running" });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
