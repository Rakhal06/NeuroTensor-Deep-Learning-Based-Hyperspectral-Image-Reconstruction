# DeepTensor++ Research Navigator
**Interactive Code Navigation & Verification Engine for Phase 2 Faculty Evaluation**

The **DeepTensor++ Research Navigator** is an interactive local web application designed to demonstrate the complete research journey of DeepTensor++. It pairs directly with the academic PowerPoint presentation, enabling faculty and evaluators to click any research concept or slide topic and immediately inspect its exact, verified implementation in the underlying PyTorch source code with line-level highlighting.

---

## 1. Key Features

- **Monaco Code Editor:** Embedded Visual Studio Code editor engine running read-only with full Python syntax highlighting, automatic line-range highlighting, and jump-to-definition.
- **24 Synchronized Research Topics:** Covers Topics `DT-01` through `DT-24` across 6 research sections with LaTeX math formulas, scientific concepts, research significance, and faculty viva defenses.
- **Direct Presentation-to-Code Hyperlinks:** Slides in `evaluation/presentation/DeepTensor_Phase2_Evaluation.pptx` link directly to `http://localhost:8000/?topic=DT-XX`.
- **Controlled 5-Model Comparative Matrix:** Dedicated experiment view comparing Baseline CNN, Hybrid CNN-Transformer, CNN Control, Pointwise Control, and Pure Transformer across 4 distinct application testbeds.
- **Presentation Mode:** Built-in slide-by-slide step-through toolbar with keyboard arrow navigation (`←` / `→`) synchronizing code and theory simultaneously.
- **AST Code Search:** Live query search across all topic formulations, mathematical terms, and Python class/def symbols.
- **Secure Local Backend:** FastAPI backend enforcing strict repository bounds and preventing unauthorized directory traversal.

---

## 2. Quick Start & Launch

### Prerequisites
- Python 3.9+ installed with the existing `.venv` environment.
- Required packages installed: `fastapi`, `uvicorn`, `python-pptx` (already installed in workspace `.venv`).

### Launch Command

From the repository root (`E:\Rakhal_Tensor`):

```bash
# Using the workspace virtual environment
.venv\Scripts\python.exe evaluation/code_navigator/backend/main.py
```

Or run via Uvicorn directly:

```bash
.venv\Scripts\python.exe -m uvicorn evaluation.code_navigator.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The web dashboard will be accessible immediately at:
👉 **`http://localhost:8000`**

---

## 3. URL Query Parameter Navigation

The Navigator supports direct deep-linking for presentation slides and faculty Q&A:

- **Navigate to specific topic:**  
  `http://localhost:8000/?topic=DT-06` (Loads Topic DT-06: U, V and W Factor Generation)  
  `http://localhost:8000/?topic=DT-08` (Loads Topic DT-08: CP/PARAFAC Decomposition)  
  `http://localhost:8000/?topic=DT-15` (Loads Topic DT-15: Five Model Architectures)  
  `http://localhost:8000/?topic=DT-22` (Loads Topic DT-22: Adaptive Rank Selection)

- **Open specific experiment modal:**  
  `http://localhost:8000/?exp=figure09` (Opens ICVL Hyperspectral Denoising)  
  `http://localhost:8000/?exp=figure10` (Opens Yale Face B Photometric Denoising)  
  `http://localhost:8000/?exp=figure11` (Opens Video Compressive Sensing)  
  `http://localhost:8000/?exp=figure12` (Opens PET/CT Radon Tomography)

---

## 4. Architecture & File Structure

```
evaluation/code_navigator/
├── backend/
│   └── main.py                 # FastAPI application serving REST endpoints & static files
├── frontend/
│   ├── index.html              # Modern IDE layout with Monaco Editor & KaTeX
│   ├── styles.css              # Dark theme stylesheet (JetBrains/VS Code aesthetic)
│   └── app.js                  # Frontend state management, Monaco line highlighter, hotkeys
├── data/
│   ├── research_topics.json    # Complete metadata for Topics DT-01 to DT-24
│   ├── topic_code_map.json     # Exact file paths, line ranges, and line highlight arrays
│   └── experiment_map.json     # Controlled 4-experiment metrics and winning model analysis
└── README.md                   # This documentation guide
```

---

## 5. Security & Isolation Safeguards

1. **Path Traversal Protection:** The backend normalizes all requested file paths using `os.path.abspath` and rejects any request attempting to escape the project root.
2. **Environment Isolation:** Protected folders such as `.venv`, `.git`, and `__pycache__` cannot be read via the API.
3. **Read-Only Codebase Guarantee:** The Code Navigator contains zero write/update endpoints. The repository source code and historical experiment checkpoints remain 100% frozen and unmodified.

