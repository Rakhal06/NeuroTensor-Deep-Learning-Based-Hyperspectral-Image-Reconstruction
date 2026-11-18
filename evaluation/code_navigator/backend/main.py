"""
DeepTensor++ Research Navigator — FastAPI Backend
Provides REST APIs for research topics, code indexing, source retrieval with line highlighting,
and experiment verification metrics.
"""

import os
import sys
import json
import ast
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

# Base workspace directory
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = FastAPI(
    title="DeepTensor++ Research Navigator API",
    description="Interactive Code Navigation & Verification Backend for Phase 2 Faculty Evaluation",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: Load JSON datasets
def load_json(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

RESEARCH_TOPICS = load_json('research_topics.json')
TOPIC_CODE_MAP = load_json('topic_code_map.json')
EXPERIMENT_MAP = load_json('experiment_map.json')

# Path security verification
def resolve_safe_path(rel_path: str) -> str:
    """Resolve and enforce that requested path remains strictly within WORKSPACE_DIR."""
    # Clean leading slashes
    clean_rel = rel_path.lstrip('/\\')
    full_path = os.path.abspath(os.path.join(WORKSPACE_DIR, clean_rel))
    
    if not full_path.startswith(WORKSPACE_DIR):
        raise HTTPException(status_code=403, detail="Access denied: Path traversal outside repository")
    
    # Disallow access to .venv, .git, sensitive system directories
    parts = Path(full_path).parts
    if '.git' in parts or '.venv' in parts or '__pycache__' in parts:
        raise HTTPException(status_code=403, detail="Access denied: Protected repository directory")
        
    return full_path


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "service": "DeepTensor++ Research Navigator",
        "workspace": WORKSPACE_DIR,
        "total_topics": len(RESEARCH_TOPICS),
        "total_experiments": len(EXPERIMENT_MAP),
        "evaluation_phase": "Phase 2 Faculty Evaluation"
    }


@app.get("/api/topics")
def get_topics():
    """Return summary list of all 24 research topics."""
    return RESEARCH_TOPICS


@app.get("/api/topics/{topic_id}")
def get_topic_detail(topic_id: str):
    """Return detailed metadata, formula, context, and code link for a given topic."""
    tid = topic_id.upper()
    for topic in RESEARCH_TOPICS:
        if topic['id'] == tid:
            return topic
    raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")


@app.get("/api/experiments")
def get_experiments():
    """Return comparative data across the 4 application experiments."""
    return EXPERIMENT_MAP


@app.get("/api/experiments/{exp_id}")
def get_experiment_detail(exp_id: str):
    """Return detailed metrics and image asset paths for a specific experiment."""
    eid = exp_id.lower()
    if eid in EXPERIMENT_MAP:
        return EXPERIMENT_MAP[eid]
    raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")


@app.get("/api/code")
def get_code(
    file_path: str = Query(..., description="Relative file path within repository"),
    line_start: Optional[int] = Query(None, description="Starting line number (1-indexed)"),
    line_end: Optional[int] = Query(None, description="Ending line number (1-indexed)")
):
    """Retrieve full file content and metadata for Monaco Editor display."""
    safe_path = resolve_safe_path(file_path)
    
    if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
    try:
        with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    lines = content.split('\n')
    total_lines = len(lines)

    # Parse AST for Python symbols if python file
    symbols = []
    if file_path.endswith('.py'):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append({
                        "name": node.name,
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        "line": getattr(node, 'lineno', 1),
                        "end_line": getattr(node, 'end_lineno', getattr(node, 'lineno', 1))
                    })
        except Exception:
            pass

    return {
        "file_path": file_path.replace('\\', '/'),
        "total_lines": total_lines,
        "content": content,
        "line_start": line_start,
        "line_end": line_end,
        "symbols": symbols
    }


@app.get("/api/search")
def search_repository(
    q: str = Query(..., min_length=2, description="Search query keyword or symbol")
):
    """Search topics and code symbols across the repository."""
    q_lower = q.lower()
    matched_topics = []
    
    # 1. Search Research Topics (rich text search across all attributes)
    for t in RESEARCH_TOPICS:
        searchable_text = f"{t.get('title','')} {t.get('id','')} {t.get('concept','')} {t.get('math_explanation','')} {t.get('research_context','')} {t.get('code_reference',{}).get('symbol_name','')} {t.get('code_reference',{}).get('summary','')}".lower()
        if q_lower in searchable_text:
            matched_topics.append({
                "type": "topic",
                "id": t['id'],
                "title": t['title'],
                "section": t['section_title'],
                "file_path": t['code_reference']['file_path'],
                "line_start": t['code_reference']['line_start'],
                "line_end": t['code_reference']['line_end']
            })

    # 2. Search key python files for matching class/def symbols or content lines
    matched_symbols = []
    search_dirs = ['src/deeptensor', 'experiments/hybrid_prototypes', 'experiments/reproduction', 'experiments/extensions']
    for sdir in search_dirs:
        full_dir = os.path.join(WORKSPACE_DIR, sdir)
        if not os.path.exists(full_dir):
            continue
        for root, _, files in os.walk(full_dir):
            for file in files:
                if file.endswith('.py'):
                    p = os.path.join(root, file)
                    rel_p = os.path.relpath(p, WORKSPACE_DIR).replace('\\', '/')
                    try:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                            content = fp.read()
                        if q_lower in content.lower():
                            for idx, line in enumerate(content.split('\n'), start=1):
                                if q_lower in line.lower():
                                    matched_symbols.append({
                                        "type": "symbol",
                                        "name": line.strip()[:80],
                                        "file_path": rel_p,
                                        "line": idx
                                    })
                                    if len(matched_symbols) >= 15:
                                        break
                    except Exception:
                        pass
            if len(matched_symbols) >= 15:
                break

    return {
        "query": q,
        "matched_topics": matched_topics,
        "matched_symbols": matched_symbols[:15]
    }


@app.get("/api/image")
def get_image(path: str = Query(..., description="Relative path to image asset")):
    """Safely stream result images to the web client."""
    safe_path = resolve_safe_path(path)
    if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    ext = os.path.splitext(safe_path)[1].lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    }
    
    if ext not in media_types:
        raise HTTPException(status_code=400, detail="Invalid image file type")
        
    return FileResponse(safe_path, media_type=media_types[ext])


# Mount Frontend Static Assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>DeepTensor++ Research Navigator Backend Online</h2><p>Frontend assets loading...</p>")


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("NAVIGATOR_PORT", 8000))
    print(f"Starting DeepTensor++ Research Navigator on http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, app_dir=os.path.dirname(__file__))
