"""动漫角色标注工具 - Web服务入口"""
import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from data import load_annotations, save_annotation, delete_annotation, load_roles, save_roles, ANNOTATIONS_DIR, ROLES_FILE
from services import DATA_DIR, MODELS_DIR, get_untrainable_dirs

app = FastAPI(title="Anime Role Annotation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent.parent.parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

annotations = {}
roles = []
images = []


def scan_directory():
    global images
    data_dir = DATA_DIR
    if not data_dir.exists():
        images = []
        return
    images = [
        {"path": str(p), "filename": p.name}
        for p in data_dir.glob("*/*.*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
    ]
    images.sort(key=lambda x: x["filename"])


@app.on_event("startup")
async def startup():
    global annotations, roles
    annotations = load_annotations()
    roles = load_roles()
    scan_directory()


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent.parent.parent / "web" / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return "<html><body><h1>Anime Role Annotation API</h1></body></html>"


@app.get("/api/images")
async def get_images():
    return {"images": images, "total": len(images)}


@app.get("/api/annotations")
async def get_annotations():
    return {"annotations": annotations}


@app.get("/api/roles")
async def get_roles():
    return {"roles": [r.__dict__ for r in roles]}


@app.post("/api/annotations/{image_path:path}")
async def create_or_update_annotation(image_path: str, annotation_data: dict):
    from data import AnnotationData
    ann = AnnotationData(
        role_ids=annotation_data.get("role_ids", []),
        is_multi_role=annotation_data.get("is_multi_role", False),
        is_nsfw=annotation_data.get("is_nsfw", False),
        notes=annotation_data.get("notes", ""),
        timestamp=annotation_data.get("timestamp", "")
    )
    save_annotation(image_path, ann)
    annotations[image_path] = ann
    return {"status": "ok", "annotation": ann.__dict__}


@app.delete("/api/annotations/{image_path:path}")
async def remove_annotation(image_path: str):
    delete_annotation(image_path)
    if image_path in annotations:
        del annotations[image_path]
    return {"status": "ok"}


@app.get("/api/stats")
async def get_stats():
    total = len(images)
    annotated = len(annotations)
    role_count = len(roles)
    return {
        "total": total,
        "annotated": annotated,
        "unannotated": total - annotated,
        "role_count": role_count
    }


if __name__ == "__main__":
    print("=" * 50)
    print("动漫角色标注工具")
    print("=" * 50)
    print(f"数据目录: {DATA_DIR}")
    print(f"标注目录: {ANNOTATIONS_DIR}")
    print(f"角色文件: {ROLES_FILE}")
    print("=" * 50)
    print("访问地址: http://localhost:8090")
    print("API文档: http://localhost:8090/docs")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8090)
