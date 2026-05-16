"""
动漫角色标注工具 - FastAPI后端服务
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="动漫角色标注工具", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"
DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ROLES_FILE = DATA_DIR / "roles.json"
NSFW_SUSPICIOUS_DIR = DATA_DIR / "nsfw_suspicious"

for d in [DATA_DIR, ANNOTATIONS_DIR, NSFW_SUSPICIOUS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class Annotation(BaseModel):
    image_path: str
    roles: List[str]
    is_multi_role: bool = False
    is_nsfw: bool = False
    nsfw_reason: Optional[str] = None
    notes: Optional[str] = ""
    annotator: Optional[str] = "anonymous"
    timestamp: Optional[str] = None

class Role(BaseModel):
    id: str
    name: str
    name_cn: Optional[str] = ""
    category: Optional[str] = ""
    description: Optional[str] = ""

def load_roles() -> List[Role]:
    """加载角色列表"""
    if not ROLES_FILE.exists():
        return []
    try:
        with open(ROLES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [Role(**r) for r in data]
    except:
        return []

def save_roles(roles: List[Role]):
    """保存角色列表"""
    with open(ROLES_FILE, 'w', encoding='utf-8') as f:
        json.dump([r.dict() for r in roles], f, ensure_ascii=False, indent=2)

def load_annotations() -> dict:
    """加载所有标注"""
    annotations = {}
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    annotations[data.get('image_path', '')] = data
            except:
                pass
    return annotations

def save_annotation(annotation: Annotation):
    """保存单个标注"""
    if annotation.timestamp is None:
        annotation.timestamp = datetime.now().isoformat()

    safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in annotation.image_path)
    safe_name = safe_name[:100]
    file_path = ANNOTATIONS_DIR / f"{safe_name}.json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(annotation.dict(), f, ensure_ascii=False, indent=2)

def scan_images(directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')) -> List[dict]:
    """扫描目录下的所有图片"""
    images = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return images

    for ext in extensions:
        for img_path in sorted(dir_path.rglob(f"*{ext}")):
            rel_path = str(img_path.relative_to(dir_path.parent))
            images.append({
                "path": str(img_path),
                "relative_path": rel_path,
                "filename": img_path.name,
                "size": img_path.stat().st_size
            })

    for ext in extensions:
        for img_path in sorted(dir_path.rglob(f"*{ext.upper()}")):
            rel_path = str(img_path.relative_to(dir_path.parent))
            if not any(i['relative_path'] == rel_path for i in images):
                images.append({
                    "path": str(img_path),
                    "relative_path": rel_path,
                    "filename": img_path.name,
                    "size": img_path.stat().st_size
                })

    return sorted(images, key=lambda x: x['relative_path'])

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回标注工具主页面"""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html><head><title>标注工具</title></head>
    <body><h1>标注工具</h1>
    <p>请访问 <a href="/static/index.html">标注界面</a></p>
    </body></html>
    """

@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "annotation_tool"}

@app.post("/api/directory/scan")
async def scan_directory(directory: str = Body(..., embed=True)):
    """扫描指定目录获取图片列表"""
    images = scan_images(directory)
    return {
        "success": True,
        "directory": directory,
        "count": len(images),
        "images": images
    }

@app.get("/api/annotations")
async def get_annotations():
    """获取所有标注"""
    annotations = load_annotations()
    return {"success": True, "annotations": annotations}

@app.post("/api/annotation")
async def create_or_update_annotation(annotation: Annotation):
    """创建或更新标注"""
    try:
        save_annotation(annotation)
        return {"success": True, "message": "标注已保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/annotation/batch")
async def batch_create_annotations(annotations: List[Annotation]):
    """批量创建标注"""
    saved = 0
    for annotation in annotations:
        try:
            save_annotation(annotation)
            saved += 1
        except:
            pass
    return {"success": True, "saved": saved, "total": len(annotations)}

@app.get("/api/roles")
async def get_roles():
    """获取角色列表"""
    roles = load_roles()
    return {"success": True, "roles": [r.dict() for r in roles]}

@app.post("/api/roles")
async def add_role(role: Role):
    """添加角色"""
    roles = load_roles()
    if any(r.id == role.id for r in roles):
        raise HTTPException(status_code=400, detail="角色ID已存在")
    roles.append(role)
    save_roles(roles)
    return {"success": True, "message": "角色已添加"}

@app.put("/api/roles/{role_id}")
async def update_role(role_id: str, role: Role):
    """更新角色"""
    roles = load_roles()
    for i, r in enumerate(roles):
        if r.id == role_id:
            roles[i] = role
            save_roles(roles)
            return {"success": True, "message": "角色已更新"}
    raise HTTPException(status_code=404, detail="角色不存在")

@app.delete("/api/roles/{role_id}")
async def delete_role(role_id: str):
    """删除角色"""
    roles = load_roles()
    roles = [r for r in roles if r.id != role_id]
    save_roles(roles)
    return {"success": True, "message": "角色已删除"}

@app.post("/api/roles/import")
async def import_roles(roles: List[Role]):
    """批量导入角色"""
    existing = {r.id: r for r in load_roles()}
    for role in roles:
        existing[role.id] = role
    save_roles(list(existing.values()))
    return {"success": True, "count": len(roles)}

@app.get("/api/export/csv")
async def export_csv():
    """导出CSV格式标注文件"""
    annotations = load_annotations()
    lines = ["image_path,roles,is_multi_role,is_nsfw,nsfw_reason,notes,annotator,timestamp"]

    for ann in annotations.values():
        roles_str = "|".join(ann.get('roles', []))
        lines.append(f"{ann.get('image_path', '')},{roles_str},{ann.get('is_multi_role', False)},{ann.get('is_nsfw', False)},{ann.get('nsfw_reason', '')},{ann.get('notes', '')},{ann.get('annotator', '')},{ann.get('timestamp', '')}")

    return {"success": True, "csv": "\n".join(lines)}

@app.get("/api/export/json")
async def export_json():
    """导出JSON格式标注文件"""
    annotations = load_annotations()
    return {"success": True, "annotations": list(annotations.values())}

@app.post("/api/nsfw/move")
async def move_nsfw_image(source_path: str = Body(..., embed=True)):
    """将疑似R18图片移动到可疑目录"""
    source = Path(source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    dest = NSFW_SUSPICIOUS_DIR / source.name
    shutil.move(str(source), str(dest))
    return {"success": True, "new_path": str(dest)}

@app.get("/api/stats")
async def get_stats():
    """获取标注统计信息"""
    annotations = load_annotations()
    roles = load_roles()

    role_counts = {}
    nsfw_count = 0
    multi_role_count = 0

    for ann in annotations.values():
        if ann.get('is_nsfw'):
            nsfw_count += 1
        if ann.get('is_multi_role'):
            multi_role_count += 1
        for role in ann.get('roles', []):
            role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "success": True,
        "total_annotations": len(annotations),
        "total_roles": len(roles),
        "role_counts": role_counts,
        "nsfw_count": nsfw_count,
        "multi_role_count": multi_role_count
    }

if __name__ == "__main__":
    print("=" * 50)
    print("🎬 动漫角色标注工具")
    print("=" * 50)
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"📁 标注目录: {ANNOTATIONS_DIR}")
    print(f"👥 角色文件: {ROLES_FILE}")
    print("=" * 50)
    print("🌐 访问地址: http://localhost:8090")
    print("📖 API文档: http://localhost:8090/docs")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8090)