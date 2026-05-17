from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ROLES_FILE = DATA_DIR / "roles.json"
NSFW_SUSPICIOUS_DIR = DATA_DIR / "nsfw_suspicious"
UNTRAINABLE_DIR = DATA_DIR / "无法训练"
UNTRAINABLE_R18_DIR = UNTRAINABLE_DIR / "R18"
UNTRAINABLE_MULTI_DIR = UNTRAINABLE_DIR / "多角色"
UNTRAINABLE_OTHER_DIR = UNTRAINABLE_DIR / "其他"

for d in [DATA_DIR, ANNOTATIONS_DIR, NSFW_SUSPICIOUS_DIR, UNTRAINABLE_DIR, UNTRAINABLE_R18_DIR, UNTRAINABLE_MULTI_DIR, UNTRAINABLE_OTHER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_data_dir():
    return DATA_DIR


def get_annotations_dir():
    return ANNOTATIONS_DIR


def get_roles_file():
    return ROLES_FILE


def get_untrainable_dirs():
    return {
        "R18": UNTRAINABLE_R18_DIR,
        "多角色": UNTRAINABLE_MULTI_DIR,
        "其他": UNTRAINABLE_OTHER_DIR
    }


def get_excluded_dirs():
    return {"无法训练", "annotations", "nsfw_suspicious"}
