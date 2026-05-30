class AnnotationData:
    def __init__(self):
        self.image_path = ""
        self.roles = []
        self.is_multi_role = False
        self.is_nsfw = False
        self.nsfw_reason = ""
        self.notes = ""
        self.annotator = "anonymous"
        self.timestamp = ""

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "roles": self.roles,
            "is_multi_role": self.is_multi_role,
            "is_nsfw": self.is_nsfw,
            "nsfw_reason": self.nsfw_reason,
            "notes": self.notes,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d):
        ann = AnnotationData()
        ann.image_path = d.get("image_path", "")
        ann.roles = d.get("roles", [])
        ann.is_multi_role = d.get("is_multi_role", False)
        ann.is_nsfw = d.get("is_nsfw", False)
        ann.nsfw_reason = d.get("nsfw_reason", "")
        ann.notes = d.get("notes", "")
        ann.annotator = d.get("annotator", "anonymous")
        ann.timestamp = d.get("timestamp", "")
        return ann


class Role:
    def __init__(self, id="", name="", name_cn="", category=""):
        self.id = id
        self.name = name
        self.name_cn = name_cn
        self.category = category
        self.count = 0
        self.last_modified = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "name_cn": self.name_cn,
            "category": self.category,
            "count": self.count,
            "last_modified": self.last_modified,
        }

    @staticmethod
    def from_dict(d):
        role = Role()
        role.id = d.get("id", "")
        role.name = d.get("name", "")
        role.name_cn = d.get("name_cn", "")
        role.category = d.get("category", "")
        role.count = d.get("count", 0)
        role.last_modified = d.get("last_modified", "")
        return role

    def __repr__(self):
        return f"Role({self.name}, {self.name_cn}, {self.category})"
