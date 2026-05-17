"""对话框组件模块"""
import time
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QTextEdit,
    QDialogButtonBox, QVBoxLayout, QLabel
)
from core.models import Role
from ui.styles import get_style_sheet

class AddRoleDialog(QDialog):
    """添加角色对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加角色")
        self.setMinimumWidth(400)
        layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入角色罗马音名称")
        self.name_cn_edit = QLineEdit()
        self.name_cn_edit.setPlaceholderText("输入角色中文名称")
        self.category_combo = QComboBox()
        self.category_combo.addItems(["主角", "配角", "反派", "路人", "其他"])
        layout.addRow("角色名 (罗马音):", self.name_edit)
        layout.addRow("中文名:", self.name_cn_edit)
        layout.addRow("分类:", self.category_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.setLayout(layout)
        self.setStyleSheet(get_style_sheet())
    
    def get_role(self):
        return Role(
            id=f"role_{int(time.time())}",
            name=self.name_edit.text().strip(),
            name_cn=self.name_cn_edit.text().strip(),
            category=self.category_combo.currentText()
        )

class BatchImportDialog(QDialog):
    """批量导入角色对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量导入角色")
        self.setMinimumWidth(500)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("每行一个角色，格式: 罗马音,中文名,分类"))
        layout.addWidget(QLabel("例如: Kirito,桐人,主角"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("在此输入角色列表...")
        layout.addWidget(self.text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setStyleSheet(get_style_sheet())
    
    def get_roles(self):
        roles = []
        for line in self.text_edit.toPlainText().strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            name = parts[0] if len(parts) > 0 else ""
            name_cn = parts[1] if len(parts) > 1 else ""
            category = parts[2] if len(parts) > 2 else "其他"
            if name:
                roles.append(Role(
                    id=f"role_{int(time.time())}_{len(roles)}",
                    name=name,
                    name_cn=name_cn,
                    category=category
                ))
        return roles
