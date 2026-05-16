"""
动漫角色标注工具 - PyQt5独立桌面版本
支持Windows、Mac、Linux跨平台运行
"""
import os
import sys
import json
import threading
import socket
import logging
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QShortcut, QGroupBox, QCheckBox, QComboBox,
    QProgressBar, QStatusBar, QMenuBar, QMenu, QAction, QToolBar,
    QSplitter, QListWidget, QListWidgetItem, QAbstractItemView,
    QDialog, QFormLayout, QDialogButtonBox, QScrollArea, QFrame,
    QSlider, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QPixmap, QImage, QPainter, QPen
from PyQt5.QtNetwork import QLocalServer, QLocalSocket
import socket
import webbrowser

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
file_handler = logging.FileHandler(LOG_DIR / f"annotation_tool_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger("AnnotationTool")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ROLES_FILE = DATA_DIR / "roles.json"
NSFW_SUSPICIOUS_DIR = DATA_DIR / "nsfw_suspicious"
UNTRAINABLE_DIR = DATA_DIR / "无法训练"
UNTRAINABLE_R18_DIR = UNTRAINABLE_DIR / "R18"
UNTRAINABLE_MULTI_DIR = UNTRAINABLE_DIR / "多角色"
UNTRAINABLE_OTHER_DIR = UNTRAINABLE_DIR / "其他"

for d in [DATA_DIR, ANNOTATIONS_DIR, NSFW_SUSPICIOUS_DIR, UNTRAINABLE_DIR, UNTRAINABLE_R18_DIR, UNTRAINABLE_MULTI_DIR, UNTRAINABLE_OTHER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
            "timestamp": self.timestamp
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

    def to_dict(self):
        return {"id": self.id, "name": self.name, "name_cn": self.name_cn, "category": self.category}

    @staticmethod
    def from_dict(d):
        return Role(d.get("id", ""), d.get("name", ""), d.get("name_cn", ""), d.get("category", ""))

def load_roles():
    if not ROLES_FILE.exists():
        return []
    try:
        with open(ROLES_FILE, 'r', encoding='utf-8') as f:
            return [Role.from_dict(r) for r in json.load(f)]
    except:
        return []

def save_roles(roles):
    with open(ROLES_FILE, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in roles], f, ensure_ascii=False, indent=2)

def load_annotations():
    annotations = {}
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    annotations[data.get('image_path', '')] = AnnotationData.from_dict(data)
            except:
                pass
    return annotations

def save_annotation(annotation: AnnotationData):
    if not annotation.timestamp:
        annotation.timestamp = datetime.now().isoformat()
    safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in annotation.image_path)[:100]
    safe_name = safe_name if safe_name else str(hash(annotation.image_path))
    file_path = ANNOTATIONS_DIR / f"{safe_name}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)

def scan_images(directory, extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
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

class AddRoleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加角色")
        self.setMinimumWidth(350)
        layout = QFormLayout(self)

        self.id_edit = QLineEdit()
        self.name_edit = QLineEdit()
        self.name_cn_edit = QLineEdit()
        self.category_edit = QLineEdit()

        layout.addRow("角色ID (英文):", self.id_edit)
        layout.addRow("角色名 (英文):", self.name_edit)
        layout.addRow("角色名 (中文):", self.name_cn_edit)
        layout.addRow("分类:", self.category_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_role(self):
        return Role(
            self.id_edit.text(),
            self.name_edit.text(),
            self.name_cn_edit.text(),
            self.category_edit.text()
        )

class BatchImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量导入角色")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText('JSON格式，例如:\n[{"id":"anya","name":"Anya","name_cn":"阿尼亚"}]')
        layout.addWidget(QLabel("粘贴JSON数组:"))
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_roles(self):
        try:
            data = json.loads(self.text_edit.toPlainText())
            return [Role.from_dict(r) for r in data]
        except:
            return []

class ZoomableDragableLabel(QLabel):
    clicked = pyqtSignal()
    zoomChanged = pyqtSignal(int)
    gridCellClicked = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_level = 100
        self.min_zoom = 25
        self.max_zoom = 400
        self.pan_offset = [0, 0]
        self.is_panning = False
        self.last_pan_pos = [0, 0]
        self.current_pixmap = None
        self.delete_mode = False
        self.grid_mode = 0
        self.current_grid_index = 0
        self.grid_start_index = 0
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setCursor(Qt.OpenHandCursor)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self._event_count = 0

    def _log_event(self, event_type, event):
        self._event_count += 1
        btn = "Left" if event.button() == Qt.LeftButton else "Right" if event.button() == Qt.RightButton else "Middle" if event.button() == Qt.MiddleButton else str(event.button())
        pos = f"({event.x()}, {event.y()})"
        size = f"label_size={self.size().width()}x{self.size().height()}"
        parent = self.parent()
        parent_size = f"parent_size={parent.size().width()}x{parent.size().height()}" if parent else "no_parent"
        logger.info(f"[Event#{self._event_count}] {event_type}: btn={btn}, pos={pos}, {size}, {parent_size}, is_panning={self.is_panning}, zoom={self.zoom_level}")

    def set_grid_info(self, grid_mode, current_index, grid_start_index):
        self.grid_mode = grid_mode
        self.current_grid_index = current_index
        self.grid_start_index = grid_start_index

    def get_main_window(self):
        parent = self.parent()
        while parent is not None:
            from PyQt5.QtWidgets import QMainWindow
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parent()
        return None

    def show_context_menu(self, pos):
        logger.info(f"[Menu] Context menu requested at pos={pos}, grid_mode={self.grid_mode}")
        if self.grid_mode == 0:
            logger.info("[Menu] grid_mode is 0, returning")
            return

        main_window = self.get_main_window()
        if main_window is None:
            logger.info("[Menu] No main window, returning")
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2a2a3e;
                color: #ffffff;
                border: 1px solid #444;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px 8px 24px;
            }
            QMenu::item:hover {
                background-color: #3a3a5e;
            }
            QMenu::separator {
                background-color: #444;
                height: 1px;
            }
        """)

        grid_configs = {1: 1, 2: 4, 3: 8}
        count = grid_configs.get(self.grid_mode, 1)

        clicked_idx = self.get_clicked_grid_index(pos)
        if clicked_idx is None or clicked_idx >= len(main_window.images):
            logger.info(f"[Menu] Invalid clicked index: {clicked_idx}, total images: {len(main_window.images)}")
            return

        img = main_window.images[clicked_idx]
        filename = img['filename']

        delete_action = QAction(f"删除图片 ({clicked_idx + 1})", self)
        delete_action.triggered.connect(lambda idx=clicked_idx: main_window.delete_image_at_index(idx))
        menu.addAction(delete_action)

        move_r18_action = QAction(f"移动到R18 ({clicked_idx + 1})", self)
        move_r18_action.triggered.connect(lambda idx=clicked_idx: main_window.move_image_at_index(idx, "R18"))
        menu.addAction(move_r18_action)

        move_multi_action = QAction(f"移动到多角色 ({clicked_idx + 1})", self)
        move_multi_action.triggered.connect(lambda idx=clicked_idx: main_window.move_image_at_index(idx, "多角色"))
        menu.addAction(move_multi_action)

        move_other_action = QAction(f"移动到其他 ({clicked_idx + 1})", self)
        move_other_action.triggered.connect(lambda idx=clicked_idx: main_window.move_image_at_index(idx, "其他"))
        menu.addAction(move_other_action)

        menu.addSeparator()

        jump_action = QAction(f"跳转查看 ({clicked_idx + 1})", self)
        jump_action.triggered.connect(lambda idx=clicked_idx: main_window.jump_to_index(idx))
        menu.addAction(jump_action)

        logger.info(f"[Menu] About to exec menu at global pos={self.mapToGlobal(pos)}")
        menu.exec_(self.mapToGlobal(pos))
        logger.info("[Menu] Menu closed")

    def get_clicked_grid_index(self, pos):
        if self.grid_mode == 0:
            return self.current_grid_index

        grid_configs = {1: (2, 2), 2: (2, 4), 3: (4, 4)}
        rows, cols = grid_configs.get(self.grid_mode, (2, 2))

        cell_width = self.width() // cols if cols > 0 else self.width()
        cell_height = self.height() // rows if rows > 0 else self.height()

        col = pos.x() // cell_width
        row = pos.y() // cell_height

        if row >= rows or col >= cols:
            return None

        cell_index = row * cols + col
        clicked_idx = self.grid_start_index + cell_index

        main_window = self.get_main_window()
        if main_window is None or clicked_idx >= len(main_window.images):
            return None

        return clicked_idx

    def setPixmap(self, pixmap):
        self.current_pixmap = pixmap
        super().setPixmap(pixmap)
        self.reset_pan()

    def reset_pan(self):
        self.pan_offset = [0, 0]
        self.update()

    def wheelEvent(self, event):
        self._event_count += 1
        delta = event.angleDelta().y()
        logger.info(f"[Event#{self._event_count}] wheelEvent: delta={delta}, has_pixmap={self.current_pixmap is not None}, zoom_before={self.zoom_level}")
        
        if not self.current_pixmap:
            logger.info("[Event] wheelEvent - no pixmap, calling super")
            super().wheelEvent(event)
            return

        if delta > 0:
            self.zoom_level = min(self.max_zoom, self.zoom_level + 25)
            logger.info(f"[Event] wheelEvent - zoom_in to {self.zoom_level}")
        elif delta < 0:
            self.zoom_level = max(self.min_zoom, self.zoom_level - 25)
            logger.info(f"[Event] wheelEvent - zoom_out to {self.zoom_level}")
        else:
            logger.info("[Event] wheelEvent - delta=0, ignoring")
            event.accept()
            return

        logger.info(f"[Event] wheelEvent - zoom_after={self.zoom_level}")
        self.zoomChanged.emit(self.zoom_level)
        self.update_pixmap_size()
        event.accept()
        logger.info("[Event] wheelEvent - accepted")

    def update_pixmap_size(self):
        if not self.current_pixmap:
            return

        scaled_w = int(self.current_pixmap.width() * self.zoom_level / 100)
        scaled_h = int(self.current_pixmap.height() * self.zoom_level / 100)

        if scaled_w > 0 and scaled_h > 0:
            scaled_pixmap = self.current_pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        self._log_event("mousePressEvent", event)
        if event.button() == Qt.LeftButton:
            main_window = self.get_main_window()
            if self.delete_mode:
                if self.grid_mode > 0 and main_window:
                    clicked_idx = self.get_clicked_grid_index(event.pos())
                    if clicked_idx is not None:
                        main_window.delete_image_at_index(clicked_idx)
                else:
                    self.clicked.emit()
            else:
                if self.grid_mode > 0 and main_window:
                    clicked_idx = self.get_clicked_grid_index(event.pos())
                    if clicked_idx is not None:
                        main_window.current_index = clicked_idx
                        main_window.show_current_image()
                else:
                    self.is_panning = True
                    self.last_pan_pos = [event.x(), event.y()]
                    self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            event.accept()
            logger.info("[Event] Right button pressed - accepted and returning")
            return
        else:
            logger.info(f"[Event] Other button pressed: {event.button()}")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._log_event("mouseMoveEvent", event)
        if self.is_panning and self.current_pixmap:
            dx = event.x() - self.last_pan_pos[0]
            dy = event.y() - self.last_pan_pos[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.last_pan_pos = [event.x(), event.y()]
            self.update()
            event.accept()
            logger.info(f"[Event] Panning - dx={dx}, dy={dy}, pan_offset={self.pan_offset}")
            return
        logger.info("[Event] mouseMoveEvent - not panning, calling super")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._log_event("mouseReleaseEvent", event)
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)
            logger.info("[Event] Left button released - panning stopped")
        elif event.button() == Qt.RightButton:
            event.accept()
            logger.info("[Event] Right button released - accepted and returning")
            return
        else:
            logger.info(f"[Event] Other button released: {event.button()}")
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        if self.current_pixmap:
            qp = QPainter(self)
            if self.zoom_level != 100:
                scaled_w = int(self.current_pixmap.width() * self.zoom_level / 100)
                scaled_h = int(self.current_pixmap.height() * self.zoom_level / 100)
                scaled_pixmap = self.current_pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                x = (self.width() - scaled_pixmap.width()) // 2 + self.pan_offset[0]
                y = (self.height() - scaled_pixmap.height()) // 2 + self.pan_offset[1]
                qp.drawPixmap(x, y, scaled_pixmap)
            else:
                x = (self.width() - self.current_pixmap.width()) // 2 + self.pan_offset[0]
                y = (self.height() - self.current_pixmap.height()) // 2 + self.pan_offset[1]
                qp.drawPixmap(x, y, self.current_pixmap)
        else:
            super().paintEvent(event)

    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)

ClickableLabel = ZoomableDragableLabel

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = []
        self.current_index = 0
        self.roles = []
        self.annotations = {}
        self.selected_role_ids = []
        self.current_image_label = None
        self.grid_mode = 0
        self.zoom_level = 100
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_delay = 2000
        self.undo_stack = []
        self.max_undo = 20
        self.delete_mode = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🎬 动漫角色标注工具")
        self.setGeometry(100, 100, 1400, 900)

        self.create_menu_bar()
        self.create_tool_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)

        left_panel = self.create_left_panel()
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(350)
        main_layout.addWidget(left_panel)

        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.load_roles()
        self.load_annotations()

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")

        open_dir_action = QAction("打开目录", self)
        open_dir_action.setShortcut("Ctrl+O")
        open_dir_action.triggered.connect(self.browse_directory)
        file_menu.addAction(open_dir_action)

        export_json_action = QAction("导出JSON", self)
        export_json_action.setShortcut("Ctrl+E")
        export_json_action.triggered.connect(self.export_json)
        file_menu.addAction(export_json_action)

        export_csv_action = QAction("导出CSV", self)
        export_csv_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        roles_menu = menubar.addMenu("角色")

        add_role_action = QAction("添加角色", self)
        add_role_action.triggered.connect(self.add_role_dialog)
        roles_menu.addAction(add_role_action)

        import_roles_action = QAction("批量导入", self)
        import_roles_action.triggered.connect(self.batch_import_dialog)
        roles_menu.addAction(import_roles_action)

        help_menu = menubar.addMenu("帮助")

        shortcuts_action = QAction("快捷键", self)
        shortcuts_action.setShortcut("H")
        shortcuts_action.triggered.connect(self.show_shortcuts_help)
        help_menu.addAction(shortcuts_action)

        undo_action = QAction("撤销", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        help_menu.addAction(undo_action)

        help_menu.addSeparator()

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.dir_label = QLabel("未选择目录")
        toolbar.addWidget(self.dir_label)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("标注者:"))
        self.annotator_edit = QLineEdit("anonymous")
        self.annotator_edit.setMaximumWidth(150)
        toolbar.addWidget(self.annotator_edit)

        toolbar.addSeparator()

        stats_label = QLabel("进度:")
        toolbar.addWidget(stats_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setMaximum(100)
        toolbar.addWidget(self.progress_bar)

        self.stats_label = QLabel("0 / 0")
        toolbar.addWidget(self.stats_label)

    def create_left_panel(self):
        splitter = QSplitter(Qt.Vertical)

        group_dir = QGroupBox("图片目录")
        dir_layout = QVBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setText(str(Path.home() / "PycharmProjects" / "anime_role_detect" / "data" / "combined_dataset"))
        dir_layout.addWidget(self.dir_edit)

        btn_layout = QHBoxLayout()
        scan_btn = QPushButton("🔍 扫描")
        scan_btn.clicked.connect(self.scan_directory)
        open_btn = QPushButton("📁 浏览")
        open_btn.clicked.connect(self.browse_directory)
        btn_layout.addWidget(scan_btn)
        btn_layout.addWidget(open_btn)
        dir_layout.addLayout(btn_layout)
        group_dir.setLayout(dir_layout)
        splitter.addWidget(group_dir)

        group_stats = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()
        self.total_label = QLabel("总图片: 0")
        self.annotated_label = QLabel("已标注: 0")
        self.nsfw_label = QLabel("R18: 0")
        self.multi_role_label = QLabel("多角色: 0")
        self.untrainable_label = QLabel("无法训练: 0")
        stats_layout.addWidget(self.total_label)
        stats_layout.addWidget(self.annotated_label)
        stats_layout.addWidget(self.nsfw_label)
        stats_layout.addWidget(self.multi_role_label)
        stats_layout.addWidget(self.untrainable_label)
        group_stats.setLayout(stats_layout)
        splitter.addWidget(group_stats)

        group_folders = QGroupBox("目录信息")
        folders_layout = QVBoxLayout()
        self.ann_folder_label = QLabel("标注文件夹: 0个文件")
        self.r18_folder_label = QLabel("R18目录: 0个文件")
        self.multi_folder_label = QLabel("多角色目录: 0个文件")
        self.other_folder_label = QLabel("其他目录: 0个文件")
        folders_layout.addWidget(self.ann_folder_label)
        folders_layout.addWidget(self.r18_folder_label)
        folders_layout.addWidget(self.multi_folder_label)
        folders_layout.addWidget(self.other_folder_label)
        group_folders.setLayout(folders_layout)
        splitter.addWidget(group_folders)

        group_roles = QGroupBox("角色列表")
        roles_layout = QVBoxLayout()

        add_role_btn = QPushButton("➕ 添加角色")
        add_role_btn.clicked.connect(self.add_role_dialog)
        roles_layout.addWidget(add_role_btn)

        self.role_list = QListWidget()
        self.role_list.setAlternatingRowColors(True)
        self.role_list.itemClicked.connect(self.on_role_clicked)
        roles_layout.addWidget(self.role_list)

        group_roles.setLayout(roles_layout)
        splitter.addWidget(group_roles)

        splitter.setSizes([100, 80, 400])

        scroll = QScrollArea()
        scroll.setWidget(splitter)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(350)
        return scroll

    def create_right_panel(self):
        logger.info("[Layout] Creating right panel")
        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("RightPanelSplitter")
        splitter.splitterMoved.connect(lambda pos, idx: logger.info(f"[Layout] Splitter moved: pos={pos}, idx={idx}, sizes={splitter.sizes()}"))

        group_image = QGroupBox("图片预览")
        image_layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        self.image_name_label = QLabel("未加载图片")
        self.image_name_label.setStyleSheet("color: #888; font-size: 12px;")
        self.image_name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_layout.addWidget(self.image_name_label)

        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet("color: #0af; font-size: 11px;")
        self.image_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.image_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header_layout.addWidget(self.image_info_label)
        image_layout.addLayout(header_layout)

        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.clicked.connect(self.on_image_clicked)
        self.image_label.setMinimumHeight(500)
        self.image_label.setStyleSheet("background-color: #1a1a2e; border: 1px solid #333;")
        self.image_label.setText("\n\n\n\n 请先扫描目录加载图片 \n\n\n")

        image_layout.addWidget(self.image_label)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<< 上一个")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn = QPushButton("下一个 >>")
        self.next_btn.clicked.connect(self.next_image)
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("序号")
        self.jump_edit.setMaximumWidth(60)
        self.jump_edit.returnPressed.connect(self.jump_to_image)
        self.jump_btn = QPushButton("跳转")
        self.jump_btn.clicked.connect(self.jump_to_image)
        self.unannotated_btn = QPushButton("下一未标")
        self.unannotated_btn.clicked.connect(self.jump_to_unannotated)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.jump_edit)
        nav_layout.addWidget(self.jump_btn)
        nav_layout.addWidget(self.unannotated_btn)
        nav_layout.addStretch()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("预览模式:"))
        self.grid_mode_combo = QComboBox()
        self.grid_mode_combo.addItems(["单图", "4宫格", "8宫格", "16宫格"])
        self.grid_mode_combo.currentIndexChanged.connect(self.on_grid_mode_changed)
        mode_layout.addWidget(self.grid_mode_combo)

        self.delete_mode_check = QCheckBox("删除模式")
        self.delete_mode_check.stateChanged.connect(self.on_delete_mode_changed)
        mode_layout.addWidget(self.delete_mode_check)

        mode_layout.addStretch()

        zoom_label = QLabel("缩放:")
        mode_layout.addWidget(zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        mode_layout.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel("100%")
        mode_layout.addWidget(self.zoom_value_label)

        reset_zoom_btn = QPushButton("重置")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        mode_layout.addWidget(reset_zoom_btn)

        image_layout.addLayout(mode_layout)
        image_layout.addLayout(nav_layout)

        group_image.setLayout(image_layout)
        splitter.addWidget(group_image)

        group_annotation = QGroupBox("标注信息")
        annotation_layout = QFormLayout()

        self.selected_roles_label = QLabel("未选择")
        annotation_layout.addRow("已选角色:", self.selected_roles_label)

        self.multi_role_check = QCheckBox("多角色图片")
        self.multi_role_check.stateChanged.connect(self.on_annotation_changed)
        annotation_layout.addRow("", self.multi_role_check)

        self.nsfw_check = QCheckBox("R18内容")
        self.nsfw_check.stateChanged.connect(self.on_annotation_changed)
        annotation_layout.addRow("", self.nsfw_check)

        self.notes_edit = QLineEdit()
        annotation_layout.addRow("备注:", self.notes_edit)
        self.notes_edit.textChanged.connect(self.on_annotation_changed)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("[保存]")
        save_btn.clicked.connect(self.save_annotation)
        clear_btn = QPushButton("[清除]")
        clear_btn.clicked.connect(self.clear_selection)
        delete_ann_btn = QPushButton("[删除标注]")
        delete_ann_btn.clicked.connect(self.delete_annotation)
        ai_infer_btn = QPushButton("[AI识别]")
        ai_infer_btn.clicked.connect(self.infer_role_with_ai)

        btn_row.addWidget(save_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(delete_ann_btn)
        btn_row.addWidget(ai_infer_btn)
        btn_row.addStretch()
        annotation_layout.addRow("", btn_row)

        move_label = QLabel("移动到无法训练:")
        move_label.setStyleSheet("color: #f00; font-weight: bold;")
        annotation_layout.addRow("", move_label)

        move_row = QHBoxLayout()
        move_r18_btn = QPushButton("[R18]")
        move_r18_btn.setStyleSheet("background-color: #ffcccc;")
        move_r18_btn.clicked.connect(lambda: self.move_to_untrainable("R18"))
        move_multi_btn = QPushButton("[多角色]")
        move_multi_btn.setStyleSheet("background-color: #ccffcc;")
        move_multi_btn.clicked.connect(lambda: self.move_to_untrainable("多角色"))
        move_other_btn = QPushButton("[其他]")
        move_other_btn.setStyleSheet("background-color: #ccccff;")
        move_other_btn.clicked.connect(lambda: self.move_to_untrainable("其他"))

        move_row.addWidget(move_r18_btn)
        move_row.addWidget(move_multi_btn)
        move_row.addWidget(move_other_btn)
        move_row.addStretch()
        annotation_layout.addRow("", move_row)

        group_annotation.setLayout(annotation_layout)
        splitter.addWidget(group_annotation)

        group_log = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(80)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1a1a2e; color: #0f0; font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self.log_text)
        group_log.setLayout(log_layout)
        splitter.addWidget(group_log)

        splitter.setSizes([550, 200, 80])
        return splitter

    def load_roles(self):
        self.roles = load_roles()
        self.update_role_list()

    def load_annotations(self):
        self.annotations = load_annotations()
        self.update_stats()

    def update_role_list(self):
        self.role_list.clear()
        for role in self.roles:
            item = QListWidgetItem(f"{role.name} ({role.name_cn})" if role.name_cn else role.name)
            item.setData(Qt.UserRole, role.id)
            self.role_list.addItem(item)

    def update_stats(self):
        total = len(self.images)
        annotated = sum(1 for img in self.images if img['path'] in self.annotations)
        nsfw_count = sum(1 for ann in self.annotations.values() if ann.is_nsfw)
        multi_count = sum(1 for ann in self.annotations.values() if ann.is_multi_role)

        untrainable_count = (
            len(list(UNTRAINABLE_R18_DIR.glob("*"))) +
            len(list(UNTRAINABLE_MULTI_DIR.glob("*"))) +
            len(list(UNTRAINABLE_OTHER_DIR.glob("*")))
        )

        self.total_label.setText(f"总图片: {total}")
        self.annotated_label.setText(f"已标注: {annotated}")
        self.nsfw_label.setText(f"R18: {nsfw_count}")
        self.multi_role_label.setText(f"多角色: {multi_count}")
        self.untrainable_label.setText(f"无法训练: {untrainable_count}")

        ann_files = len(list(ANNOTATIONS_DIR.glob("*.json")))
        r18_files = len(list(UNTRAINABLE_R18_DIR.glob("*")))
        multi_files = len(list(UNTRAINABLE_MULTI_DIR.glob("*")))
        other_files = len(list(UNTRAINABLE_OTHER_DIR.glob("*")))

        self.ann_folder_label.setText(f"标注文件夹: {ann_files}个文件")
        self.r18_folder_label.setText(f"R18目录: {r18_files}个文件")
        self.multi_folder_label.setText(f"多角色目录: {multi_files}个文件")
        self.other_folder_label.setText(f"其他目录: {other_files}个文件")

        if total > 0:
            progress = int(annotated / total * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)

        self.stats_label.setText(f"{annotated} / {total}")

    def scan_directory(self):
        directory = self.dir_edit.text().strip()
        if not directory:
            QMessageBox.warning(self, "警告", "请输入目录路径")
            return

        old_count = len(self.images)
        self.images = scan_images(directory)
        self.dir_label.setText(f"目录: {Path(directory).name}")
        self.status_bar.showMessage(f"加载了 {len(self.images)} 张图片")

        if old_count == 0:
            self.current_index = 0
            self.reset_zoom()
        else:
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)

        self.show_current_image()
        self.update_stats()

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if directory:
            self.dir_edit.setText(directory)
            self.scan_directory()

    def show_current_image(self):
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2] if len(stack) > 1 else "unknown"
        logger.info(f"[Image] Showing image {self.current_index}/{len(self.images)}, grid_mode={self.grid_mode}, called from {caller.filename}:{caller.lineno} {caller.name}")
        if not self.images:
            self.image_label.setText("\n\n\n\n 请先扫描目录加载图片 \n\n\n")
            self.image_info_label.setText("")
            return

        if self.current_index < 0 or self.current_index >= len(self.images):
            return

        img = self.images[self.current_index]
        img_path = Path(img['path'])

        filename = img_path.name
        folder_name = img_path.parent.name

        role_info = ""
        if img['path'] in self.annotations:
            ann = self.annotations[img['path']]
            if ann.roles:
                role_names = []
                for role_id in ann.roles:
                    for role in self.roles:
                        if role.id == role_id:
                            role_names.append(role.name)
                            break
                role_info = " | 角色: " + ", ".join(role_names)
            else:
                role_info = " | 未标注"
        else:
            role_info = " | 未标注"

        self.image_info_label.setText(f"{filename}{role_info}")

        if self.grid_mode == 0:
            self.show_single_image()
        else:
            self.show_grid_images()

        self.load_current_annotation()

    def show_single_image(self):
        img = self.images[self.current_index]
        pixmap = QPixmap(img['path'])

        if pixmap.isNull():
            self.image_label.setText(f"\n\n\n\n 无法加载图片: {img['filename']} \n\n\n")
            return

        base_width = self.image_label.width()
        base_height = self.image_label.height()
        scale_factor = self.zoom_level / 100.0

        new_width = int(base_width * scale_factor)
        new_height = int(base_height * scale_factor)

        if new_width > 0 and new_height > 0:
            if pixmap.width() > new_width or pixmap.height() > new_height:
                scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaled_pixmap = pixmap.scaled(pixmap.width(), pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            scaled_pixmap = pixmap.scaled(base_width, base_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

        zoom_str = f" ({self.zoom_level}%)" if self.zoom_level != 100 else ""
        self.image_name_label.setText(f"{img['filename']} [{self.current_index + 1}/{len(self.images)}]{zoom_str}")
        self.jump_edit.setText(str(self.current_index + 1))

        self.status_bar.showMessage(f"[{self.current_index + 1}/{len(self.images)}] {img['filename']}{zoom_str}")

    def show_grid_images(self):
        grid_configs = {1: (2, 2), 2: (2, 4), 3: (4, 4)}
        rows, cols = grid_configs.get(self.grid_mode, (2, 2))
        count = rows * cols

        start_idx = self.current_index - (self.current_index % count)
        end_idx = min(start_idx + count, len(self.images))
        
        logger.info(f"[Grid] current_index={self.current_index}, count={count}, start_idx={start_idx}, end_idx={end_idx}, total={len(self.images)}")

        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        max_width = 1200
        max_height = 800
        label_width = min(label_width, max_width)
        label_height = min(label_height, max_height)

        cell_width = label_width // cols if cols > 0 else label_width
        cell_height = label_height // rows if rows > 0 else label_height

        grid_pixmap = QPixmap(label_width, label_height)
        grid_pixmap.fill(Qt.darkGray)
        painter = QPainter(grid_pixmap)
        painter.setPen(QPen(Qt.white, 1))

        for i, idx in enumerate(range(start_idx, end_idx)):
            if idx >= len(self.images):
                break

            img_data = self.images[idx]
            pixmap = QPixmap(img_data['path'])

            if pixmap.isNull():
                continue

            row = i // cols
            col = i % cols

            scaled = pixmap.scaled(cell_width - 4, cell_height - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = col * cell_width + (cell_width - scaled.width()) // 2
            y = row * cell_height + (cell_height - scaled.height()) // 2

            painter.drawPixmap(x, y, scaled)

            border_color = Qt.white if idx == self.current_index else Qt.cyan
            painter.setPen(QPen(border_color, 2))
            painter.drawRect(col * cell_width + 1, row * cell_height + 1, cell_width - 2, cell_height - 2)
            painter.setPen(QPen(Qt.white, 1))

            painter.drawText(col * cell_width + 4, row * cell_height + 14, f"{idx + 1}")

        painter.end()
        self.image_label.setPixmap(grid_pixmap)
        self.image_label.set_grid_info(self.grid_mode, self.current_index, start_idx)

        total_grids = (len(self.images) + count - 1) // count
        current_grid = start_idx // count + 1
        self.image_name_label.setText(f"宫格预览 [{current_grid}/{total_grids}] {start_idx + 1}-{end_idx}/{len(self.images)}")
        self.jump_edit.setText(str(self.current_index + 1))
        self.status_bar.showMessage(f"宫格模式: 第{current_grid}页，共{total_grids}页")

    def load_current_annotation(self):
        if not self.images:
            return

        img = self.images[self.current_index]
        ann = self.annotations.get(img['path'])

        self.selected_role_ids = []
        if ann:
            self.selected_role_ids = list(ann.roles)
            self.multi_role_check.setChecked(ann.is_multi_role)
            self.nsfw_check.setChecked(ann.is_nsfw)
            self.notes_edit.setText(ann.notes)
        else:
            self.multi_role_check.setChecked(False)
            self.nsfw_check.setChecked(False)
            self.notes_edit.setText("")

        self.update_selected_roles_display()

    def update_selected_roles_display(self):
        if not self.selected_role_ids:
            self.selected_roles_label.setText("未选择")
        else:
            names = []
            for role_id in self.selected_role_ids:
                role = next((r for r in self.roles if r.id == role_id), None)
                if role:
                    names.append(f"{role.name}({role.name_cn})" if role.name_cn else role.name)
                else:
                    names.append(role_id)
            self.selected_roles_label.setText(", ".join(names))

    def on_role_clicked(self, item):
        role_id = item.data(Qt.UserRole)
        if role_id in self.selected_role_ids:
            self.selected_role_ids.remove(role_id)
            item.setBackground(Qt.white)
        else:
            self.selected_role_ids.append(role_id)
            item.setBackground(Qt.green)

        self.update_selected_roles_display()

    def on_annotation_changed(self):
        self.auto_save_timer.stop()
        self.auto_save_timer.start(self.auto_save_delay)

    def on_grid_mode_changed(self, index):
        modes = ["单图", "4宫格", "8宫格", "16宫格"]
        mode = modes[index] if index < len(modes) else "单图"
        self.grid_mode = index
        self.log_message(f"切换到{mode}预览模式")
        self.show_current_image()

    def on_delete_mode_changed(self, state):
        self.delete_mode = bool(state)
        if hasattr(self.image_label, 'delete_mode'):
            self.image_label.delete_mode = self.delete_mode
        if state:
            self.log_message("删除模式已开启 - 点击图片可删除")
            self.image_label.setStyleSheet("background-color: #2a1a1e; border: 2px solid #f00;")
        else:
            self.log_message("删除模式已关闭")
            self.image_label.setStyleSheet("background-color: #1a1a2e; border: 1px solid #333;")

    def on_zoom_changed(self, value):
        self.zoom_level = value
        self.zoom_value_label.setText(f"{value}%")
        if hasattr(self.image_label, 'zoom_level'):
            self.image_label.zoom_level = value
            self.image_label.update_pixmap_size()

    def reset_zoom(self):
        self.zoom_level = 100
        self.zoom_slider.setValue(100)
        if hasattr(self.image_label, 'zoom_level'):
            self.image_label.zoom_level = 100
            self.image_label.reset_pan()
            self.image_label.update_pixmap_size()

    def log_message(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logger.info(f"{msg}")

    def on_image_clicked(self):
        if self.delete_mode_check.isChecked() and self.images:
            img = self.images[self.current_index]
            reply = QMessageBox.question(self, "确认删除", f"确定要删除图片吗？\n{img['filename']}",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    Path(img['path']).unlink()
                    self.log_message(f"已删除: {img['filename']}")
                    if hasattr(self.image_label, 'current_pixmap'):
                        self.image_label.current_pixmap = None
                    self.image_label.clear()
                    self.image_label.setText("\n\n\n\n 图片已删除 \n\n\n")
                    QApplication.processEvents()
                    self.scan_directory()
                    if self.current_index >= len(self.images):
                        self.current_index = max(0, len(self.images) - 1)
                    self.show_current_image()
                except Exception as e:
                    self.log_message(f"删除失败: {str(e)}")
                    QMessageBox.warning(self, "错误", f"删除失败:\n{str(e)}")

    def delete_image_at_index(self, idx):
        if idx < 0 or idx >= len(self.images):
            return
        img = self.images[idx]
        reply = QMessageBox.question(self, "确认删除", f"确定要删除图片吗？\n{img['filename']}",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                Path(img['path']).unlink()
                self.log_message(f"已删除: {img['filename']}")
                if hasattr(self.image_label, 'current_pixmap'):
                    self.image_label.current_pixmap = None
                self.image_label.clear()
                self.image_label.setText("\n\n\n\n 图片已删除 \n\n\n")
                QApplication.processEvents()
                self.scan_directory()
                if self.current_index >= len(self.images):
                    self.current_index = max(0, len(self.images) - 1)
                self.show_current_image()
            except Exception as e:
                self.log_message(f"删除失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"删除失败:\n{str(e)}")

    def move_image_at_index(self, idx, category):
        if idx < 0 or idx >= len(self.images):
            return
        img = self.images[idx]
        source = Path(img['path'])

        if not source.exists():
            QMessageBox.warning(self, "警告", "源文件不存在")
            return

        if category == "R18":
            dest = UNTRAINABLE_R18_DIR / source.name
        elif category == "多角色":
            dest = UNTRAINABLE_MULTI_DIR / source.name
        else:
            dest = UNTRAINABLE_OTHER_DIR / source.name

        import shutil
        try:
            shutil.move(str(source), str(dest))
            self.log_message(f"已移动到无法训练/{category}: {img['filename']}")
            if hasattr(self.image_label, 'current_pixmap'):
                self.image_label.current_pixmap = None
            self.image_label.clear()
            self.image_label.setText("\n\n\n\n 图片已移动 \n\n\n")
            QApplication.processEvents()
            self.scan_directory()
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)
            self.show_current_image()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"移动失败:\n{str(e)}")

    def jump_to_index(self, idx):
        if 0 <= idx < len(self.images):
            self.current_index = idx
            self.show_current_image()
            self.log_message(f"跳转到第 {self.current_index + 1} 张图片")

    def prev_image(self):
        if not self.images:
            return
        if self.grid_mode > 0:
            grid_configs = {1: 4, 2: 8, 3: 16}
            count = grid_configs.get(self.grid_mode, 4)
            current_group_start = self.current_index - (self.current_index % count)
            prev_group_start = max(0, current_group_start - count)
            self.current_index = prev_group_start
        else:
            if self.current_index > 0:
                self.current_index -= 1
        self.show_current_image()
        self.log_message(f"切换到第 {self.current_index + 1} 张图片")

    def next_image(self):
        if not self.images:
            return
        if self.grid_mode > 0:
            grid_configs = {1: 4, 2: 8, 3: 16}
            count = grid_configs.get(self.grid_mode, 4)
            current_group_start = self.current_index - (self.current_index % count)
            next_group_start = current_group_start + count
            if next_group_start < len(self.images):
                self.current_index = next_group_start
            else:
                self.current_index = current_group_start
        else:
            if self.current_index < len(self.images) - 1:
                self.current_index += 1
        self.show_current_image()
        self.log_message(f"切换到第 {self.current_index + 1} 张图片")

    def jump_to_image(self):
        try:
            idx = int(self.jump_edit.text()) - 1
            if 0 <= idx < len(self.images):
                self.current_index = idx
                self.show_current_image()
                self.log_message(f"跳转到第 {self.current_index + 1} 张图片")
        except:
            pass

    def jump_to_unannotated(self):
        for i, img in enumerate(self.images):
            if img['path'] not in self.annotations:
                self.current_index = i
                self.show_current_image()
                self.status_bar.showMessage("已跳转到未标注图片")
                return
        QMessageBox.information(self, "提示", "所有图片都已标注！")

    def save_annotation(self, auto=False):
        if not self.images:
            return

        img = self.images[self.current_index]

        old_ann = self.annotations.get(img['path'])
        if old_ann:
            old_state = {
                'image_path': img['path'],
                'roles': list(old_ann.roles),
                'is_multi_role': old_ann.is_multi_role,
                'is_nsfw': old_ann.is_nsfw,
                'notes': old_ann.notes
            }
        else:
            old_state = {
                'image_path': img['path'],
                'roles': [],
                'is_multi_role': False,
                'is_nsfw': False,
                'notes': ''
            }

        if len(self.undo_stack) == 0 or self.undo_stack[-1] != old_state:
            self.undo_stack.append(old_state)
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

        ann = AnnotationData()
        ann.image_path = img['path']
        ann.roles = list(self.selected_role_ids)
        ann.is_multi_role = self.multi_role_check.isChecked()
        ann.is_nsfw = self.nsfw_check.isChecked()
        ann.notes = self.notes_edit.text()
        ann.annotator = self.annotator_edit.text() or "anonymous"
        ann.timestamp = datetime.now().isoformat()

        save_annotation(ann)
        self.annotations[img['path']] = ann
        self.update_stats()

        if not auto:
            self.log_message(f"已保存标注: {img['filename']}")
            self.status_bar.showMessage(f"已保存标注: {img['filename']}")

    def auto_save(self):
        if self.annotations:
            self.save_annotation(auto=True)

    def undo(self):
        if not self.undo_stack or not self.images:
            self.log_message("没有可撤销的操作")
            return

        state = self.undo_stack.pop()
        img_path = state['image_path']

        img_idx = next((i for i, img in enumerate(self.images) if img['path'] == img_path), -1)
        if img_idx == -1:
            self.log_message("无法撤销：图片不存在")
            return

        if img_path in self.annotations:
            del self.annotations[img_path]

        if state['roles'] or state['is_multi_role'] or state['is_nsfw'] or state['notes']:
            ann = AnnotationData()
            ann.image_path = img_path
            ann.roles = state['roles']
            ann.is_multi_role = state['is_multi_role']
            ann.is_nsfw = state['is_nsfw']
            ann.notes = state['notes']
            save_annotation(ann)
            self.annotations[img_path] = ann

        self.log_message(f"已撤销: {Path(img_path).name}")
        self.update_stats()

    def clear_selection(self):
        self.selected_role_ids = []
        self.multi_role_check.setChecked(False)
        self.nsfw_check.setChecked(False)
        self.notes_edit.setText("")
        self.update_selected_roles_display()
        for i in range(self.role_list.count()):
            self.role_list.item(i).setBackground(Qt.white)
        self.log_message("已清除标注")

    def move_nsfw(self):
        if not self.images:
            return

        img = self.images[self.current_index]
        source = Path(img['path'])

        if not source.exists():
            QMessageBox.warning(self, "警告", "源文件不存在")
            return

        dest = NSFW_SUSPICIOUS_DIR / source.name
        import shutil
        try:
            shutil.move(str(source), str(dest))
            QMessageBox.information(self, "提示", f"图片已移动到:\n{dest}")
            self.save_annotation()
            self.scan_directory()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"移动失败:\n{str(e)}")

    def delete_annotation(self):
        if not self.images:
            return

        img = self.images[self.current_index]
        if img['path'] in self.annotations:
            del self.annotations[img['path']]
            annotation_file = ANNOTATIONS_DIR / f"{img['path'].name}.json"
            if annotation_file.exists():
                annotation_file.unlink()
            self.log_message(f"已删除标注: {img['filename']}")
            self.update_stats()
            self.status_bar.showMessage("标注已删除")

    def move_to_untrainable(self, category):
        if not self.images:
            return

        img = self.images[self.current_index]
        source = Path(img['path'])

        if not source.exists():
            QMessageBox.warning(self, "警告", "源文件不存在")
            return

        if category == "R18":
            dest = UNTRAINABLE_R18_DIR / source.name
        elif category == "多角色":
            dest = UNTRAINABLE_MULTI_DIR / source.name
        else:
            dest = UNTRAINABLE_OTHER_DIR / source.name

        import shutil
        try:
            shutil.move(str(source), str(dest))
            self.log_message(f"已移动到无法训练/{category}: {img['filename']}")
            if hasattr(self.image_label, 'current_pixmap'):
                self.image_label.current_pixmap = None
            self.image_label.clear()
            self.image_label.setText("\n\n\n\n 图片已移动 \n\n\n")
            QApplication.processEvents()
            self.scan_directory()
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)
            self.show_current_image()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"移动失败:\n{str(e)}")

    def infer_role_with_ai(self):
        if not self.images:
            return

        img = self.images[self.current_index]
        self.log_message(f"正在调用AI识别: {img['filename']}...")

        try:
            import requests
            files = {'image': open(img['path'], 'rb')}
            response = requests.post('http://localhost:8001/predict', files=files, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get('success') and result.get('role'):
                role_name = result['role']
                role_ids = [r.id for r in self.roles if r.name == role_name or r.name_cn == role_name]

                if role_ids:
                    self.selected_role_ids = role_ids[:1]
                    self.update_selected_roles_display()
                    self.log_message(f"AI识别结果: {role_name}")
                    self.status_bar.showMessage(f"AI识别: {role_name}")
                else:
                    self.log_message(f"AI识别到角色但未在列表中找到: {role_name}")
                    QMessageBox.information(self, "AI识别", f"识别到角色: {role_name}\n\n请在角色列表中添加该角色后再选择")
            else:
                self.log_message("AI识别未返回有效结果")
        except Exception as e:
            self.log_message(f"AI识别失败: {str(e)}")
            QMessageBox.warning(self, "AI识别", f"识别失败:\n{str(e)}")

    def add_role_dialog(self):
        dialog = AddRoleDialog(self)
        if dialog.exec_():
            role = dialog.get_role()
            if not role.id or not role.name:
                QMessageBox.warning(self, "警告", "请填写角色ID和名称")
                return
            if any(r.id == role.id for r in self.roles):
                QMessageBox.warning(self, "警告", "角色ID已存在")
                return
            self.roles.append(role)
            save_roles(self.roles)
            self.update_role_list()
            self.status_bar.showMessage("✅ 角色已添加")

    def batch_import_dialog(self):
        dialog = BatchImportDialog(self)
        if dialog.exec_():
            new_roles = dialog.get_roles()
            if not new_roles:
                QMessageBox.warning(self, "警告", "JSON格式错误")
                return
            existing_ids = {r.id for r in self.roles}
            added = 0
            for role in new_roles:
                if role.id not in existing_ids:
                    self.roles.append(role)
                    added += 1
            save_roles(self.roles)
            self.update_role_list()
            QMessageBox.information(self, "提示", f"已导入 {added} 个角色")

    def export_json(self):
        if not self.annotations:
            QMessageBox.information(self, "提示", "没有标注数据可导出")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出JSON", "annotations.json", "JSON Files (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([ann.to_dict() for ann in self.annotations.values()], f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "提示", f"已导出到:\n{path}")

    def export_csv(self):
        if not self.annotations:
            QMessageBox.information(self, "提示", "没有标注数据可导出")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "annotations.csv", "CSV Files (*.csv)")
        if path:
            lines = ["image_path,roles,is_multi_role,is_nsfw,notes,annotator,timestamp"]
            for ann in self.annotations.values():
                roles_str = "|".join(ann.roles)
                lines.append(f'"{ann.image_path}","{roles_str}",{ann.is_multi_role},{ann.is_nsfw},"{ann.notes}","{ann.annotator}","{ann.timestamp}"')
            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            QMessageBox.information(self, "提示", f"已导出到:\n{path}")

    def show_about(self):
        QMessageBox.about(self, "关于",
            "[Anime Role Annotator] v1.0\n\n"
            "一个简单易用的图片标注工具\n"
            "支持批量标注、多角色识别、R18检测\n\n"
            "快捷键:\n"
            "< > 切换图片\n"
            "Ctrl+O 打开目录\n"
            "Ctrl+E 导出\n"
            "Ctrl+Q 退出"
        )

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key_Left:
            self.prev_image()
        elif key == Qt.Key_Right:
            self.next_image()
        elif key == Qt.Key_Up:
            if self.grid_mode > 0:
                self.grid_mode = max(0, self.grid_mode - 1)
                self.grid_mode_combo.setCurrentIndex(self.grid_mode)
            else:
                self.zoom_level = min(200, self.zoom_level + 25)
                self.show_current_image()
        elif key == Qt.Key_Down:
            if self.grid_mode < 3:
                self.grid_mode += 1
                self.grid_mode_combo.setCurrentIndex(self.grid_mode)
            else:
                self.zoom_level = max(25, self.zoom_level - 25)
                self.show_current_image()
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.jump_edit.hasFocus():
                self.jump_to_image()
            else:
                self.save_annotation()
                self.next_image()
        elif key == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.save_annotation()
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_A:
            self.jump_to_unannotated()
        elif key == Qt.Key_H:
            self.show_shortcuts_help()
        elif key == Qt.Key_Equal or key == Qt.Key_Plus:
            self.zoom_level = min(200, self.zoom_level + 25)
            self.show_current_image()
        elif key == Qt.Key_Minus:
            self.zoom_level = max(25, self.zoom_level - 25)
            self.show_current_image()
        elif Qt.Key_1 <= key <= Qt.Key_9:
            role_index = key - Qt.Key_1
            if role_index < len(self.roles):
                role = self.roles[role_index]
                if role.id in self.selected_role_ids:
                    self.selected_role_ids.remove(role.id)
                    self.update_role_item_bg(role.id, Qt.white)
                else:
                    self.selected_role_ids.append(role.id)
                    self.update_role_item_bg(role.id, Qt.green)
                self.update_selected_roles_display()
        elif key == Qt.Key_0:
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    def update_role_item_bg(self, role_id, color):
        for i in range(self.role_list.count()):
            item = self.role_list.item(i)
            if item.data(Qt.UserRole) == role_id:
                item.setBackground(color)
                break

    def show_shortcuts_help(self):
        shortcuts_text = """
<b>快捷键帮助</b>

<b>导航:</b>
← / → : 上一张 / 下一张
↑ / ↓ : 增大/减小宫格数量
1-9 : 快速选择第1-9个角色
0 : 清除当前选择

<b>标注:</b>
Enter : 保存并跳转下一张
Ctrl+S : 保存
Ctrl+Z : 撤销
A : 跳转下一未标注

<b>视图:</b>
+ / - : 放大 / 缩小图片
H : 显示此帮助

<b>其他:</b>
Ctrl+O : 打开目录
Ctrl+E : 导出
"""
        QMessageBox.information(self, "快捷键帮助", shortcuts_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.images:
            self.show_current_image()

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AnnotationTool()
    window.show()

    sys.exit(app.exec_())