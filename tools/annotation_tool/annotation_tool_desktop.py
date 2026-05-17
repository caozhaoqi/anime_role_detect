"""
动漫角色标注工具 - PyQt5独立桌面版本
支持Windows、Mac、Linux跨平台运行
"""
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QStatusBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer

from core import ImageLoader, ImageCache, AnnotationData, Role
from ui.styles import get_style_sheet, STYLES
from ui.panels import LeftPanel, RightPanel
from ui.menu import MainMenuBar, MainToolBar
from handlers.image_handlers import ImageDisplayHandler
from handlers.navigation_handlers import NavigationHandler
from handlers.annotation_handlers import AnnotationHandler
from data import load_roles, save_roles, load_annotations, scan_images
from utils import setup_logger

logger = setup_logger()


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
        self.auto_save_delay = 2000
        self.delete_mode = False
        self.image_cache = ImageCache(max_size=100)
        self.loading_tasks = {}
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._delayed_show_image)
        
        # 初始化处理器
        self.ImageLoader = ImageLoader
        self.image_handler = ImageDisplayHandler(self)
        self.navigation_handler = NavigationHandler(self)
        self.annotation_handler = AnnotationHandler(self)
        
        # 连接处理器信号
        self.auto_save_timer.timeout.connect(self.annotation_handler.auto_save)
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("动漫角色标注工具")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(get_style_sheet())

        # 创建菜单栏
        self.menu_bar = MainMenuBar(self)
        self.setMenuBar(self.menu_bar)
        
        # 创建工具栏
        self.tool_bar = MainToolBar(self)
        self.addToolBar(self.tool_bar)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建左侧面板
        self.left_panel = LeftPanel(self)
        self.left_panel.setMinimumWidth(280)
        self.left_panel.setMaximumWidth(320)
        self.left_panel.role_search.textChanged.connect(self.filter_roles)
        self.left_panel.role_list.itemClicked.connect(self.annotation_handler.on_role_item_clicked)
        main_layout.addWidget(self.left_panel)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet(f"background-color: {STYLES['border_color']};")
        separator.setFixedWidth(1)
        main_layout.addWidget(separator)

        # 创建右侧面板
        self.right_panel = RightPanel(self)
        self.right_panel.image_label.clicked.connect(self.on_image_clicked)
        main_layout.addWidget(self.right_panel, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.load_roles()
        self.load_annotations()
    
    def load_roles(self):
        self.roles = load_roles()
        self.update_role_list()
    
    def load_annotations(self):
        self.annotations = load_annotations()
        self.update_stats()
    
    def update_role_list(self):
        self.left_panel.role_list.clear()
        for role in self.roles:
            item_text = f"{role.name} ({role.name_cn})" if role.name_cn else role.name
            if role.category and role.category != "其他":
                category_color = {
                    "主角": "#10b981",
                    "配角": "#3b82f6",
                    "反派": "#ef4444",
                    "路人": "#6b7280"
                }.get(role.category, "#6b7280")
                item_text += f" <span style='color:{category_color};'>[{role.category}]</span>"
            item = self.left_panel.role_list.addItem(item_text)
    
    def filter_roles(self, text):
        for i in range(self.left_panel.role_list.count()):
            item = self.left_panel.role_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())
    
    def update_stats(self):
        total = len(self.images)
        annotated = len(self.annotations)
        self.status_bar.showMessage(f"总图片: {total} | 已标注: {annotated} | 未标注: {total - annotated}")
    
    def browse_directory(self):
        from PyQt5.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.data_dir = dir_path
            self.tool_bar.dir_label.setText(f"目录: {dir_path[:50]}...")
            self.scan_directory()
    
    def scan_directory(self):
        if not hasattr(self, 'data_dir') or not self.data_dir:
            QMessageBox.warning(self, "警告", "请先选择目录")
            return

        self.images = scan_images(self.data_dir)
        self.current_index = 0
        self.tool_bar.img_count_label.setText(f"共 {len(self.images)} 张图片")
        self.log_message(f"扫描完成，共发现 {len(self.images)} 张图片")
        
        if self.images:
            self.image_handler.show_current_image()
        self.update_stats()
    
    def log_message(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.right_panel.log_text.append(f"[{timestamp}] {msg}")
        logger.info(msg)
    
    def on_image_clicked(self):
        if self.delete_mode and self.images:
            self.annotation_handler.delete_image_at_index(self.current_index)
    
    def export_json(self):
        if not self.annotations:
            QMessageBox.information(self, "提示", "没有可导出的标注")
            return

        from PyQt5.QtWidgets import QFileDialog
        import json
        file_path, _ = QFileDialog.getSaveFileName(self, "导出JSON", "annotations_export.json", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    data = [ann.to_dict() for ann in self.annotations.values()]
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.log_message(f"已导出到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导出失败:\n{str(e)}")
    
    def show_stats(self):
        total = len(self.images)
        annotated = len(self.annotations)
        
        role_stats = {}
        for ann in self.annotations.values():
            for role_id in ann.roles:
                role_stats[role_id] = role_stats.get(role_id, 0) + 1
        
        stats_text = f"总图片: {total}\n已标注: {annotated}\n未标注: {total - annotated}\n\n"
        stats_text += "角色统计:\n"
        for role_id, count in sorted(role_stats.items(), key=lambda x: -x[1])[:10]:
            for role in self.roles:
                if role.id == role_id:
                    stats_text += f"  {role.name}: {count}\n"
                    break
        
        QMessageBox.information(self, "统计", stats_text)
    
    def show_about(self):
        QMessageBox.about(self, "关于",
            "动漫角色标注工具 v1.0\n\n"
            "用于标注动漫角色图片的工具\n"
            "支持批量标注和AI识别"
        )
    
    def reset_layout(self):
        pass
    
    def infer_role_with_ai(self):
        if not self.images:
            QMessageBox.information(self, "提示", "请先加载图片")
            return
        QMessageBox.information(self, "AI识别", "AI识别功能开发中...")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.images:
            return
        if hasattr(self, '_last_resize_time') and time.time() - self._last_resize_time < 0.1:
            return
        self._last_resize_time = time.time()
        self.resize_timer.start(100)
    
    def _delayed_show_image(self):
        if not hasattr(self, '_is_showing_image'):
            self._is_showing_image = False
        if self._is_showing_image:
            return
        self._is_showing_image = True
        try:
            if self.images:
                self.image_handler.show_current_image()
        finally:
            self._is_showing_image = False
    
    # 导航方法委托给导航处理器
    def prev_image(self):
        self.navigation_handler.prev_image()
    
    def next_image(self):
        self.navigation_handler.next_image()
    
    def jump_to_image(self):
        self.navigation_handler.jump_to_image()
    
    def jump_to_unannotated(self):
        self.navigation_handler.jump_to_unannotated()
    
    def on_grid_mode_changed(self, index):
        self.navigation_handler.on_grid_mode_changed(index)
    
    def on_delete_mode_changed(self, state):
        self.navigation_handler.on_delete_mode_changed(state)
    
    def on_zoom_changed(self, value):
        self.navigation_handler.on_zoom_changed(value)
    
    def reset_zoom(self):
        self.navigation_handler.reset_zoom()
    
    # 标注方法委托给标注处理器
    def on_annotation_changed(self):
        self.annotation_handler.on_annotation_changed()
    
    def load_current_annotation(self):
        self.annotation_handler.load_current_annotation()
    
    def save_annotation(self):
        self.annotation_handler.save_annotation()
    
    def clear_selection(self):
        self.annotation_handler.clear_selection()
    
    def delete_annotation(self):
        self.annotation_handler.delete_annotation()
    
    def delete_image_at_index(self, idx):
        self.annotation_handler.delete_image_at_index(idx)

    def jump_to_index(self, idx):
        if self.grid_mode != 0:
            self.grid_mode_combo.blockSignals(True)
            self.grid_mode_combo.setCurrentIndex(0)
            self.grid_mode_combo.blockSignals(False)
            self.grid_mode = 0
        self.current_index = idx
        self.image_handler.show_current_image()
        self.log_message(f"跳转到第 {idx + 1} 张图片")

    def move_to_untrainable(self, category):
        self.annotation_handler.move_to_untrainable(category)
    
    def move_image_at_index(self, idx, category):
        self.annotation_handler.move_image_at_index(idx, category)
    
    def add_role(self):
        self.annotation_handler.add_role()
    
    def delete_role(self):
        self.annotation_handler.delete_role()
    
    def batch_import(self):
        self.annotation_handler.batch_import()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AnnotationTool()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
