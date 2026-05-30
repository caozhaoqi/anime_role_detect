"""图像处理模块 - 负责图像显示和缓存管理"""

from pathlib import Path
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect


class ImageDisplayHandler:
    """图像显示处理器"""

    def __init__(self, main_window):
        self.main_window = main_window

    def show_current_image(self):
        """显示当前图像"""
        if not self.main_window.images:
            self.main_window.right_panel.image_label.setText("\n\n\n\n 请先扫描目录加载图片 \n\n\n")
            self.main_window.right_panel.image_info_label.setText("")
            return

        if self.main_window.current_index < 0 or self.main_window.current_index >= len(
            self.main_window.images
        ):
            return

        img = self.main_window.images[self.main_window.current_index]
        img_path = Path(img["path"])

        filename = img_path.name
        role_info = ""
        if img["path"] in self.main_window.annotations:
            ann = self.main_window.annotations[img["path"]]
            if ann.roles:
                role_names = []
                for role_id in ann.roles:
                    for role in self.main_window.roles:
                        if role.id == role_id:
                            role_names.append(role.name)
                            break
                role_info = " | 角色: " + ", ".join(role_names)
            else:
                role_info = " | 未标注"
        else:
            role_info = " | 未标注"

        self.main_window.right_panel.image_info_label.setText(f"{filename}{role_info}")

        if self.main_window.grid_mode == 0:
            self.show_single_image()
        else:
            self.show_grid_images()

        self.main_window.load_current_annotation()

    def show_single_image(self):
        """显示单张图片"""
        img = self.main_window.images[self.main_window.current_index]
        img_path = img["path"]

        cached = self.main_window.image_cache.get(img_path)
        if cached:
            self._display_single_pixmap(cached, img)
            return

        if img_path in self.main_window.loading_tasks:
            return

        loader = self.main_window.ImageLoader(img_path)
        loader.finished.connect(self._on_image_loaded)
        loader.error.connect(self._on_image_load_error)
        self.main_window.loading_tasks[img_path] = loader
        loader.start()

    def _on_image_loaded(self, path, pixmap):
        """图像加载完成回调"""
        if path in self.main_window.loading_tasks:
            del self.main_window.loading_tasks[path]

        self.main_window.image_cache.add(path, pixmap)

        current_img = (
            self.main_window.images[self.main_window.current_index]
            if self.main_window.images
            else None
        )
        if current_img and current_img["path"] == path:
            self._display_single_pixmap(pixmap, current_img)

    def _on_image_load_error(self, path, error_msg):
        """图像加载错误回调"""
        if path in self.main_window.loading_tasks:
            del self.main_window.loading_tasks[path]

        current_img = (
            self.main_window.images[self.main_window.current_index]
            if self.main_window.images
            else None
        )
        if current_img and current_img["path"] == path:
            self.main_window.right_panel.image_label.setText(
                f"\n\n\n\n ❌ 无法加载图片: {current_img['filename']} \n\n\n"
            )

    def _display_single_pixmap(self, pixmap, img):
        """显示单张图片的像素图"""
        if pixmap.isNull():
            self.main_window.right_panel.image_label.setText(
                f"\n\n\n\n ❌ 无法加载图片: {img['filename']} \n\n\n"
            )
            return

        base_width = self.main_window.right_panel.image_label.width()
        base_height = self.main_window.right_panel.image_label.height()

        if self.main_window.zoom_level != 100:
            new_width = int(base_width * self.main_window.zoom_level / 100)
            new_height = int(base_height * self.main_window.zoom_level / 100)
            if pixmap.width() > new_width or pixmap.height() > new_height:
                scaled_pixmap = pixmap.scaled(
                    new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = pixmap
        else:
            scaled_pixmap = pixmap.scaled(
                base_width, base_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self.main_window.right_panel.image_label.setPixmap(scaled_pixmap)

        zoom_str = (
            f" ({self.main_window.zoom_level}%)" if self.main_window.zoom_level != 100 else ""
        )
        self.main_window.right_panel.image_name_label.setText(
            f"📄 {img['filename']} [{self.main_window.current_index + 1}/{len(self.main_window.images)}]{zoom_str}"
        )
        self.main_window.right_panel.jump_edit.setText(str(self.main_window.current_index + 1))

    def show_grid_images(self):
        """显示网格图片"""
        grid_configs = {1: (2, 2), 2: (2, 4), 3: (4, 4)}
        rows, cols = grid_configs.get(self.main_window.grid_mode, (2, 2))
        count = rows * cols

        start_idx = self.main_window.current_index - (self.main_window.current_index % count)
        end_idx = min(start_idx + count, len(self.main_window.images))

        label_width = self.main_window.right_panel.image_label.width()
        label_height = self.main_window.right_panel.image_label.height()

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

        missing_paths = []
        for idx in range(start_idx, end_idx):
            if idx >= len(self.main_window.images):
                break
            img_data = self.main_window.images[idx]
            if not self.main_window.image_cache.get(img_data["path"]):
                missing_paths.append(img_data["path"])

        for path in missing_paths:
            if path not in self.main_window.loading_tasks:
                loader = self.main_window.ImageLoader(path)
                loader.finished.connect(self._on_grid_image_loaded)
                loader.error.connect(self._on_image_load_error)
                self.main_window.loading_tasks[path] = loader
                loader.start()

        for i, idx in enumerate(range(start_idx, end_idx)):
            if idx >= len(self.main_window.images):
                break

            img_data = self.main_window.images[idx]
            cached = self.main_window.image_cache.get(img_data["path"])
            if not cached:
                continue

            row = i // cols
            col = i % cols

            scaled = cached.scaled(
                cell_width - 4, cell_height - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = col * cell_width + (cell_width - scaled.width()) // 2
            y = row * cell_height + (cell_height - scaled.height()) // 2

            painter.drawPixmap(x, y, scaled)

            is_selected = idx in self.main_window.selected_indices
            cell_x = col * cell_width + 1
            cell_y = row * cell_height + 1
            cell_w = cell_width - 2
            cell_h = cell_height - 2

            if is_selected:
                painter.fillRect(cell_x, cell_y, cell_w, cell_h, QColor(255, 0, 128, 80))
                border_color = Qt.magenta
                painter.setPen(QPen(border_color, 4))
            elif idx == self.main_window.current_index:
                painter.fillRect(cell_x, cell_y, cell_w, cell_h, QColor(0, 255, 0, 40))
                border_color = Qt.white
                painter.setPen(QPen(border_color, 3))
            else:
                border_color = Qt.cyan
                painter.setPen(QPen(border_color, 1))
            painter.drawRect(cell_x, cell_y, cell_w, cell_h)

            cell_num = i + 1
            role_name = ""
            if img_data["path"] in self.main_window.annotations:
                ann = self.main_window.annotations[img_data["path"]]
                if ann.roles:
                    for role_id in ann.roles:
                        for role in self.main_window.roles:
                            if role.id == role_id:
                                role_name = role.name
                                break
                        if role_name:
                            break

            font = painter.font()
            painter.setPen(QPen(Qt.white, 1))

            bg_rect = QRect(col * cell_width + 2, row * cell_height + 2, 90, 42)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 180))

            font.setPixelSize(12)
            painter.setFont(font)
            painter.drawText(col * cell_width + 6, row * cell_height + 18, f"#{cell_num}")
            if role_name:
                font.setPixelSize(11)
                painter.setFont(font)
                painter.drawText(col * cell_width + 6, row * cell_height + 33, role_name[:10])
            font.setPixelSize(10)
            painter.setFont(font)
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawText(col * cell_width + 6, row * cell_height + 47, f"[{idx + 1}]")

        painter.end()
        self.main_window.right_panel.image_label.setPixmap(grid_pixmap)
        self.main_window.right_panel.image_label.set_grid_info(
            self.main_window.grid_mode, self.main_window.current_index, start_idx
        )

    def _on_grid_image_loaded(self, path, pixmap):
        """网格图像加载完成回调"""
        if path in self.main_window.loading_tasks:
            del self.main_window.loading_tasks[path]
        self.main_window.image_cache.add(path, pixmap)
        if self.main_window.grid_mode > 0:
            self.show_grid_images()
