"""导航处理模块 - 负责图片导航和索引管理"""

from PyQt5.QtWidgets import QMessageBox


class NavigationHandler:
    """导航处理器"""

    def __init__(self, main_window):
        self.main_window = main_window

    def prev_image(self):
        """上一张图片"""
        if not self.main_window.images:
            return
        if self.main_window.grid_mode > 0:
            grid_configs = {1: 4, 2: 8, 3: 16}
            count = grid_configs.get(self.main_window.grid_mode, 4)
            current_group_start = self.main_window.current_index - (
                self.main_window.current_index % count
            )
            prev_group_start = max(0, current_group_start - count)
            self.main_window.current_index = prev_group_start
        else:
            if self.main_window.current_index > 0:
                self.main_window.current_index -= 1
        self.main_window.image_handler.show_current_image()
        self.main_window.log_message(f"切换到第 {self.main_window.current_index + 1} 张图片")

    def next_image(self):
        """下一张图片"""
        if not self.main_window.images:
            return
        if self.main_window.grid_mode > 0:
            grid_configs = {1: 4, 2: 8, 3: 16}
            count = grid_configs.get(self.main_window.grid_mode, 4)
            current_group_start = self.main_window.current_index - (
                self.main_window.current_index % count
            )
            next_group_start = current_group_start + count
            if next_group_start < len(self.main_window.images):
                self.main_window.current_index = next_group_start
            else:
                self.main_window.current_index = current_group_start
        else:
            if self.main_window.current_index < len(self.main_window.images) - 1:
                self.main_window.current_index += 1
        self.main_window.image_handler.show_current_image()
        self.main_window.log_message(f"切换到第 {self.main_window.current_index + 1} 张图片")

    def jump_to_image(self):
        """跳转到指定序号的图片"""
        try:
            idx = int(self.main_window.right_panel.jump_edit.text()) - 1
            if 0 <= idx < len(self.main_window.images):
                self.main_window.current_index = idx
                self.main_window.image_handler.show_current_image()
                self.main_window.log_message(
                    f"跳转到第 {self.main_window.current_index + 1} 张图片"
                )
            else:
                QMessageBox.warning(
                    self.main_window, "警告", f"序号必须在 1 到 {len(self.main_window.images)} 之间"
                )
        except ValueError:
            QMessageBox.warning(self.main_window, "警告", "请输入有效的序号")

    def jump_to_unannotated(self):
        """跳转到下一张未标注的图片"""
        for i in range(len(self.main_window.images)):
            idx = (self.main_window.current_index + 1 + i) % len(self.main_window.images)
            if self.main_window.images[idx]["path"] not in self.main_window.annotations:
                self.main_window.current_index = idx
                self.main_window.image_handler.show_current_image()
                self.main_window.log_message(
                    f"跳转到未标注图片: {self.main_window.images[idx]['filename']}"
                )
                return
        QMessageBox.information(self.main_window, "提示", "所有图片都已标注！")

    def on_grid_mode_changed(self, index):
        """网格模式改变处理"""
        self.main_window.grid_mode = index
        self.main_window.right_panel.zoom_slider.setEnabled(index == 0)
        self.main_window.right_panel.zoom_value_label.setEnabled(index == 0)
        if hasattr(self.main_window.right_panel, "reset_zoom_btn"):
            self.main_window.right_panel.reset_zoom_btn.setEnabled(index == 0)
        if self.main_window.images:
            self.main_window.image_handler.show_current_image()

    def on_delete_mode_changed(self, state):
        """删除模式改变处理"""
        from PyQt5.QtCore import Qt

        self.main_window.delete_mode = state == Qt.Checked
        if self.main_window.delete_mode:
            self.main_window.log_message("删除模式已开启 - 点击图片将直接删除")

    def on_zoom_changed(self, value):
        """缩放改变处理"""
        self.main_window.zoom_level = value
        self.main_window.right_panel.zoom_value_label.setText(f"{value}%")
        if self.main_window.images and self.main_window.grid_mode == 0:
            self.main_window.image_handler.show_current_image()

    def reset_zoom(self):
        """重置缩放"""
        self.main_window.right_panel.zoom_slider.setValue(100)
        self.main_window.zoom_level = 100
        self.main_window.right_panel.zoom_value_label.setText("100%")
        if self.main_window.images and self.main_window.grid_mode == 0:
            self.main_window.image_handler.show_current_image()
