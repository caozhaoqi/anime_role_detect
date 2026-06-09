"""标注处理模块 - 负责角色标注和文件操作"""

import time
import shutil
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox, QDialog

from core import AnnotationData
from data import save_annotation, delete_annotation, save_roles
from services import get_untrainable_dirs


class AnnotationHandler:
    """标注处理器"""

    def __init__(self, main_window):
        self.main_window = main_window

    def load_current_annotation(self):
        """加载当前图片的标注信息"""
        if not self.main_window.images:
            return

        img = self.main_window.images[self.main_window.current_index]
        self.main_window.selected_role_ids = []

        if img["path"] in self.main_window.annotations:
            ann = self.main_window.annotations[img["path"]]
            self.main_window.selected_role_ids = ann.roles.copy()
            self.main_window.right_panel.multi_role_check.setChecked(ann.is_multi_role)
            self.main_window.right_panel.nsfw_check.setChecked(ann.is_nsfw)
            self.main_window.right_panel.notes_edit.setText(ann.notes)
        else:
            self.main_window.right_panel.multi_role_check.setChecked(False)
            self.main_window.right_panel.nsfw_check.setChecked(False)
            self.main_window.right_panel.notes_edit.setText("")

        self.update_selected_roles_label()
        self.update_role_list_selection()

    def update_selected_roles_label(self):
        """更新已选角色标签"""
        if not self.main_window.selected_role_ids:
            self.main_window.right_panel.selected_roles_label.setText("未选择")
            return

        names = []
        for role_id in self.main_window.selected_role_ids:
            for role in self.main_window.roles:
                if role.id == role_id:
                    names.append(role.name)
                    break
        self.main_window.right_panel.selected_roles_label.setText(", ".join(names))

    def update_role_list_selection(self):
        """更新角色列表选择状态"""
        for i in range(self.main_window.left_panel.role_list.count()):
            item = self.main_window.left_panel.role_list.item(i)
            role = self.main_window.roles[i] if i < len(self.main_window.roles) else None
            if role and role.id in self.main_window.selected_role_ids:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def on_role_item_clicked(self, item):
        """角色列表项点击处理"""
        row = self.main_window.left_panel.role_list.row(item)
        if row < len(self.main_window.roles):
            role = self.main_window.roles[row]
            if role.id in self.main_window.selected_role_ids:
                self.main_window.selected_role_ids.remove(role.id)
            else:
                self.main_window.selected_role_ids.append(role.id)
            self.update_selected_roles_label()
            self.main_window.auto_save_timer.start(self.main_window.auto_save_delay)

    def on_annotation_changed(self):
        """标注信息改变处理"""
        self.main_window.auto_save_timer.start(self.main_window.auto_save_delay)

    def auto_save(self):
        """自动保存标注"""
        if not self.main_window.images or self.main_window.current_index >= len(
            self.main_window.images
        ):
            return

        img = self.main_window.images[self.main_window.current_index]
        ann = self.main_window.annotations.get(img["path"], AnnotationData())
        ann.image_path = img["path"]
        ann.roles = self.main_window.selected_role_ids.copy()
        ann.is_multi_role = self.main_window.right_panel.multi_role_check.isChecked()
        ann.is_nsfw = self.main_window.right_panel.nsfw_check.isChecked()
        ann.notes = self.main_window.right_panel.notes_edit.text()
        ann.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        self.main_window.annotations[img["path"]] = ann
        save_annotation(img["path"], ann)
        self.main_window.log_message(f"已保存标注: {img['filename']}")
        self.main_window.update_stats()

    def save_annotation(self):
        """手动保存标注"""
        self.main_window.auto_save_timer.stop()
        self.auto_save()

    def clear_selection(self):
        """清除选择"""
        self.main_window.selected_role_ids = []
        self.main_window.right_panel.multi_role_check.setChecked(False)
        self.main_window.right_panel.nsfw_check.setChecked(False)
        self.main_window.right_panel.notes_edit.setText("")
        self.update_selected_roles_label()
        self.update_role_list_selection()
        self.main_window.auto_save_timer.start(self.main_window.auto_save_delay)

    def delete_annotation(self):
        """删除标注"""
        if not self.main_window.images:
            return

        img = self.main_window.images[self.main_window.current_index]
        if img["path"] in self.main_window.annotations:
            delete_annotation(img["path"])
            del self.main_window.annotations[img["path"]]
            self.main_window.log_message(f"已删除标注: {img['filename']}")

        self.clear_selection()
        self.main_window.update_stats()

    def move_to_untrainable(self, category):
        """移动到无法训练目录"""
        if not self.main_window.images:
            return
        self.move_image_at_index(self.main_window.current_index, category)

    def move_image_at_index(self, idx, category):
        """移动指定索引的图片"""
        if idx < 0 or idx >= len(self.main_window.images):
            return
        img = self.main_window.images[idx]
        source = Path(img["path"])

        if not source.exists():
            QMessageBox.warning(self.main_window, "警告", "源文件不存在")
            return

        untrainable_dirs = get_untrainable_dirs()
        dest_dir = untrainable_dirs.get(category, untrainable_dirs["其他"])
        dest = dest_dir / source.name

        old_index = self.main_window.current_index
        old_folder = img.get("folder", "")

        try:
            shutil.move(str(source), str(dest))
            self.main_window.log_message(f"已移动到无法训练/{category}: {img['filename']}")
            self.main_window.image_cache.remove(img["path"])
            self.main_window.right_panel.image_label.clear()
            self.main_window.right_panel.image_label.setText("\n\n\n\n 图片已移动 \n\n\n")
            self.main_window.scan_directory()

            new_index = -1
            if old_index < len(self.main_window.images):
                new_index = old_index
            elif old_folder:
                for i, img_data in enumerate(self.main_window.images):
                    if img_data.get("folder", "") == old_folder and i < len(
                        self.main_window.images
                    ):
                        new_index = min(i, len(self.main_window.images) - 1)
                        break

            if new_index < 0:
                new_index = (
                    max(0, len(self.main_window.images) - 1) if self.main_window.images else 0
                )

            self.main_window.current_index = new_index

            if self.main_window.images:
                self.main_window.image_handler.show_current_image()
        except Exception as e:
            QMessageBox.warning(self.main_window, "错误", f"移动失败:\n{str(e)}")

    def delete_image_at_index(self, idx):
        """删除指定索引的图片"""
        if idx < 0 or idx >= len(self.main_window.images):
            return
        img = self.main_window.images[idx]
        source = Path(img["path"])

        if not source.exists():
            QMessageBox.warning(self.main_window, "警告", "文件不存在")
            return

        reply = QMessageBox.question(
            self.main_window, "确认", f"确定要删除图片吗？\n{img['filename']}"
        )
        if reply == QMessageBox.Yes:
            old_index = self.main_window.current_index
            old_folder = img.get("folder", "")
            try:
                source.unlink()
                self.main_window.log_message(f"已删除: {img['filename']}")
                if img["path"] in self.main_window.annotations:
                    delete_annotation(img["path"])
                    del self.main_window.annotations[img["path"]]
                self.main_window.image_cache.remove(img["path"])
                self.main_window.scan_directory()

                new_index = -1
                if old_index < len(self.main_window.images):
                    new_index = old_index
                elif old_folder:
                    for i, img_data in enumerate(self.main_window.images):
                        if img_data.get("folder", "") == old_folder and i < len(
                            self.main_window.images
                        ):
                            new_index = min(i, len(self.main_window.images) - 1)
                            break

                if new_index < 0:
                    new_index = (
                        max(0, len(self.main_window.images) - 1) if self.main_window.images else 0
                    )

                self.main_window.current_index = new_index

                if self.main_window.images:
                    self.main_window.image_handler.show_current_image()
            except Exception as e:
                QMessageBox.warning(self.main_window, "错误", f"删除失败:\n{str(e)}")

    def add_role(self):
        """添加角色"""
        from ui.dialogs import AddRoleDialog

        dialog = AddRoleDialog(self.main_window)
        if dialog.exec_() == QDialog.Accepted:
            role = dialog.get_role()
            if role.name:
                self.main_window.roles.append(role)
                save_roles(self.main_window.roles)
                self.main_window.update_role_list()
                self.main_window.log_message(f"已添加角色: {role.name}")

    def delete_role(self):
        """删除角色"""
        from PyQt5.QtWidgets import QDialog

        current_row = self.main_window.left_panel.role_list.currentRow()
        if current_row >= 0 and current_row < len(self.main_window.roles):
            role = self.main_window.roles[current_row]
            reply = QMessageBox.question(
                self.main_window, "确认", f"确定要删除角色 {role.name} 吗？"
            )
            if reply == QMessageBox.Yes:
                del self.main_window.roles[current_row]
                save_roles(self.main_window.roles)
                self.main_window.update_role_list()
                self.main_window.log_message(f"已删除角色: {role.name}")

    def batch_import(self):
        """批量导入角色"""
        from ui.dialogs import BatchImportDialog
        from PyQt5.QtWidgets import QDialog

        dialog = BatchImportDialog(self.main_window)
        if dialog.exec_() == QDialog.Accepted:
            new_roles = dialog.get_roles()
            for role in new_roles:
                existing = False
                for existing_role in self.main_window.roles:
                    if existing_role.name == role.name:
                        existing = True
                        break
                if not existing:
                    self.main_window.roles.append(role)
            save_roles(self.main_window.roles)
            self.main_window.update_role_list()
            self.main_window.log_message(f"批量导入完成: {len(new_roles)} 个角色")
