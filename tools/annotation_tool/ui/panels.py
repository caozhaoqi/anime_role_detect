"""面板组件模块 - 简化布局，聚焦快速标注"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit,
    QListWidget, QPushButton, QScrollArea, QFrame, QSplitter,
    QLabel, QTextEdit, QComboBox, QCheckBox, QSlider, QSizePolicy
)
from PyQt5.QtCore import Qt

from .styles import STYLES, get_button_style, get_gradient_button_style
from .image_label import ClickableLabel


class LeftPanel(QScrollArea):
    """左侧角色列表面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {STYLES['bg_card']};")
        layout = QVBoxLayout(panel)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        roles_group = QGroupBox()
        roles_group.setTitle("角色列表")
        roles_group.setStyleSheet(f"""
            QGroupBox {{ background-color: transparent; border: none; margin: 0; padding: 12px; }}
            QGroupBox::title {{
                color: {STYLES['text_secondary']}; font-size: 13px; font-weight: 600;
                padding: 0 8px; margin-bottom: 8px;
            }}
        """)

        roles_layout = QVBoxLayout(roles_group)
        roles_layout.setSpacing(8)

        self.role_search = QLineEdit()
        self.role_search.setPlaceholderText("搜索角色...")
        self.role_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {STYLES['bg_input']}; border: 1px solid {STYLES['border_color']};
                border-radius: {STYLES['border_radius']}; padding: 8px 12px; font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {STYLES['primary_color']}; }}
        """)
        roles_layout.addWidget(self.role_search)

        self.role_list = QListWidget()
        self.role_list.setSelectionMode(QListWidget.MultiSelection)
        self.role_list.setMinimumHeight(300)
        self.role_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {STYLES['bg_input']}; border: 1px solid {STYLES['border_color']};
                border-radius: {STYLES['border_radius']}; padding: 4px;
                font-size: 13px;
            }}
            QListWidget::item {{ padding: 8px 10px; border-radius: 4px; margin: 1px; }}
            QListWidget::item:hover {{ background-color: {STYLES['bg_card_hover']}; }}
            QListWidget::item:selected {{ background-color: {STYLES['primary_color']}; color: white; }}
        """)
        roles_layout.addWidget(self.role_list)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        add_btn = QPushButton("添加")
        add_btn.setStyleSheet(get_button_style(STYLES['success_color'], "#34d399"))
        add_btn.setMinimumHeight(32)
        add_btn.clicked.connect(self.main_window.add_role)

        del_btn = QPushButton("删除")
        del_btn.setStyleSheet(get_button_style(STYLES['danger_color'], "#f87171"))
        del_btn.setMinimumHeight(32)
        del_btn.clicked.connect(self.main_window.delete_role)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        roles_layout.addLayout(btn_layout)

        layout.addWidget(roles_group)
        self.setWidget(panel)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {STYLES['bg_card']}; border: none;")


class RightPanel(QWidget):
    """右侧图片预览和标注面板 - 固定布局防止大小变化"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._create_image_section()
        self._create_annotation_section()
        self._create_log_section()

        main_layout.addWidget(self.image_group, 3)
        main_layout.addWidget(self.annotation_group, 1)
        main_layout.addWidget(self.log_group, 0)

    def _create_image_section(self):
        self.image_group = QWidget()
        self.image_group.setStyleSheet(f"""
            QWidget {{
                background-color: {STYLES['bg_card']};
                border: 1px solid {STYLES['border_color']};
                border-radius: {STYLES['border_radius']};
            }}
        """)
        image_layout = QVBoxLayout(self.image_group)
        image_layout.setSpacing(6)
        image_layout.setContentsMargins(10, 10, 10, 10)

        header = QHBoxLayout()
        self.image_name_label = QLabel("未加载图片")
        self.image_name_label.setStyleSheet(f"color: {STYLES['text_secondary']}; font-size: 12px;")
        header.addWidget(self.image_name_label)

        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet(f"color: {STYLES['info_color']}; font-size: 11px;")
        self.image_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.image_info_label, 0, Qt.AlignRight)
        image_layout.addLayout(header)

        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background: linear-gradient(135deg, {STYLES['bg_dark']} 0%, #1e293b 100%);
                border: 2px dashed {STYLES['border_color']}; border-radius: {STYLES['border_radius_large']};
                color: {STYLES['text_muted']}; font-size: 13px;
            }}
        """)
        self.image_label.setText("\n\n\n\n 请先扫描目录加载图片 \n\n\n")
        image_layout.addWidget(self.image_label, 1)

        nav_bar = QWidget()
        nav_bar.setStyleSheet(f"background-color: {STYLES['bg_input']}; border-radius: {STYLES['border_radius']}; padding: 6px;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setSpacing(6)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        self.prev_btn = QPushButton("上一个")
        self.prev_btn.setMinimumHeight(32)
        self.prev_btn.clicked.connect(self.main_window.prev_image)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("下一个")
        self.next_btn.setMinimumHeight(32)
        self.next_btn.clicked.connect(self.main_window.next_image)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addSpacing(12)

        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("序号")
        self.jump_edit.setMaximumWidth(60)
        self.jump_edit.setMinimumHeight(32)
        nav_layout.addWidget(self.jump_edit)

        self.jump_btn = QPushButton("跳转")
        self.jump_btn.setMinimumHeight(32)
        self.jump_btn.clicked.connect(self.main_window.jump_to_image)
        nav_layout.addWidget(self.jump_btn)

        self.unannotated_btn = QPushButton("下一未标")
        self.unannotated_btn.setStyleSheet(get_button_style(STYLES['warning_color'], "#fbbf24"))
        self.unannotated_btn.setMinimumHeight(32)
        self.unannotated_btn.clicked.connect(self.main_window.jump_to_unannotated)
        nav_layout.addWidget(self.unannotated_btn)

        nav_layout.addStretch()
        image_layout.addWidget(nav_bar)

        control_bar = QWidget()
        control_bar.setStyleSheet("background-color: transparent;")
        control_layout = QHBoxLayout(control_bar)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(0, 0, 0, 0)

        control_layout.addWidget(QLabel("预览:"))
        self.grid_mode_combo = QComboBox()
        self.grid_mode_combo.addItems(["单图", "4宫格", "8宫格", "16宫格"])
        self.grid_mode_combo.currentIndexChanged.connect(self.main_window.on_grid_mode_changed)
        self.grid_mode_combo.setMinimumHeight(28)
        control_layout.addWidget(self.grid_mode_combo)

        control_layout.addSpacing(8)
        self.delete_mode_check = QCheckBox("删除模式")
        self.delete_mode_check.stateChanged.connect(self.main_window.on_delete_mode_changed)
        control_layout.addWidget(self.delete_mode_check)

        control_layout.addStretch()

        control_layout.addWidget(QLabel("缩放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(100)
        self.zoom_slider.setMinimumHeight(20)
        self.zoom_slider.valueChanged.connect(self.main_window.on_zoom_changed)
        control_layout.addWidget(self.zoom_slider)

        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setStyleSheet(f"color: {STYLES['primary_color']}; font-weight: 600; font-size: 11px;")
        self.zoom_value_label.setMinimumWidth(40)
        control_layout.addWidget(self.zoom_value_label)

        self.reset_zoom_btn = QPushButton("重置")
        self.reset_zoom_btn.setMinimumHeight(28)
        self.reset_zoom_btn.clicked.connect(self.main_window.reset_zoom)
        control_layout.addWidget(self.reset_zoom_btn)

        image_layout.addWidget(control_bar)

    def _create_annotation_section(self):
        self.annotation_group = QWidget()
        self.annotation_group.setStyleSheet(f"""
            QWidget {{
                background-color: {STYLES['bg_card']};
                border: 1px solid {STYLES['border_color']};
                border-radius: {STYLES['border_radius']};
            }}
        """)
        ann_layout = QVBoxLayout(self.annotation_group)
        ann_layout.setSpacing(6)
        ann_layout.setContentsMargins(10, 8, 10, 8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("已选:"))
        self.selected_roles_label = QLabel("未选择")
        self.selected_roles_label.setStyleSheet(f"color: {STYLES['primary_color']}; font-weight: 500; font-size: 12px;")
        top_row.addWidget(self.selected_roles_label)
        top_row.addStretch()

        self.multi_role_check = QCheckBox("多角色")
        self.multi_role_check.stateChanged.connect(self.main_window.on_annotation_changed)
        top_row.addWidget(self.multi_role_check)

        self.nsfw_check = QCheckBox("R18")
        self.nsfw_check.setStyleSheet(f"QCheckBox {{ color: {STYLES['danger_color']}; font-size: 12px; }}")
        self.nsfw_check.stateChanged.connect(self.main_window.on_annotation_changed)
        top_row.addWidget(self.nsfw_check)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("备注...")
        self.notes_edit.setMinimumHeight(28)
        self.notes_edit.textChanged.connect(self.main_window.on_annotation_changed)
        top_row.addWidget(self.notes_edit, 2)
        ann_layout.addLayout(top_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        save_btn = QPushButton("保存")
        save_btn.setStyleSheet(get_gradient_button_style(STYLES['primary_color'], STYLES['secondary_color'], STYLES['primary_light'], STYLES['secondary_color']))
        save_btn.setMinimumHeight(32)
        save_btn.clicked.connect(self.main_window.save_annotation)
        btn_row.addWidget(save_btn)

        clear_btn = QPushButton("清除")
        clear_btn.setMinimumHeight(32)
        clear_btn.clicked.connect(self.main_window.clear_selection)
        btn_row.addWidget(clear_btn)

        ai_infer_btn = QPushButton("AI识别")
        ai_infer_btn.setStyleSheet(get_gradient_button_style(STYLES['secondary_color'], "#a855f7", "#a78bfa", "#c084fc"))
        ai_infer_btn.setMinimumHeight(32)
        ai_infer_btn.clicked.connect(self.main_window.infer_role_with_ai)
        btn_row.addWidget(ai_infer_btn)

        btn_row.addStretch()
        ann_layout.addLayout(btn_row)

        move_row = QHBoxLayout()
        move_row.setSpacing(6)

        move_r18_btn = QPushButton("R18")
        move_r18_btn.setStyleSheet(get_gradient_button_style("#ef4444", "#f97316", "#f87171", "#fb923c"))
        move_r18_btn.setMinimumHeight(30)
        move_r18_btn.clicked.connect(lambda: self.main_window.move_to_untrainable("R18"))
        move_row.addWidget(move_r18_btn)

        move_multi_btn = QPushButton("多角色")
        move_multi_btn.setStyleSheet(get_gradient_button_style("#10b981", "#34d399", "#34d399", "#6ee7b7"))
        move_multi_btn.setMinimumHeight(30)
        move_multi_btn.clicked.connect(lambda: self.main_window.move_to_untrainable("多角色"))
        move_row.addWidget(move_multi_btn)

        move_other_btn = QPushButton("其他")
        move_other_btn.setStyleSheet(get_gradient_button_style("#6b7280", "#9ca3af", "#9ca3af", "#d1d5db"))
        move_other_btn.setMinimumHeight(30)
        move_other_btn.clicked.connect(lambda: self.main_window.move_to_untrainable("其他"))
        move_row.addWidget(move_other_btn)

        move_row.addStretch()
        ann_layout.addLayout(move_row)

    def _create_log_section(self):
        self.log_group = QWidget()
        self.log_group.setStyleSheet(f"""
            QWidget {{
                background-color: {STYLES['bg_card']};
                border: 1px solid {STYLES['border_color']};
                border-radius: {STYLES['border_radius']};
            }}
        """)
        log_layout = QVBoxLayout(self.log_group)
        log_layout.setSpacing(4)
        log_layout.setContentsMargins(8, 6, 8, 6)

        log_header = QLabel("操作日志")
        log_header.setStyleSheet(f"color: {STYLES['text_secondary']}; font-size: 11px; font-weight: 600;")
        log_layout.addWidget(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(60)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {STYLES['bg_dark']}; color: {STYLES['success_color']};
                font-family: monospace; font-size: 10px;
                border: 1px solid {STYLES['border_color']}; border-radius: {STYLES['border_radius']};
                padding: 4px;
            }}
        """)
        log_layout.addWidget(self.log_text)
