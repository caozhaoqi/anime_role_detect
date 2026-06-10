"""菜单组件模块 - 包含菜单栏和工具栏"""

from PyQt5.QtWidgets import QMenuBar, QToolBar, QMenu, QAction, QLabel, QPushButton

from .styles import STYLES, get_gradient_button_style


class MainMenuBar(QMenuBar):
    """主菜单栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        # 文件菜单
        file_menu = self.addMenu("文件")

        open_dir_action = QAction("打开目录", self)
        open_dir_action.setShortcut("Ctrl+O")
        open_dir_action.triggered.connect(self.main_window.browse_directory)
        file_menu.addAction(open_dir_action)

        export_json_action = QAction("导出JSON", self)
        export_json_action.setShortcut("Ctrl+E")
        export_json_action.triggered.connect(self.main_window.export_json)
        file_menu.addAction(export_json_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.main_window.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = self.addMenu("编辑")

        add_role_action = QAction("添加角色", self)
        add_role_action.setShortcut("Ctrl+N")
        add_role_action.triggered.connect(self.main_window.add_role)
        edit_menu.addAction(add_role_action)

        batch_import_action = QAction("批量导入", self)
        batch_import_action.triggered.connect(self.main_window.batch_import)
        edit_menu.addAction(batch_import_action)

        # 视图菜单
        view_menu = self.addMenu("视图")

        reset_layout_action = QAction("重置布局", self)
        reset_layout_action.triggered.connect(self.main_window.reset_layout)
        view_menu.addAction(reset_layout_action)

        # 帮助菜单
        help_menu = self.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.main_window.show_about)
        help_menu.addAction(about_action)


class MainToolBar(QToolBar):
    """主工具栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMovable(False)
        self.init_ui()

    def init_ui(self):
        self.dir_label = QLabel("未选择目录")
        self.dir_label.setStyleSheet(f"color: {STYLES['text_secondary']}; font-size: 13px;")
        self.addWidget(self.dir_label)
        self.addSeparator()

        scan_btn = QPushButton("扫描目录")
        scan_btn.setStyleSheet(
            get_gradient_button_style(
                STYLES["primary_color"],
                STYLES["secondary_color"],
                STYLES["primary_light"],
                STYLES["secondary_color"],
            )
        )
        scan_btn.clicked.connect(self.main_window.scan_directory)
        self.addWidget(scan_btn)

        self.img_count_label = QLabel("")
        self.img_count_label.setStyleSheet(f"color: {STYLES['text_secondary']}; font-size: 13px;")
        self.addWidget(self.img_count_label)

        self.addSeparator()

        stats_btn = QPushButton("统计")
        stats_btn.clicked.connect(self.main_window.show_stats)
        self.addWidget(stats_btn)
