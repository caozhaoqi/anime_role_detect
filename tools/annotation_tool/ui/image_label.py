import logging
from PyQt5.QtWidgets import QLabel, QMenu, QAction
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QPixmap

logger = logging.getLogger("AnnotationTool")


class ZoomableDragableLabel(QLabel):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.zoom_level = 100
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint(0, 0)
        self.zoom_step = 25
        self.min_zoom = 25
        self.max_zoom = 400
        self._is_resizing = False
        self._last_resize_time = 0
        
    def _log_event(self, event_name, event):
        btn_map = {0: "None", 1: "Left", 2: "Right", 4: "Mid"}
        if hasattr(event, 'button'):
            btn = event.button()
            btn_str = btn_map.get(int(btn), str(btn))
        else:
            btn_str = "N/A"
        pos = event.pos()
        logger.debug(f"[Event#{event_name}] {event_name}: btn={btn_str}, pos=({pos.x()},{pos.y()}), size={self.width()}x{self.height()}")
    
    def setPixmap(self, pixmap):
        self.current_pixmap = pixmap
        super().setPixmap(pixmap)
        self.reset_pan()
    
    def reset_pan(self):
        self.pan_offset = QPoint(0, 0)
    
    def clear(self):
        self.current_pixmap = None
        super().clear()
        self.reset_pan()
    
    def has_pixmap(self):
        return self.current_pixmap is not None and not self.current_pixmap.isNull()
    
    def wheelEvent(self, event):
        self._log_event("wheelEvent", event)
        if not self.current_pixmap:
            return
        
        delta = event.angleDelta().y()
        if delta == 0:
            logger.info(f"[Event] wheelEvent - delta=0, ignoring")
            return
        
        old_zoom = self.zoom_level
        if delta > 0:
            self.zoom_level = min(self.max_zoom, self.zoom_level + self.zoom_step)
        else:
            self.zoom_level = max(self.min_zoom, self.zoom_level - self.zoom_step)
        
        if self.zoom_level != old_zoom:
            logger.info(f"[Event#] wheelEvent: delta={delta}, zoom_before={old_zoom}, zoom_after={self.zoom_level}")
            self.update()
    
    def mousePressEvent(self, event):
        self._log_event("mousePressEvent", event)
        if event.button() == Qt.LeftButton:
            if self.zoom_level != 100:
                self.is_panning = True
                self.last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                logger.info(f"[Event] Left button pressed - panning started")
            else:
                self.clicked.emit()
                logger.info(f"[Event] Left button pressed - clicked (no zoom)")
        elif event.button() == Qt.RightButton:
            logger.info(f"[Event] Right button pressed - context menu requested")
            self.clicked.emit()
        elif event.button() == Qt.MidButton:
            self.reset_zoom()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.is_panning and self.current_pixmap:
            pos = event.pos()
            self.pan_offset += pos - self.last_pan_pos
            self.last_pan_pos = pos
            self.update()
        
        if hasattr(self, '_log_event'):
            parent = self.parent()
            parent_size = (parent.width(), parent.height()) if parent else (0, 0)
            label_size = (self.width(), self.height())
            zoom = self.zoom_level if hasattr(self, 'zoom_level') else 100
            move_pos = event.pos() if event else None
            if move_pos:
                logger.debug(f"[Event#] mouseMoveEvent: btn=0, pos=({move_pos.x()},{move_pos.y()}), label_size={label_size[0]}x{label_size[1]}, parent_size={parent_size[0]}x{parent_size[1]}, is_panning={self.is_panning}, zoom={zoom}")
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        self._log_event("mouseReleaseEvent", event)
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)
            logger.info("[Event] Left button released - panning stopped")
        elif event.button() == Qt.RightButton:
            logger.info("[Event] Right button released - accepted")
            event.accept()
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

                x = (self.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x()
                y = (self.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y()
                qp.drawPixmap(x, y, scaled_pixmap)
            else:
                x = (self.width() - self.current_pixmap.width()) // 2 + self.pan_offset.x()
                y = (self.height() - self.current_pixmap.height()) // 2 + self.pan_offset.y()
                qp.drawPixmap(x, y, self.current_pixmap)
        else:
            super().paintEvent(event)
    
    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)


class ClickableLabel(ZoomableDragableLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_mode = 0
        self.current_grid_index = 0
        self.grid_start_index = 0
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
    
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
    
    def get_clicked_grid_index(self, pos):
        if self.grid_mode == 0:
            return self.current_grid_index

        main_window = self.get_main_window()
        if main_window is None:
            return None

        grid_configs = {1: (2, 2), 2: (2, 4), 3: (4, 4)}
        rows, cols = grid_configs.get(self.grid_mode, (2, 2))

        label_width = main_window.right_panel.image_label.width()
        label_height = main_window.right_panel.image_label.height()
        
        max_width = 1200
        max_height = 800
        label_width = min(label_width, max_width)
        label_height = min(label_height, max_height)

        cell_width = label_width // cols if cols > 0 else label_width
        cell_height = label_height // rows if rows > 0 else label_height

        col = pos.x() // cell_width
        row = pos.y() // cell_height

        if row >= rows or col >= cols:
            return None

        cell_index = row * cols + col
        clicked_idx = self.grid_start_index + cell_index

        logger.info(f"[GridClick] pos={pos}, label={label_width}x{label_height}, cell={cell_width}x{cell_height}, row={row}, col={col}, cell_index={cell_index}, grid_start={self.grid_start_index}, clicked_idx={clicked_idx}")

        if clicked_idx >= len(main_window.images):
            return None

        return clicked_idx
    
    def show_context_menu(self, pos):
        import functools
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

        grid_configs = {1: 4, 2: 8, 3: 16}
        count = grid_configs.get(self.grid_mode, 4)

        clicked_idx = self.get_clicked_grid_index(pos)
        logger.info(f"[Menu] clicked_idx from get_clicked_grid_index: {clicked_idx}, type: {type(clicked_idx)}")
        if clicked_idx is None or clicked_idx >= len(main_window.images):
            logger.info(f"[Menu] Invalid clicked index: {clicked_idx}, total images: {len(main_window.images)}")
            return

        img = main_window.images[clicked_idx]
        filename = img['filename']

        delete_action = QAction(f"删除图片 ({clicked_idx + 1})", self)
        delete_action.triggered.connect(functools.partial(main_window.delete_image_at_index, clicked_idx))
        menu.addAction(delete_action)

        move_r18_action = QAction(f"移动到R18 ({clicked_idx + 1})", self)
        move_r18_action.triggered.connect(functools.partial(main_window.move_image_at_index, clicked_idx, "R18"))
        menu.addAction(move_r18_action)

        move_multi_action = QAction(f"移动到多角色 ({clicked_idx + 1})", self)
        move_multi_action.triggered.connect(functools.partial(main_window.move_image_at_index, clicked_idx, "多角色"))
        menu.addAction(move_multi_action)

        move_other_action = QAction(f"移动到其他 ({clicked_idx + 1})", self)
        move_other_action.triggered.connect(functools.partial(main_window.move_image_at_index, clicked_idx, "其他"))
        menu.addAction(move_other_action)

        menu.addSeparator()

        jump_action = QAction(f"跳转查看 ({clicked_idx + 1})", self)
        jump_action.triggered.connect(functools.partial(main_window.jump_to_index, clicked_idx))
        menu.addAction(jump_action)

        global_pos = self.mapToGlobal(pos)
        logger.info(f"[Menu] About to exec menu at global pos={global_pos}")
        menu.exec_(global_pos)
        logger.info("[Menu] Menu closed")
