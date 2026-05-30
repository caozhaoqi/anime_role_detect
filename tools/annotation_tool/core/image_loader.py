from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap


class ImageLoader(QThread):
    finished = pyqtSignal(str, QPixmap)
    error = pyqtSignal(str, str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            pixmap = QPixmap(self.path)
            if pixmap.isNull():
                self.error.emit(self.path, "图片加载失败")
            else:
                self.finished.emit(self.path, pixmap)
        except Exception as e:
            self.error.emit(self.path, str(e))
