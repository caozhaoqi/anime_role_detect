import sys
import os
sys.path.append('spider_image_system')
from spider_image_system.src.image.spider_img_save import download_img_txt

class MockLabel:
    def setText(self, text):
        pass

class MockUI:
    def __init__(self):
        self.download_show_label = MockLabel()
    
    def sys_tips(self, text):
        pass
    
    def success_tips(self, text):
        pass

download_img_txt(MockUI())
