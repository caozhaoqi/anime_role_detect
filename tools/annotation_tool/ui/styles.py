"""样式常量和样式表生成模块"""

# 现代化样式常量
STYLES = {
    "primary_color": "#6366f1",
    "primary_light": "#818cf8",
    "primary_dark": "#4f46e5",
    "secondary_color": "#8b5cf6",
    "success_color": "#10b981",
    "warning_color": "#f59e0b",
    "danger_color": "#ef4444",
    "info_color": "#3b82f6",
    
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "bg_card_hover": "#334155",
    "bg_input": "#334155",
    "bg_button": "#334155",
    
    "border_color": "#475569",
    "border_radius": "8px",
    "border_radius_large": "12px",
    
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
}

def get_style_sheet():
    """生成完整的样式表"""
    return f"""
    QMainWindow {{ background-color: {STYLES['bg_dark']}; }}
    
    QMenuBar {{
        background-color: {STYLES['bg_card']};
        color: {STYLES['text_primary']};
        border-bottom: 1px solid {STYLES['border_color']};
    }}
    QMenuBar::item {{ padding: 8px 16px; margin: 0; }}
    QMenuBar::item:hover {{ background-color: {STYLES['bg_card_hover']}; }}
    
    QMenu {{
        background-color: {STYLES['bg_card']};
        color: {STYLES['text_primary']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius']};
    }}
    QMenu::item {{ padding: 10px 24px; border-radius: 4px; }}
    QMenu::item:hover {{ background-color: {STYLES['bg_card_hover']}; }}
    
    QToolBar {{
        background: linear-gradient(180deg, {STYLES['bg_card']} 0%, {STYLES['bg_dark']} 100%);
        border-bottom: 1px solid {STYLES['border_color']};
        spacing: 12px;
        padding: 10px 16px;
    }}
    
    QPushButton {{
        background-color: {STYLES['bg_button']};
        color: {STYLES['text_primary']};
        border: none;
        border-radius: {STYLES['border_radius']};
        padding: 8px 16px;
        font-size: 13px;
        min-width: 60px;
    }}
    QPushButton:hover {{ background-color: {STYLES['bg_card_hover']}; }}
    QPushButton:pressed {{ background-color: {STYLES['border_color']}; }}
    
    QLabel {{ color: {STYLES['text_primary']}; }}
    
    QLineEdit {{
        background-color: {STYLES['bg_input']};
        color: {STYLES['text_primary']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius']};
        padding: 8px 12px;
        font-size: 13px;
    }}
    QLineEdit:focus {{ border-color: {STYLES['primary_color']}; outline: none; }}
    
    QListWidget {{
        background-color: {STYLES['bg_input']};
        color: {STYLES['text_primary']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius']};
        padding: 4px;
    }}
    QListWidget::item {{ padding: 8px 12px; border-radius: 4px; }}
    QListWidget::item:hover {{ background-color: {STYLES['bg_card_hover']}; }}
    QListWidget::item:selected {{ background-color: {STYLES['primary_color']}; color: white; }}
    
    QCheckBox {{ color: {STYLES['text_primary']}; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px;
        border-radius: 4px;
        background-color: {STYLES['bg_input']};
        border: 1px solid {STYLES['border_color']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {STYLES['primary_color']};
        border-color: {STYLES['primary_color']};
    }}
    
    QComboBox {{
        background-color: {STYLES['bg_input']};
        color: {STYLES['text_primary']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius']};
        padding: 6px 30px 6px 12px;
        min-width: 100px;
    }}
    
    QSlider::groove:horizontal {{
        height: 6px; background-color: {STYLES['bg_input']}; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 16px; height: 16px;
        background-color: {STYLES['primary_color']};
        border-radius: 50%; margin: -5px 0;
    }}
    
    QScrollBar:vertical {{
        background-color: {STYLES['bg_input']}; width: 8px; border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {STYLES['border_color']}; border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {STYLES['text_muted']}; }}
    
    QGroupBox {{
        background-color: {STYLES['bg_card']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius_large']};
        padding: 12px; margin: 8px;
    }}
    QGroupBox::title {{
        color: {STYLES['text_secondary']};
        font-size: 12px; font-weight: 600;
        padding: 0 8px; margin-top: -8px;
    }}
    
    QStatusBar {{
        background-color: {STYLES['bg_card']};
        color: {STYLES['text_secondary']};
        border-top: 1px solid {STYLES['border_color']};
        padding: 0 12px;
    }}
    
    QTextEdit {{
        background-color: {STYLES['bg_input']};
        color: {STYLES['text_primary']};
        border: 1px solid {STYLES['border_color']};
        border-radius: {STYLES['border_radius']};
        padding: 8px; font-size: 12px;
    }}
    
    QSplitter {{ background-color: {STYLES['bg_dark']}; }}
    QSplitter::handle {{
        background-color: {STYLES['border_color']};
        width: 6px; height: 6px;
    }}
    QSplitter::handle:hover {{ background-color: {STYLES['text_muted']}; }}
    """

def get_button_style(color, hover_color=None):
    """生成按钮样式"""
    if hover_color is None:
        hover_color = color
    return f"""
        QPushButton {{
            background-color: {color}; color: white;
            padding: 10px 16px; font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {hover_color}; }}
    """

def get_gradient_button_style(start_color, end_color, hover_start=None, hover_end=None):
    """生成渐变按钮样式"""
    if hover_start is None:
        hover_start = start_color
    if hover_end is None:
        hover_end = end_color
    return f"""
        QPushButton {{
            background: linear-gradient(135deg, {start_color} 0%, {end_color} 100%);
            color: white; padding: 10px 24px; font-weight: 600;
        }}
        QPushButton:hover {{
            background: linear-gradient(135deg, {hover_start} 0%, {hover_end} 100%);
        }}
    """
