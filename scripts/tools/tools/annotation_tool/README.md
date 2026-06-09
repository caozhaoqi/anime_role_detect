# 动漫角色标注工具

一个简单易用的动漫角色图片标注工具，支持批量标注、多角色识别、R18检测。

## 版本说明

### 桌面版本 (PyQt5) - 推荐
- 完全独立运行，无需浏览器
- 支持 Windows、Mac、Linux
- 直接显示本地图片，加载更快
- 跨平台原生界面体验

### 网页版本 (FastAPI)
- 通过浏览器访问
- 适合远程协作
- 需要启动Web服务器

## 安装

### 桌面版本
```bash
pip install -r requirements_desktop.txt
python annotation_tool_desktop.py
```

### 网页版本
```bash
pip install -r requirements_web.txt
python main.py
```

然后访问 http://localhost:8090

## 功能特性

1. **图片扫描** - 读取指定目录下的所有图片文件
2. **批量标注** - 支持单角色和多角色图片标注
3. **角色管理** - 添加、编辑、批量导入角色
4. **R18检测** - 标记和移动可疑图片
5. **导出功能** - 支持JSON和CSV格式导出
6. **进度统计** - 实时显示标注进度

## 快捷键

- **← / →** - 上一个/下一个图片
- **Ctrl+O** - 打开目录
- **Ctrl+E** - 导出数据
- **Ctrl+Q** - 退出程序

## 数据存储

- 标注目录: `data/annotations/`
- 角色文件: `data/roles.json`
- 可疑图片: `data/nsfw_suspicious/`

## 导出格式

### JSON
```json
{
  "image_path": "/path/to/image.jpg",
  "roles": ["Anya", "Bond"],
  "is_multi_role": true,
  "is_nsfw": false,
  "notes": "",
  "annotator": "anonymous",
  "timestamp": "2024-01-01T12:00:00"
}
```

### CSV
```
image_path,roles,is_multi_role,is_nsfw,notes,annotator,timestamp
```

## 系统要求

### 桌面版本
- Python 3.8+
- PyQt5 5.15+
- Windows/Mac/Linux

### 网页版本
- Python 3.8+
- FastAPI
- 现代浏览器

## 截图预览

工具界面包含：
- 左侧栏：目录选择、统计信息、角色列表
- 右侧：图片预览、标注控制、导航按钮
- 底部：状态栏、进度条

## 许可证

MIT License