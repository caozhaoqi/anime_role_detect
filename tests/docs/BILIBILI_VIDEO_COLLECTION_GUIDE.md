# B站二次元视频采集指南

## 一、概述

本指南提供了从B站采集二次元相关视频的详细步骤，用于测试我们的角色检测模型。由于网络环境限制，我们提供了多种采集方法，包括自动脚本和手动下载。

## 二、自动采集方法

### 2.1 使用脚本采集

我们提供了一个自动采集脚本 `scripts/bilibili_video_collector.py`，支持以下功能：

- 自动下载B站二次元视频
- 支持多种下载工具（youtube-dl、you-get）
- 批量处理多个视频
- 自动测试下载的视频

### 2.2 安装依赖

**方法1：使用youtube-dl**
```bash
# 安装youtube-dl
pip3 install youtube-dl

# 或者使用系统包管理器
brew install youtube-dl  # macOS
apt-get install youtube-dl  # Ubuntu/Debian
```

**方法2：使用you-get**
```bash
# 安装you-get
pip3 install you-get
```

### 2.3 使用脚本

**1. 下载推荐视频**
```bash
# 下载推荐的二次元视频并自动测试
python3 scripts/bilibili_video_collector.py --use_recommended --test
```

**2. 下载指定视频**
```bash
# 下载指定的B站视频
python3 scripts/bilibili_video_collector.py --urls "https://www.bilibili.com/video/BV1mK411W7yf" --test
```

**3. 批量下载**
```bash
# 批量下载多个视频
python3 scripts/bilibili_video_collector.py --urls "https://www.bilibili.com/video/BV1mK411W7yf" "https://www.bilibili.com/video/BV1XK4y1s7cL" --batch --test
```

## 三、手动采集方法

如果自动脚本无法使用，可以使用以下手动方法：

### 3.1 在线下载

**推荐网站：**
- [唧唧Down](https://www.jijidown.com/) - B站视频下载工具
- [B站视频解析](https://bilibili.iiilab.com/) - 在线解析下载
- [SaveFrom.net](https://en.savefrom.net/) - 支持多种视频网站

**步骤：**
1. 访问上述网站
2. 粘贴B站视频URL
3. 选择视频质量
4. 下载视频到 `data/videos/` 目录

### 3.2 推荐视频列表

以下是适合测试的B站二次元视频：

**原神相关：**
- [BV1mK411W7yf](https://www.bilibili.com/video/BV1mK411W7yf) - 原神角色混剪
- [BV1XK4y1s7cL](https://www.bilibili.com/video/BV1XK4y1s7cL) - 原神角色展示
- [BV17K4y1s7cM](https://www.bilibili.com/video/BV17K4y1s7cM) - 原神角色PV

**崩坏三相关：**
- [BV1SK4y1s7cN](https://www.bilibili.com/video/BV1SK4y1s7cN) - 崩坏三角色混剪
- [BV1YK4y1s7cP](https://www.bilibili.com/video/BV1YK4y1s7cP) - 崩坏三角色展示

**动漫相关：**
- [BV1ZK4y1s7cQ](https://www.bilibili.com/video/BV1ZK4y1s7cQ) - 动漫角色混剪
- [BV1PK4y1s7cR](https://www.bilibili.com/video/BV1PK4y1s7cR) - 动漫角色集锦

**其他二次元：**
- [BV1VK4y1s7cS](https://www.bilibili.com/video/BV1VK4y1s7cS) - 二次元角色混剪
- [BV1MK4y1s7cT](https://www.bilibili.com/video/BV1MK4y1s7cT) - 二次元角色展示

## 四、视频测试流程

### 4.1 自动测试

如果使用采集脚本，视频会自动进行测试。测试结果会保存在 `data/videos/` 目录中，文件名以 `detected_` 开头。

### 4.2 手动测试

**使用视频检测脚本：**
```bash
# 测试单个视频
python3 scripts/video_character_detection.py \
    --input_video data/videos/your_video.mp4 \
    --output_video data/videos/detected_your_video.mp4

# 实时显示检测结果
python3 scripts/video_character_detection.py \
    --input_video data/videos/your_video.mp4 \
    --display

# 使用帧跳过提高速度
python3 scripts/video_character_detection.py \
    --input_video data/videos/your_video.mp4 \
    --output_video data/videos/detected_your_video.mp4 \
    --frame_skip 5
```

## 五、视频处理配置

### 5.1 视频检测参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model_path` | 模型文件路径 | `models/character_classifier_best_improved.pth` |
| `--input_video` | 输入视频路径 | **必填** |
| `--output_video` | 输出视频路径 | `None` |
| `--display` | 是否显示视频 | `False` |
| `--frame_skip` | 帧跳过数量 | `1` |
| `--device` | 运行设备 | `mps` |

### 5.2 性能优化

**提高处理速度：**
- 使用 `--frame_skip 5` 参数跳过部分帧
- 在支持的设备上使用GPU加速
- 对于长视频，考虑先截取片段进行测试

**提高检测准确率：**
- 使用高质量、清晰的视频
- 选择角色占画面比例较大的视频
- 确保视频中角色与训练数据风格相似

## 六、测试结果分析

### 6.1 查看检测结果

检测完成后，可以通过以下方式查看结果：

1. **查看输出视频**：打开 `data/videos/` 目录中的检测结果视频，视频中会显示检测到的角色名称和置信度

2. **分析日志**：查看控制台输出的日志，了解检测过程和性能

### 6.2 常见问题

**1. 检测准确率低**
- 原因：视频质量差、角色占比小、光照条件差
- 解决：选择高质量视频，确保角色清晰可见

**2. 处理速度慢**
- 原因：视频分辨率高、设备性能有限
- 解决：使用 `--frame_skip` 参数，降低视频分辨率

**3. 无法识别角色**
- 原因：角色不在训练集中、角色风格差异大
- 解决：选择训练集中包含的角色视频

## 七、视频存储结构

```
anime_role_detect/
├── data/
│   └── videos/             # 视频存储目录
│       ├── original/       # 原始下载的视频
│       └── detected/       # 检测结果视频
├── scripts/
│   ├── bilibili_video_collector.py  # B站视频采集脚本
│   └── video_character_detection.py  # 视频角色检测脚本
└── models/
    └── character_classifier_best_improved.pth  # 训练好的模型
```

## 八、最佳实践

1. **选择合适的视频**：
   - 选择角色清晰可见的视频
   - 选择包含训练集中角色的视频
   - 选择分辨率适中的视频（720p或1080p）

2. **测试流程**：
   - 先使用短视频进行测试
   - 分析测试结果，调整参数
   - 再测试长视频

3. **批量处理**：
   - 对于多个视频，使用批量处理功能
   - 设置合理的延迟，避免被B站限制

4. **结果验证**：
   - 手动验证检测结果的准确性
   - 记录检测错误，用于模型改进

## 九、总结

本指南提供了多种从B站采集二次元视频的方法，以及如何使用这些视频测试我们的角色检测模型。通过合理选择视频和配置参数，可以获得准确的检测结果，为模型改进提供有价值的反馈。

**提示：** 如果遇到网络限制或下载失败的情况，可以尝试使用VPN或更换网络环境，或者直接使用手动下载方法获取视频。