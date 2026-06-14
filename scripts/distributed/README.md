# 分布式采集系统

## 系统架构

```
┌─────────────────┐     ┌─────────────────┐
│    服务端        │     │    客户端        │
│                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ 采集脚本   │  │     │  │ 数据同步   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │     │  ┌─────▼─────┐  │
│  │ 数据打包   │  │     │  │ 数据下载   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │◄────│──►┌─────▼─────┐  │
│  │ REST API  │  │HTTP │   │ 解压存储   │  │
│  └─────┬─────┘  │     │   └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │     │        │        │
│  │  数据库   │  │     │        │        │
│  │ packages.db│ │     │        │        │
│  └───────────┘  │     │        │        │
└─────────────────┘     └─────────────────┘
```

## 目录结构

```
scripts/distributed/
├── collector_server.py  # 服务端
├── collector_client.py  # 客户端
└── README.md           # 使用说明

data/
├── packages.db         # 数据包记录数据库
├── packages/           # 数据包存储目录
│   ├── dataset_*.zip   # 数据包文件
│   └── *.meta.json     # 元数据文件
└── downloaded/         # 客户端下载目录
    ├── client_id.txt   # 客户端ID
    └── downloaded_packages.json # 本地下载记录
```

## 数据库表结构

### packages 表（数据包记录）
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| package_name | TEXT | 数据包名称 |
| created_at | TEXT | 创建时间 |
| file_size | INTEGER | 文件大小 |
| total_chars | INTEGER | 角色数 |
| total_images | INTEGER | 图片数 |
| download_count | INTEGER | 下载次数 |

### client_downloads 表（客户端下载记录）
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| client_id | TEXT | 客户端ID |
| package_name | TEXT | 数据包名称 |
| download_time | TIMESTAMP | 下载时间 |
| success | BOOLEAN | 是否成功 |

## 服务端

### 启动服务

```bash
cd /Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/distributed
python3 collector_server.py --port 5001
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/status | 获取采集状态 |
| POST | /api/start | 启动采集任务 |
| POST | /api/stop | 停止采集任务 |
| GET | /api/package/list | 获取数据包列表 |
| GET | /api/package/latest | 获取最新数据包 |
| GET | /api/package/{name}?client_id={id} | 下载数据包（避免重复下载） |
| POST | /api/package/create | 手动创建数据包 |
| GET | /api/client/packages?client_id={id} | 获取客户端已下载列表 |
| GET | /api/stats | 获取统计信息 |
| GET | /api/health | 健康检查 |

### 使用示例

```bash
# 启动服务
python3 collector_server.py --port 5001

# 启动采集任务
curl -X POST http://localhost:5001/api/start

# 查看状态
curl http://localhost:5001/api/status

# 创建数据包
curl -X POST http://localhost:5001/api/package/create

# 查看数据包列表
curl http://localhost:5001/api/package/list

# 下载数据包（带client_id避免重复）
curl "http://localhost:5001/api/package/dataset_20260614.zip?client_id=client_abc123"

# 查看客户端已下载列表
curl "http://localhost:5001/api/client/packages?client_id=client_abc123"
```

## 客户端

### 启动客户端

```bash
# 持续运行，定时同步
python3 collector_client.py --server http://localhost:5001 --interval 300

# 单次同步
python3 collector_client.py --server http://localhost:5001 --once

# 指定客户端ID
python3 collector_client.py --server http://localhost:5001 --client-id my_client_001

# 指定本地目录
python3 collector_client.py --server http://localhost:5001 --local-dir /path/to/data
```

### 客户端参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --server | 服务端地址 | http://localhost:5000 |
| --interval | 检查间隔(秒) | 300 (5分钟) |
| --once | 只执行一次 | False |
| --local-dir | 本地数据目录 | data/downloaded |
| --client-id | 客户端ID（可选） | 自动生成 |

## 防重复下载机制

### 工作原理

1. **客户端ID生成**：首次运行时自动生成唯一ID，保存到 `client_id.txt`
2. **服务端记录**：数据库记录每个客户端下载过的数据包
3. **下载检查**：
   - 客户端请求时携带 `client_id`
   - 服务端检查数据库是否已下载过
   - 已下载返回 HTTP 409，未下载则允许下载
4. **双向同步**：
   - 服务端记录到 `client_downloads` 表
   - 客户端记录到本地 `downloaded_packages.json`

### 示例流程

```
客户端请求: GET /api/package/dataset_xxx.zip?client_id=client_abc123
服务端检查: SELECT * FROM client_downloads WHERE client_id='client_abc123' AND package_name='dataset_xxx.zip'
结果:
  - 已下载 → HTTP 409 {"error": "已下载过此数据包"}
  - 未下载 → HTTP 200 + 文件流 + INSERT记录到数据库
```

## 数据包格式

数据包为 ZIP 压缩包，包含：
- `角色名/*.jpg` - 图片文件
- `角色名/*.png` - PNG图片

元数据文件 `.meta.json` 包含：
```json
{
  "package_name": "dataset_20260613_120000.zip",
  "created_at": "20260613_120000",
  "file_size": 123456789,
  "stats": {
    "total_chars": 98,
    "total_images": 4237,
    "distribution": {
      "100+": 1,
      "50-99": 31,
      "30-49": 63
    }
  }
}
```

## 部署建议

### 服务端部署
- 建议部署在有稳定网络的机器上
- 可以配置开机自启
- 日志输出到文件便于排查问题
- 数据库文件定期备份

### 客户端部署
- 可以部署在多台机器上
- 每个客户端有唯一ID
- 支持断点续传
- 已下载记录保存在本地

### 飞书通知
- 服务端可发送采集进度通知
- 客户端同步完成会发送通知
- 需要配置 `notification_config.json`
