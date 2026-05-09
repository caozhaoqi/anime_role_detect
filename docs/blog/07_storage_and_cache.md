# 【技术难点】数据持久化与缓存层设计

> 在生产级系统中，高效的数据存储和缓存策略是保证系统性能和可扩展性的关键。

---

## 🔍 问题背景

系统需要处理以下数据需求：

| 数据类型 | 特点 | 存储需求 |
|---------|------|---------|
| 识别记录 | 写多读少，需要持久化 | 关系型数据库 |
| 图片哈希 | 读多写少，查询频繁 | 缓存层 |
| 用户会话 | 临时数据，快速访问 | 缓存层 |
| 模型结果 | 计算密集，可复用 | 缓存层 |

**核心挑战**：如何设计高效的存储架构，避免重复计算，提升响应速度？

---

## 💡 解决方案：分层存储架构

### Redis 缓存层

```python
import redis
import hashlib
import json
from typing import Optional, Any

class CacheManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def get_image_hash(self, image_path: str) -> str:
        """计算图片的 MD5 哈希值"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_cached_result(self, image_hash: str) -> Optional[dict]:
        """从缓存获取识别结果"""
        result = self.client.get(f"result:{image_hash}")
        if result:
            return json.loads(result)
        return None
    
    def set_cached_result(self, image_hash: str, result: dict, ttl: int = 86400):
        """将识别结果存入缓存（默认有效期1天）"""
        self.client.setex(f"result:{image_hash}", ttl, json.dumps(result))
    
    def is_cached(self, image_hash: str) -> bool:
        """检查图片是否已缓存"""
        return self.client.exists(f"result:{image_hash}") > 0
    
    def get_user_session(self, session_id: str) -> Optional[dict]:
        """获取用户会话"""
        session = self.client.get(f"session:{session_id}")
        if session:
            return json.loads(session)
        return None
    
    def set_user_session(self, session_id: str, data: dict, ttl: int = 3600):
        """设置用户会话（默认有效期1小时）"""
        self.client.setex(f"session:{session_id}", ttl, json.dumps(data))
    
    def increment_request_count(self, user_id: str):
        """增加用户请求计数"""
        self.client.incr(f"count:{user_id}")
    
    def get_request_count(self, user_id: str) -> int:
        """获取用户请求计数"""
        count = self.client.get(f"count:{user_id}")
        return int(count) if count else 0
```

### PostgreSQL 数据库层

#### 表结构设计

```python
import psycopg2
from psycopg2 import sql
from datetime import datetime
from typing import List, Dict

class DatabaseManager:
    def __init__(self, host='localhost', port=5432, dbname='anime_db', user='postgres', password='password'):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """创建必要的表"""
        # 识别记录表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(50),
                image_hash VARCHAR(32) UNIQUE,
                prediction JSONB,
                confidence FLOAT,
                model_name VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 用户表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password_hash VARCHAR(255),
                email VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        self.conn.commit()
    
    def insert_recognition(self, user_id: str, image_hash: str, prediction: dict, confidence: float, model_name: str):
        """插入识别记录"""
        try:
            self.cursor.execute("""
                INSERT INTO recognition_history (user_id, image_hash, prediction, confidence, model_name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (image_hash) DO UPDATE SET 
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    model_name = EXCLUDED.model_name
            """, (user_id, image_hash, json.dumps(prediction), confidence, model_name))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"❌ 插入记录失败: {e}")
            return False
    
    def get_user_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """获取用户识别历史"""
        self.cursor.execute("""
            SELECT * FROM recognition_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """, (user_id, limit))
        
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'user_id': row[1],
                'image_hash': row[2],
                'prediction': json.loads(row[3]) if row[3] else None,
                'confidence': row[4],
                'model_name': row[5],
                'created_at': row[6]
            })
        return results
    
    def get_statistics(self, days: int = 30) -> Dict:
        """获取统计信息"""
        # 总识别次数
        self.cursor.execute("SELECT COUNT(*) FROM recognition_history")
        total_count = self.cursor.fetchone()[0]
        
        # 最近N天的识别次数
        self.cursor.execute("""
            SELECT COUNT(*) FROM recognition_history 
            WHERE created_at >= NOW() - INTERVAL %s
        """, (f"{days} days",))
        recent_count = self.cursor.fetchone()[0]
        
        # 各模型使用次数
        self.cursor.execute("""
            SELECT model_name, COUNT(*) as count 
            FROM recognition_history 
            GROUP BY model_name 
            ORDER BY count DESC
        """)
        model_stats = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        return {
            'total_recognitions': total_count,
            'recent_recognitions': recent_count,
            'model_usage': model_stats
        }
```

### Faiss 向量索引（以图搜图）

```python
import faiss
import numpy as np
import os

class FaissIndexManager:
    def __init__(self, index_path='faiss_index.bin'):
        self.index_path = index_path
        self.index = None
        self.feature_dim = 1280  # EfficientNet-B3 特征维度
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载已存在的索引或创建新索引"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ 已加载索引，包含 {self.index.ntotal} 个向量")
        else:
            # 创建 IVF 索引（倒排文件索引，适合大规模数据）
            nlist = 100  # 聚类中心数量
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.feature_dim),
                self.feature_dim,
                nlist,
                faiss.METRIC_L2
            )
            print("✅ 创建新的 Faiss 索引")
    
    def add_features(self, features: np.ndarray, image_hashes: List[str]):
        """添加特征向量到索引"""
        if not self.index.is_trained:
            # 使用部分数据训练索引
            self.index.train(features)
        
        # 添加特征
        self.index.add(features)
        
        # 保存索引
        faiss.write_index(self.index, self.index_path)
        
        # 保存哈希映射
        self._save_hash_mapping(image_hashes)
        
        print(f"✅ 已添加 {len(image_hashes)} 个特征")
    
    def search_similar(self, query_feature: np.ndarray, k: int = 5) -> List[dict]:
        """搜索相似图片"""
        distances, indices = self.index.search(query_feature.reshape(1, -1), k)
        
        # 读取哈希映射
        hash_mapping = self._load_hash_mapping()
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            distance = distances[0][i]
            
            if idx != -1:  # -1 表示未找到
                results.append({
                    'image_hash': hash_mapping.get(idx, 'unknown'),
                    'distance': float(distance),
                    'similarity': self._distance_to_similarity(distance)
                })
        
        return results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """将距离转换为相似度（0-1）"""
        max_distance = 100.0  # 经验值
        return max(0, 1 - distance / max_distance)
    
    def _save_hash_mapping(self, image_hashes: List[str]):
        """保存索引到哈希的映射"""
        mapping_path = self.index_path + '.mapping'
        np.save(mapping_path, image_hashes)
    
    def _load_hash_mapping(self) -> dict:
        """加载索引到哈希的映射"""
        mapping_path = self.index_path + '.mapping'
        if os.path.exists(mapping_path):
            hashes = np.load(mapping_path, allow_pickle=True).tolist()
            return {i: h for i, h in enumerate(hashes)}
        return {}

# 使用示例
# faiss_manager = FaissIndexManager()
# faiss_manager.add_features(features_array, image_hashes_list)
# results = faiss_manager.search_similar(query_feature)
```

#### 向量索引流程图

```
提取图像特征
    ↓
计算特征向量 (1280维)
    ↓
┌──────────────────────┐
│ 检查 Faiss 索引      │
└──────────────────────┘
    ↓
┌──────────────┐    否     ┌──────────────────┐
│ 索引已训练?  │────────→│ 训练聚类中心      │
└──────────────┘          └──────────────────┘
    ↓ 是                      ↓
┌──────────────────────┐   ┌──────────────────┐
│ 添加特征到索引        │   │ 添加特征到索引    │
└──────────────────────┘   └──────────────────┘
    ↓
保存索引文件
```

---

## 🚀 使用示例

### 缓存优先的识别流程

```python
def recognize_with_cache(cache: CacheManager, db: DatabaseManager, image_path: str, model, user_id: str = None):
    """带缓存的识别流程"""
    
    # 1. 计算图片哈希
    image_hash = cache.get_image_hash(image_path)
    
    # 2. 检查缓存
    cached_result = cache.get_cached_result(image_hash)
    if cached_result:
        print("✅ 命中缓存")
        return cached_result
    
    # 3. 执行识别
    print("🔄 执行模型推理")
    processor = ImageProcessor(input_size=224)
    tensor = processor.preprocess(image_path)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        top1 = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, top1].item()
    
    result = {
        'prediction': top1,
        'confidence': confidence,
        'model_name': 'efficientnet-b3',
        'image_hash': image_hash,
        'timestamp': datetime.now().isoformat()
    }
    
    # 4. 更新缓存
    cache.set_cached_result(image_hash, result)
    
    # 5. 保存到数据库
    if user_id:
        db.insert_recognition(user_id, image_hash, {'prediction': top1}, confidence, 'efficientnet-b3')
    
    return result

# 使用示例
cache = CacheManager()
db = DatabaseManager()
db.create_tables()

result = recognize_with_cache(cache, db, "data/test.jpg", model, user_id="user123")
print(f"识别结果: {result}")
```

### 缓存流程图

```
用户上传图片
    ↓
计算图片 MD5 哈希
    ↓
┌──────────────────────┐
│ 检查 Redis 缓存      │
└──────────────────────┘
    ↓
┌──────────┐    否     ┌──────────────────┐
│ 缓存命中?│────────→│ 执行模型推理      │
└──────────┘          └──────────────────┘
    ↓ 是                  ↓
┌──────────────────┐   ┌──────────────────┐
│ 返回缓存结果      │   │ 更新 Redis 缓存   │
└──────────────────┘   └──────────────────┘
                              ↓
                       ┌──────────────────┐
                       │ 保存到 PostgreSQL│
                       └──────────────────┘
```

---

## ⚡ 缓存策略配置

```python
# 缓存配置
CACHE_CONFIG = {
    # 识别结果缓存（1天）
    'result_ttl': 86400,
    
    # 用户会话缓存（1小时）
    'session_ttl': 3600,
    
    # 热门角色特征缓存（1周）
    'feature_ttl': 604800,
    
    # 缓存键前缀
    'prefix': {
        'result': 'result:',
        'session': 'session:',
        'count': 'count:',
        'feature': 'feature:'
    }
}

# 数据库配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'anime_db',
    'user': 'postgres',
    'password': 'your_password',
    'max_connections': 20
}
```

---

## 📝 关键要点

1. **缓存优先**：先查缓存，再查数据库，最后计算
2. **哈希去重**：使用 MD5 哈希值作为缓存键，避免重复计算
3. **TTL 策略**：根据数据类型设置不同的过期时间
4. **数据持久化**：重要数据（如用户历史）存入数据库
5. **读写分离**：缓存层处理高频读操作，数据库处理写操作
6. **并发控制**：使用 Redis 的原子操作避免竞态条件

---

*下篇预告：Docker Compose 部署*
