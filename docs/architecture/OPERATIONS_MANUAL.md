# Anime Role Detect 系统运维手册

## 目录

1. [系统概述](#1-系统概述)
2. [日常运维](#2-日常运维)
3. [监控与告警](#3-监控与告警)
4. [故障排查](#4-故障排查)
5. [性能优化](#5-性能优化)
6. [备份与恢复](#6-备份与恢复)
7. [安全维护](#7-安全维护)
8. [升级与更新](#8-升级与更新)

## 1. 系统概述

### 1.1 系统架构

Anime Role Detect 是一个基于深度学习的动漫角色识别系统，采用微服务架构部署在Kubernetes上。

**核心组件**：
- **前端服务**：Next.js应用，提供用户界面
- **后端服务**：FastAPI应用，提供API接口和模型推理
- **监控系统**：Prometheus + Grafana，提供性能监控和告警
- **日志系统**：Loki + Promtail，提供日志收集和分析

**支持平台**：
- Linux/CUDA：NVIDIA GPU环境
- macOS/MPS：Apple Silicon (M1/M2/M3/M4) 环境
- CPU：通用CPU环境

### 1.2 关键指标

**性能指标**：
- API响应时间：< 2秒（95分位）
- 吞吐量：> 10 QPS
- 内存使用：< 4GB（后端）
- CPU使用：< 70%（后端）

**可用性指标**：
- 服务可用性：> 99.9%
- 错误率：< 1%
- 平均恢复时间（MTTR）：< 5分钟

## 2. 日常运维

### 2.1 服务管理

#### 2.1.1 查看服务状态

```bash
# 查看所有Pod状态
kubectl get pods -n default

# 查看后端服务状态
kubectl get pods -l component=backend -n default

# 查看前端服务状态
kubectl get pods -l component=frontend -n default

# 查看服务详情
kubectl describe service character-classification-backend -n default
```

#### 2.1.2 重启服务

```bash
# 重启后端服务
kubectl rollout restart deployment character-classification-backend -n default

# 重启前端服务
kubectl rollout restart deployment character-classification-frontend -n default

# 查看重启状态
kubectl rollout status deployment character-classification-backend -n default
```

#### 2.1.3 扩缩容服务

```bash
# 手动扩容
kubectl scale deployment character-classification-backend --replicas=5 -n default

# 手动缩容
kubectl scale deployment character-classification-backend --replicas=2 -n default

# 查看HPA状态
kubectl get hpa -n default
```

### 2.2 日志管理

#### 2.2.1 查看日志

```bash
# 查看后端日志
kubectl logs -l component=backend -n default --tail=100 -f

# 查看前端日志
kubectl logs -l component=frontend -n default --tail=100 -f

# 查看特定Pod日志
kubectl logs <pod-name> -n default --tail=100 -f

# 查看诊断日志
kubectl logs <pod-name> -n default | grep "崩溃前设备状态快照"

# 查看OOM错误
kubectl logs <pod-name> -n default | grep "OOM异常"
```

#### 2.2.2 日志文件位置

**容器内日志路径**：
- API日志：`/app/logs/api.log`
- 诊断日志：`/app/logs/diagnostics.log`
- 崩溃日志：`/app/logs/crash.log`
- 性能日志：`/app/logs/performance.log`

**本地日志路径**：
```bash
# 复制日志到本地
kubectl cp <pod-name>:/app/logs ./logs -n default
```

### 2.3 资源监控

#### 2.3.1 查看资源使用

```bash
# 查看Pod资源使用
kubectl top pods -n default

# 查看节点资源使用
kubectl top nodes

# 查看资源配额
kubectl describe resourcequota -n default
```

#### 2.3.2 查看事件

```bash
# 查看最近事件
kubectl get events -n default --sort-by='.lastTimestamp'

# 查看特定Pod事件
kubectl describe pod <pod-name> -n default
```

## 3. 监控与告警

### 3.1 Prometheus监控

#### 3.1.1 访问Prometheus

```bash
# 端口转发
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# 访问地址
http://localhost:9090
```

#### 3.1.2 常用查询

```promql
# CPU使用率
avg(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"character-classification-backend.*"}[5m])) * 100

# 内存使用率
avg(container_memory_usage_bytes{namespace="default",pod=~"character-classification-backend.*"} / container_spec_memory_limit_bytes{namespace="default",pod=~"character-classification-backend.*"}) * 100

# 请求速率
rate(http_requests_total{namespace="default",service="character-classification-backend"}[5m])

# 错误率
rate(http_requests_total{namespace="default",service="character-classification-backend",status=~"5.."}[5m]) / rate(http_requests_total{namespace="default",service="character-classification-backend"}[5m])

# 响应时间（95分位）
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{namespace="default",service="character-classification-backend"}[5m]))
```

### 3.2 Grafana仪表板

#### 3.2.1 访问Grafana

```bash
# 端口转发
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# 访问地址
http://localhost:3000

# 默认用户名和密码
用户名: admin
密码: admin
```

#### 3.2.2 导入仪表板

1. 登录Grafana
2. 点击左侧菜单的 "+" -> "Import"
3. 上传 `monitoring/grafana-dashboard.json` 文件
4. 点击 "Import" 完成导入

### 3.3 告警配置

#### 3.3.1 查看告警规则

```bash
# 查看告警规则
kubectl get prometheusrules -n monitoring

# 查看告警状态
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# 访问 http://localhost:9090/alerts
```

#### 3.3.2 告警级别

**Critical（严重）**：
- Pod OOMKilled
- 服务不可用
- Pod频繁重启

**Warning（警告）**：
- CPU使用率 > 70%
- 内存使用率 > 85%
- 响应时间 > 5秒
- 错误率 > 5%

## 4. 故障排查

### 4.1 常见问题

#### 4.1.1 Pod无法启动

**症状**：Pod状态为 `ImagePullBackOff` 或 `CrashLoopBackOff`

**排查步骤**：
```bash
# 查看Pod详情
kubectl describe pod <pod-name> -n default

# 查看Pod日志
kubectl logs <pod-name> -n default

# 查看事件
kubectl get events -n default --field-selector involvedObject.name=<pod-name>
```

**常见原因**：
- 镜像不存在或无法拉取
- 资源不足
- 配置错误
- 健康检查失败

#### 4.1.2 OOM错误

**症状**：Pod被OOMKilled

**排查步骤**：
```bash
# 查看Pod状态
kubectl describe pod <pod-name> -n default

# 查看OOM事件
kubectl get events -n default | grep OOMKilled

# 查看诊断日志
kubectl logs <pod-name> -n default | grep "OOM异常"

# 查看内存快照
kubectl logs <pod-name> -n default | grep "崩溃前设备状态快照"
```

**解决方案**：
1. 增加内存限制
2. 优化模型推理逻辑
3. 启用FP16半精度
4. 限制图像大小

#### 4.1.3 服务响应慢

**症状**：API响应时间过长

**排查步骤**：
```bash
# 查看CPU使用率
kubectl top pods -n default

# 查看内存使用率
kubectl top pods -n default

# 查看网络延迟
kubectl exec -it <pod-name> -n default -- ping google.com

# 查看API响应时间
kubectl logs <pod-name> -n default | grep "processing_time"
```

**解决方案**：
1. 增加副本数
2. 优化模型推理
3. 启用缓存
4. 调整资源配额

### 4.2 跨平台诊断

#### 4.2.1 Linux/CUDA环境

**查看GPU状态**：
```bash
# 进入Pod
kubectl exec -it <pod-name> -n default -- /bin/bash

# 查看GPU信息
nvidia-smi

# 查看GPU内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**诊断OOM**：
```bash
# 查看CUDA内存快照
kubectl logs <pod-name> -n default | grep "gpu_allocated_gb"

# 查看CUDA错误
kubectl logs <pod-name> -n default | grep "CUDA"
```

#### 4.2.2 macOS/MPS环境

**查看系统内存**：
```bash
# 查看系统内存使用
ps aux | grep python

# 查看统一内存使用
sudo memory_pressure
```

**诊断OOM**：
```bash
# 查看MPS状态
kubectl logs <pod-name> -n default | grep "mps_ready"

# 查看系统内存快照
kubectl logs <pod-name> -n default | grep "sys_mem_free_gb"
```

**查看崩溃报告**：
1. 打开"控制台" (Console.app)
2. 查看崩溃报告
3. 查找以 `python3` 开头的 `.ips` 文件
4. 搜索 `Termination Reason: Namespace JETSAM`

### 4.3 紧急恢复

#### 4.3.1 服务不可用

```bash
# 检查Pod状态
kubectl get pods -n default

# 重启服务
kubectl rollout restart deployment character-classification-backend -n default

# 查看重启状态
kubectl rollout status deployment character-classification-backend -n default
```

#### 4.3.2 数据丢失

```bash
# 恢复配置
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# 恢复模型文件
kubectl cp ./models-backup <pod-name>:/app/models -n default
```

## 5. 性能优化

### 5.1 资源优化

#### 5.1.1 调整资源配额

```yaml
# 修改deployment配置
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2"
    memory: "4Gi"
```

#### 5.1.2 调整HPA参数

```yaml
# 修改HPA配置
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 5.2 应用优化

#### 5.2.1 启用FP16半精度

```python
if self.device.type in ['cuda', 'mps']:
    self.model = self.model.half()
    logger.info(f"已开启 FP16 半精度模式，运行于 {self.device}")
```

#### 5.2.2 优化图像处理

```python
# 限制图像大小
if pixel_count > 4000000:
    scale_factor = (4000000 / pixel_count) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = img.resize((new_width, new_height))
```

#### 5.2.3 启用缓存

```python
# 配置缓存
CACHE_SIZE=1000
CACHE_TTL=3600
```

### 5.3 网络优化

#### 5.3.1 启用HTTP/2

```yaml
# Ingress配置
annotations:
  nginx.ingress.kubernetes.io/use-http2: "true"
```

#### 5.3.2 启用压缩

```yaml
# Ingress配置
annotations:
  nginx.ingress.kubernetes.io/enable-compression: "true"
  nginx.ingress.kubernetes.io/compression-types: "text/html,text/css,application/json,application/javascript"
```

## 6. 备份与恢复

### 6.1 配置备份

```bash
# 备份所有配置
kubectl get configmap,secret,deployment,service,ingress,hpa -n default -o yaml > backup-$(date +%Y%m%d).yaml

# 备份特定配置
kubectl get configmap classification-config -n default -o yaml > configmap-backup.yaml
```

### 6.2 数据备份

```bash
# 备份模型文件
kubectl cp <pod-name>:/app/models ./models-backup -n default

# 备份日志文件
kubectl cp <pod-name>:/app/logs ./logs-backup -n default
```

### 6.3 恢复操作

```bash
# 恢复配置
kubectl apply -f backup-$(date +%Y%m%d).yaml

# 恢复模型文件
kubectl cp ./models-backup <pod-name>:/app/models -n default

# 重启服务
kubectl rollout restart deployment character-classification-backend -n default
```

## 7. 安全维护

### 7.1 访问控制

```bash
# 查看RBAC配置
kubectl get role,rolebinding -n default

# 查看ServiceAccount
kubectl get serviceaccount -n default
```

### 7.2 网络安全

```bash
# 查看网络策略
kubectl get networkpolicy -n default

# 查看Ingress配置
kubectl get ingress -n default
```

### 7.3 镜像安全

```bash
# 扫描镜像漏洞
trivy image character-classification-backend:latest

# 更新基础镜像
# 修改Dockerfile中的基础镜像版本
```

## 8. 升级与更新

### 8.1 应用升级

```bash
# 构建新镜像
docker build -t character-classification-backend:v2.0.0 -f Dockerfile.backend .

# 推送镜像
docker push character-classification-backend:v2.0.0

# 更新部署
kubectl set image deployment/character-classification-backend backend=character-classification-backend:v2.0.0 -n default

# 查看升级状态
kubectl rollout status deployment character-classification-backend -n default
```

### 8.2 回滚操作

```bash
# 查看历史版本
kubectl rollout history deployment character-classification-backend -n default

# 回滚到上一版本
kubectl rollout undo deployment character-classification-backend -n default

# 回滚到指定版本
kubectl rollout undo deployment character-classification-backend --to-revision=2 -n default
```

### 8.3 零停机升级

```bash
# 使用滚动更新策略
kubectl patch deployment character-classification-backend -p '{"spec":{"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxUnavailable":0,"maxSurge":1}}}}' -n default

# 监控升级过程
kubectl rollout status deployment character-classification-backend -n default
```

## 附录

### A. 常用命令速查

```bash
# 查看所有资源
kubectl get all -n default

# 查看Pod详情
kubectl describe pod <pod-name> -n default

# 查看日志
kubectl logs -f <pod-name> -n default

# 进入Pod
kubectl exec -it <pod-name> -n default -- /bin/bash

# 端口转发
kubectl port-forward svc/character-classification-backend 8000:8000 -n default

# 重启服务
kubectl rollout restart deployment character-classification-backend -n default

# 扩缩容
kubectl scale deployment character-classification-backend --replicas=5 -n default
```

### B. 环境变量说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| PYTHONFAULTHANDLER | 开启Python崩溃堆栈打印 | 1 |
| PYTHONUNBUFFERED | 实时刷新日志 | 1 |
| PYTORCH_MPS_HIGH_WATERMARK_RATIO | Mac MPS显存回收阈值 | 0.8 |
| ENABLE_MEMORY_MONITOR | 启用内存监控 | true |
| ENABLE_DIAGNOSTICS | 启用诊断日志 | true |
| MEMORY_WARNING_THRESHOLD | 内存警告阈值 | 85 |
| MEMORY_CRITICAL_THRESHOLD | 内存紧急阈值 | 95 |
| GPU_MEMORY_WARNING_THRESHOLD | GPU内存警告阈值 | 85 |
| GPU_MEMORY_CRITICAL_THRESHOLD | GPU内存紧急阈值 | 95 |

### C. 联系方式

**技术支持**：
- 项目地址：https://github.com/caozhaoqi/anime-role-detect
- 问题反馈：https://github.com/caozhaoqi/anime-role-detect/issues

**紧急联系**：
- 系统管理员：admin@example.com
- 运维团队：ops@example.com