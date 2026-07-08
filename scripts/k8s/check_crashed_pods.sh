#!/bin/bash
set -e

NAMESPACE="anime-role-detect"

echo "========================================"
echo "  K8s 崩溃 Pod 日志诊断脚本"
echo "========================================"
echo ""

echo "📊 所有 Pod 状态:"
kubectl -n $NAMESPACE get pods -o wide
echo ""

echo "🚨 筛选出非 Running 或重启次数 > 0 的 Pod:"
echo ""

PODS=$(kubectl -n $NAMESPACE get pods -o name)
for pod in $PODS; do
    status=$(kubectl -n $NAMESPACE get $pod -o jsonpath='{.status.phase}')
    restarts=$(kubectl -n $NAMESPACE get $pod -o jsonpath='{.status.containerStatuses[0].restartCount}')
    reason=$(kubectl -n $NAMESPACE get $pod -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}' 2>/dev/null || true)
    
    if [[ "$status" != "Running" || "$restarts" -gt 0 ]]; then
        pod_name=$(echo "$pod" | sed 's/pod\///')
        echo "----------------------------------------"
        echo "📝 Pod: $pod_name"
        echo "   Status: $status"
        echo "   Restarts: $restarts"
        echo "   Termination Reason: $reason"
        echo ""
        echo "📜 最近日志 (最后 30 行):"
        kubectl -n $NAMESPACE logs "$pod_name" --tail=30 2>/dev/null || echo "❌ 无法获取日志"
        echo ""
    fi
done

echo "========================================"
echo "  诊断完成"
echo "========================================"