# 1. 定义输出合并文件的路径
OUTPUT_FILE="/tmp/combined_anime_role_detect.log"
NAMESPACE="anime-role-detect"

# 2. 初始化合并文件并写入当前生成时间
echo "========================================================================" > "$OUTPUT_FILE"
echo "   ARD K8s 汇总日志合并文件 (生成时间: $(date))"
echo "========================================================================" >> "$OUTPUT_FILE"

# 3. 循环遍历所有 Pod，提取日志并追加合并
for pod in $(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'); do
    echo -e "\n\n" >> "$OUTPUT_FILE"
    echo "========================================================================" >> "$OUTPUT_FILE"
    echo ">>>  POD 日志开始: $pod (Namespace: $NAMESPACE)" >> "$OUTPUT_FILE"
    echo "========================================================================" >> "$OUTPUT_FILE"
    
    # 获取该 Pod 中所有容器的前 1000 行日志
    kubectl logs -n "$NAMESPACE" "$pod" --all-containers=true --tail=1000 >> "$OUTPUT_FILE" 2>&1 || \
    echo "【警告】无法获取 $pod 的日志" >> "$OUTPUT_FILE"
    
    echo "------------------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo ">>>  POD 日志结束: $pod" >> "$OUTPUT_FILE"
    echo "------------------------------------------------------------------------" >> "$OUTPUT_FILE"
done

# 4. 打印完成提示
echo "======================================================="
echo "✅ 汇总合并完成！"
echo "合并后的日志文件已保存在: $OUTPUT_FILE"
echo "您可以运行以下命令直接查看或检索该文件："
echo "   cat $OUTPUT_FILE | grep -i 'error'"
echo "======================================================="