#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流引擎
支持可视化编排和工作流执行
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field


class WorkflowNode(BaseModel):
    """工作流节点"""
    id: str = Field(description="节点唯一ID")
    skill_id: str = Field(description="技能ID")
    skill_version: str = Field(default="latest", description="技能版本")
    name: str = Field(description="节点名称")
    inputs: Dict[str, str] = Field(default_factory=dict, description="输入参数，支持表达式")
    outputs: Dict[str, str] = Field(default_factory=dict, description="输出映射")
    x: int = Field(default=0, description="画布X坐标")
    y: int = Field(default=0, description="画布Y坐标")
    enabled: bool = Field(default=True, description="是否启用")


class WorkflowEdge(BaseModel):
    """工作流连线"""
    id: str = Field(description="连线唯一ID")
    source: str = Field(description="源节点ID")
    source_output: str = Field(description="源节点输出名称")
    target: str = Field(description="目标节点ID")
    target_input: str = Field(description="目标节点输入名称")


class Workflow(BaseModel):
    """工作流定义"""
    id: str = Field(description="工作流唯一ID")
    name: str = Field(description="工作流名称")
    description: str = Field(default="", description="工作流描述")
    nodes: List[WorkflowNode] = Field(default_factory=list, description="节点列表")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="连线列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    version: str = Field(default="1.0.0", description="工作流版本")
    
    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """根据ID获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


class ExecutionContext(BaseModel):
    """执行上下文"""
    workflow_id: str = Field(description="工作流ID")
    execution_id: str = Field(description="执行实例ID")
    variables: Dict[str, Any] = Field(default_factory=dict, description="全局变量")
    node_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="节点输出缓存")
    started_at: datetime = Field(default_factory=datetime.now, description="开始时间")
    status: str = Field(default="running", description="执行状态")


class ExecutionResult(BaseModel):
    """执行结果"""
    success: bool = Field(description="是否成功")
    message: str = Field(default="", description="结果消息")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    error: Optional[str] = Field(default=None, description="错误信息")


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, workflow_dir: str = None):
        """
        初始化工作流引擎
        
        :param workflow_dir: 工作流存储目录，默认为 ~/.ardc/workflows
        """
        if workflow_dir:
            self.workflow_dir = Path(workflow_dir)
        else:
            self.workflow_dir = Path.home() / ".ardc" / "workflows"
        
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # 技能执行器映射
        self.skill_executors: Dict[str, Callable] = {}
    
    def register_skill_executor(self, skill_id: str, executor: Callable):
        """
        注册技能执行器
        
        :param skill_id: 技能ID
        :param executor: 执行函数
        """
        self.skill_executors[skill_id] = executor
    
    def save_workflow(self, workflow: Workflow) -> bool:
        """
        保存工作流
        
        :param workflow: 工作流对象
        :return: 是否保存成功
        """
        try:
            workflow.updated_at = datetime.now()
            workflow_file = self.workflow_dir / f"{workflow.id}.json"
            
            data = workflow.dict()
            # 处理datetime
            for key in ['created_at', 'updated_at']:
                if isinstance(data.get(key), datetime):
                    data[key] = data[key].isoformat()
            
            with open(workflow_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存工作流失败: {e}")
            return False
    
    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """
        加载工作流
        
        :param workflow_id: 工作流ID
        :return: 工作流对象
        """
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        if not workflow_file.exists():
            return None
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 转换时间字段
                for key in ['created_at', 'updated_at']:
                    if isinstance(data.get(key), str):
                        data[key] = datetime.fromisoformat(data[key])
                
                return Workflow(**data)
        except Exception as e:
            print(f"加载工作流失败: {e}")
            return None
    
    def list_workflows(self) -> List[Workflow]:
        """
        获取所有工作流列表
        
        :return: 工作流列表
        """
        workflows = []
        for workflow_file in self.workflow_dir.glob("*.json"):
            workflow_id = workflow_file.stem
            workflow = self.load_workflow(workflow_id)
            if workflow:
                workflows.append(workflow)
        
        workflows.sort(key=lambda w: w.updated_at, reverse=True)
        return workflows
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        删除工作流
        
        :param workflow_id: 工作流ID
        :return: 是否删除成功
        """
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        if workflow_file.exists():
            workflow_file.unlink()
            return True
        return False
    
    async def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any] = None) -> ExecutionResult:
        """
        执行工作流
        
        :param workflow_id: 工作流ID
        :param inputs: 输入参数
        :return: 执行结果
        """
        workflow = self.load_workflow(workflow_id)
        if not workflow:
            return ExecutionResult(success=False, message=f"工作流 {workflow_id} 不存在")
        
        # 创建执行上下文
        execution_id = str(uuid4())
        context = ExecutionContext(
            workflow_id=workflow_id,
            execution_id=execution_id,
            variables=inputs or {}
        )
        
        try:
            # 构建执行顺序（拓扑排序）
            execution_order = self._build_execution_order(workflow)
            
            # 按顺序执行节点
            for node_id in execution_order:
                node = workflow.get_node(node_id)
                if not node or not node.enabled:
                    continue
                
                # 执行节点
                result = await self._execute_node(node, context)
                
                if not result.success:
                    context.status = "failed"
                    return ExecutionResult(
                        success=False,
                        message=f"节点 {node.name} 执行失败",
                        error=result.error
                    )
                
                # 保存节点输出
                context.node_outputs[node_id] = result.outputs
            
            context.status = "completed"
            return ExecutionResult(
                success=True,
                message="工作流执行成功",
                outputs=context.variables
            )
        
        except Exception as e:
            context.status = "failed"
            return ExecutionResult(
                success=False,
                message=f"工作流执行异常",
                error=str(e)
            )
    
    def _build_execution_order(self, workflow: Workflow) -> List[str]:
        """
        构建执行顺序（拓扑排序）
        
        :param workflow: 工作流对象
        :return: 节点ID执行顺序列表
        """
        # 构建依赖图
        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}
        
        for node in workflow.nodes:
            in_degree[node.id] = 0
            adjacency[node.id] = []
        
        for edge in workflow.edges:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # 拓扑排序
        from collections import deque
        queue = deque()
        result = []
        
        # 初始化入度为0的节点
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有环
        if len(result) != len(workflow.nodes):
            raise ValueError("工作流存在循环依赖")
        
        return result
    
    async def _execute_node(self, node: WorkflowNode, context: ExecutionContext) -> ExecutionResult:
        """
        执行单个节点
        
        :param node: 节点对象
        :param context: 执行上下文
        :return: 执行结果
        """
        # 解析输入参数（支持表达式）
        resolved_inputs = {}
        for key, value in node.inputs.items():
            resolved_inputs[key] = self._resolve_expression(value, context)
        
        # 查找并执行技能
        executor = self.skill_executors.get(node.skill_id)
        if executor:
            # 调用注册的执行器
            try:
                result = await executor(**resolved_inputs)
                if isinstance(result, dict):
                    return ExecutionResult(success=True, outputs=result)
                return ExecutionResult(success=True, outputs={"result": result})
            except Exception as e:
                return ExecutionResult(success=False, error=str(e))
        
        # 默认执行逻辑（模拟执行）
        print(f"执行节点: {node.name} (技能: {node.skill_id})")
        print(f"输入参数: {resolved_inputs}")
        
        # 模拟执行结果
        outputs = {
            "success": True,
            "node_id": node.id,
            "skill_id": node.skill_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # 将输出添加到上下文变量
        for output_name, var_name in node.outputs.items():
            if output_name in outputs:
                context.variables[var_name] = outputs[output_name]
        
        return ExecutionResult(success=True, outputs=outputs)
    
    def _resolve_expression(self, expression: str, context: ExecutionContext) -> Any:
        """
        解析表达式
        
        支持的表达式格式:
        - ${variable} - 引用全局变量
        - ${node_id.output_name} - 引用其他节点的输出
        - 直接值 - 返回原值
        
        :param expression: 表达式字符串
        :param context: 执行上下文
        :return: 解析后的值
        """
        if not expression:
            return expression
        
        # 检查是否为变量引用
        if expression.startswith("${") and expression.endswith("}"):
            var_path = expression[2:-1]
            
            # 检查是否为节点输出引用
            if "." in var_path:
                parts = var_path.split(".", 1)
                node_id = parts[0]
                output_name = parts[1]
                
                if node_id in context.node_outputs:
                    return context.node_outputs[node_id].get(output_name)
                return None
            
            # 全局变量引用
            return context.variables.get(var_path)
        
        # 尝试解析为JSON
        try:
            return json.loads(expression)
        except:
            pass
        
        # 返回原值
        return expression
    
    def validate_workflow(self, workflow: Workflow) -> List[str]:
        """
        验证工作流
        
        :param workflow: 工作流对象
        :return: 错误信息列表
        """
        errors = []
        
        # 检查节点是否存在重复ID
        node_ids = []
        for node in workflow.nodes:
            if node.id in node_ids:
                errors.append(f"节点ID重复: {node.id}")
            node_ids.append(node.id)
        
        # 检查连线引用的节点是否存在
        for edge in workflow.edges:
            if edge.source not in node_ids:
                errors.append(f"连线引用不存在的源节点: {edge.source}")
            if edge.target not in node_ids:
                errors.append(f"连线引用不存在的目标节点: {edge.target}")
        
        # 检查是否有循环依赖
        try:
            self._build_execution_order(workflow)
        except ValueError as e:
            errors.append(str(e))
        
        return errors
    
    def export_workflow(self, workflow_id: str) -> Optional[str]:
        """
        导出工作流为JSON字符串
        
        :param workflow_id: 工作流ID
        :return: JSON字符串
        """
        workflow = self.load_workflow(workflow_id)
        if not workflow:
            return None
        
        data = workflow.dict()
        for key in ['created_at', 'updated_at']:
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def import_workflow(self, json_str: str) -> bool:
        """
        从JSON字符串导入工作流
        
        :param json_str: JSON字符串
        :return: 是否导入成功
        """
        try:
            data = json.loads(json_str)
            
            for key in ['created_at', 'updated_at']:
                if isinstance(data.get(key), str):
                    data[key] = datetime.fromisoformat(data[key])
            
            workflow = Workflow(**data)
            return self.save_workflow(workflow)
        except Exception as e:
            print(f"导入工作流失败: {e}")
            return False