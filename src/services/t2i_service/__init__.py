#!/usr/bin/env python3
"""t2i_service — 角色图像生成微服务。

独立运行于 t2i-mac venv（torch 2.3.1 + diffusers 0.30.1），
与主后端 .venv（旧 torch，无 diffusers）隔离。由 api_gateway 代理 /api/t2i/*。
"""
