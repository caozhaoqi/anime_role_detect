.PHONY: all install run-dev run-prod build-web deploy

all: install

install:
	@echo "安装后端依赖..."
	pip install -r requirements.txt
	pip install -e .

	@echo "安装前端依赖..."
	cd web && npm install

run-dev:
	@echo "启动开发模式..."
	@echo "后端服务: http://localhost:8000"
	@echo "前端服务: http://localhost:5173"
	uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --reload &
	cd web && npm run dev

run-prod:
	@echo "启动生产模式..."
	uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4

build-web:
	@echo "构建前端..."
	cd web && npm run build

deploy: install build-web
	@echo "部署完成!"
