# 中石油加油站视频识别

监控视频逐帧处理 GUI：车牌号由车牌服务识别，画面时间由 PaddleOCR 服务识别，结果在同一帧内关联后导出 Excel。

## 克隆

需要递归克隆两个公开上游子模块（模型权重随车牌子模块获取）：

```powershell
git clone --recurse-submodules --shallow-submodules https://github.com/wangkang1019/zhongshiyou.git
Set-Location .\zhongshiyou
Copy-Item .\.env.example .\.env
```

已有克隆若识别项目目录为空，可执行 `git submodule update --init --depth 1`。

## Docker 服务

默认配置为**一个车牌识别容器 + 一个时间 OCR 容器**；GUI 需要两个 API，不能只启动其中一个。没有启动多个车牌副本。首次构建会下载基础镜像和 Python 依赖；首次启动 OCR 还会下载模型并写入 Compose 缓存卷。

从仓库根目录启动：

```powershell
docker compose up --build -d
```

若本机仅安装了独立版 Compose，可使用 `docker-compose up --build -d`。运行前确认 Docker 已启动且 GPU 环境可用。

`.env` 用于匹配目标电脑配置：

- `PLATE_HOST_PORT` / `OCR_HOST_PORT`：主机映射端口；端口变更后也要在 GUI 地址栏填入对应端口。
- `NVIDIA_VISIBLE_DEVICES`：选择可供容器使用的 GPU，默认 `all`。
- `DOCKER_SHM_SIZE`：容器共享内存大小。
- `PLATE_BASE_IMAGE`：车牌服务 PyTorch/CUDA 基础镜像。

默认使用 NVIDIA GPU。若目标机没有 NVIDIA GPU，需要从 `docker-compose.yml` 的两个服务中移除 `gpus: all`，并在 `.env` 设置 `PADDLEOCR_DEVICE=cpu`；CPU 配置仍需在目标机器上验证。Windows 主机路径无需改 Docker 容器内路径，Compose 相对仓库根目录解析上下文。

## 主机端 GUI

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements_main.txt
python .\video_plate_time_gui.py
```

在 GUI 中选择 MP4 文件并检查两个服务地址。输入视频和输出 Excel 均留在主机，不进入镜像或 Git 仓库。

更多模型路径、依赖提交版本及配置说明见 [AGENTS.md](./AGENTS.md)。
