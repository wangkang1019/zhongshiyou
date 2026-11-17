# 中石油加油站视频识别项目说明

## 项目目标

本仓库包含监控视频处理 GUI、车牌识别 HTTP 服务和时间 OCR HTTP 服务。GUI 读取用户选择的一个或多个 MP4 视频，按配置处理视频帧；车牌号与画面时间分别由两个服务识别，结果按同一帧关联后写入 Excel。

## 路径约定

- 所有项目路径都相对于仓库根目录；不要写入开发者机器的盘符或绝对路径。
- `video_plate_time_gui.py` 与 `video_plate_time_regions.json` 位于仓库根目录。
- 外部源码以 Git 子模块提供：`Chinese_license_plate_detection_recognition/` 和 `PaddleOCR/`。
- Docker 镜像内的 `/app/...` 是容器路径，可以使用；主机路径必须可移植。
- GUI 对同目录配置文件应使用 `Path(__file__).with_name(...)` 等相对定位方式。

## 获取依赖源码

两个识别项目是公开上游仓库，父仓库以子模块固定版本。首次克隆时使用递归浅克隆：

```powershell
git clone --recurse-submodules --shallow-submodules https://github.com/wangkang1019/zhongshiyou.git
Set-Location .\zhongshiyou
```

如果已经克隆但两个子模块目录为空：

```powershell
git submodule update --init --depth 1
```

上游地址和当前锁定提交：

- PaddleOCR：`https://github.com/PaddlePaddle/PaddleOCR.git`，提交 `211989f046cc1878460f9e65574690c00a127a1a`。
- 车牌识别：`https://github.com/we0091234/Chinese_license_plate_detection_recognition.git`，提交 `670b765c09fff11850a7b1c542787ad66c24ae50`。

### 模型权重

- 车牌服务需要 `Chinese_license_plate_detection_recognition/weights/plate_detect.pt` 和 `weights/plate_rec_color.pth`。这两个文件已包含在锁定的车牌识别上游提交中；正常初始化子模块即可，不要把它们复制到父仓库。若上游子模块未取全，优先重试 `git submodule update --init --depth 1`，并确认以上两个文件存在。
- 时间 OCR 模型不需要手动放入源码目录。首次启动 PaddleOCR 服务时由 PaddleX 下载到容器缓存，Compose 卷 `paddleocr_paddle_cache`、`paddleocr_paddlex_cache` 会保留缓存。首次启动需要网络。
- 不要将 Docker 镜像、视频、识别结果、缓存或生成的模型文件提交到 Git。

## 构建与服务配置

- `docker-compose.yml` 从仓库根目录构建两个镜像：`docker/plate/Dockerfile` 和 `docker/paddleocr/Dockerfile`。
- 默认只启动一个车牌识别实例（主机端口 8002）和一个时间 OCR 实例（主机端口 8868）。这两个服务分别负责车牌与时间识别，GUI 需要同时连接它们；不要再按端口复制多个车牌容器。两者均默认使用 GPU。
- 车牌镜像基于公开的 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`；PaddleOCR 镜像基于 Dockerfile 中固定的 Paddle GPU/CUDA 镜像，并安装 CUDA 12.6 对应的 `paddlepaddle-gpu==3.2.0`。
- 运行前将 `.env.example` 复制为 `.env`。根据师妹电脑修改 `.env` 中的主机端口、GPU 可见设备、共享内存和基础镜像；`.env` 不提交。
- 若主机端口被占用，改 `.env` 的 `PLATE_HOST_PORT` 或 `OCR_HOST_PORT`，并在 GUI 的服务地址框使用相同端口。
- 默认 GPU 设置要求主机已安装 NVIDIA 驱动，并且 Docker Desktop/Engine 可将 GPU 暴露给容器。多卡时在 `.env` 设置 `NVIDIA_VISIBLE_DEVICES`。无 NVIDIA GPU 时，需要在两个服务配置中去掉 `gpus: all`，并把 `PADDLEOCR_DEVICE` 设为 `cpu`；CPU 路径需在目标机器上单独验证，速度会较慢。
- 根据目标电脑可用内存调整 `.env` 中的 `DOCKER_SHM_SIZE`。不要因为机器不同而修改模型目录结构。
- GUI 默认只访问一个车牌服务。若确需增加并行实例，必须同时调整 Compose 服务、GUI 的 `DEFAULT_PLATE_URLS` / worker 配置和端口映射；普通实践环境不需要多实例。

## 主机端 GUI

在仓库根目录的 PowerShell 中创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements_main.txt
python .\video_plate_time_gui.py
```

视频由 GUI 在主机上读取并逐帧发送给服务，不放进 Docker 镜像。区域配置保存在 `video_plate_time_regions.json`，输入视频和输出 Excel 由使用者自行选择。

## 修改约束

- 不要重写或替换两个上游识别项目；需要修复上游代码时，先说明原因并以独立补丁/上游提交方式处理。
- 修改 Dockerfile、Compose、配置或 GUI 后，做与改动匹配的静态检查；实际容器验证需在 Docker 权限和目标硬件可用时进行，并如实报告。
- 不要使用外部编排工作流、生成的 workflow hooks 或 workflow state。
