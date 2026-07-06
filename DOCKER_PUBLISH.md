# Docker Image 发布说明

本项目推荐把源码和镜像分开发布：

- GitHub 仓库：源码、Dockerfile、Compose、启动文档。
- GHCR / Docker Hub：预构建 Docker 镜像。
- Google Drive：可选的离线镜像包备用下载。

镜像不包含模型权重。用户需要自行下载权重，并通过 `HOST_MODEL_ROOT` 挂载。

## 1. 确认本地镜像

当前本机已经构建好的 app 镜像名通常是：

```bash
docker image ls lychee-fd:dev
```

如果需要重新构建：

```bash
docker compose -f compose.yaml -f compose.build.yaml build frontend
```

## 2. 准备 GHCR Token

在 GitHub 创建 Personal Access Token：

1. 打开 GitHub `Settings -> Developer settings -> Personal access tokens`。
2. 创建 token。
3. 勾选 `write:packages` 和 `read:packages`。
4. 保存 token。后面 `docker login ghcr.io` 时把它当密码粘贴。

## 3. 打 Tag

把下面变量替换成你的 GitHub 用户名或组织名。镜像名建议全部小写。

```bash
export GHCR_OWNER=<your-github-username>
export IMAGE_NAME=lychee-fd
export VERSION=v0.1.0
```

打 tag：

```bash
docker tag lychee-fd:dev ghcr.io/${GHCR_OWNER}/${IMAGE_NAME}:latest
docker tag lychee-fd:dev ghcr.io/${GHCR_OWNER}/${IMAGE_NAME}:${VERSION}
```

## 4. 登录并推送

```bash
docker login ghcr.io -u ${GHCR_OWNER}
docker push ghcr.io/${GHCR_OWNER}/${IMAGE_NAME}:latest
docker push ghcr.io/${GHCR_OWNER}/${IMAGE_NAME}:${VERSION}
```

如果 GHCR package 默认是 private，推送后到 GitHub Packages 页面把 package visibility 改成 public。

## 5. 更新仓库配置

把 `.env.docker.example` 中的：

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/your-github-username/lychee-fd:latest
```

改成真实地址，例如：

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/${GHCR_OWNER}/lychee-fd:latest
```

也可以把 `README.md` 和 `启动指南.md` 里的占位地址同步替换。

用户之后只需要：

```bash
git clone https://github.com/${GHCR_OWNER}/Lychee-FD.git
cd Lychee-FD
cp .env.docker.example .env
docker compose pull
docker compose up
```

## 6. Google Drive 备用镜像包

如果用户无法访问 GHCR，可以额外提供离线包：

```bash
docker save ghcr.io/${GHCR_OWNER}/${IMAGE_NAME}:${VERSION} | gzip > lychee-fd-${VERSION}.tar.gz
```

上传 `lychee-fd-${VERSION}.tar.gz` 到 Google Drive。

用户下载后：

```bash
gunzip -c lychee-fd-v0.1.0.tar.gz | docker load
docker compose up
```

Google Drive 包只建议作为备用方式；优先推荐 GHCR，因为用户可以直接 `docker compose pull`。
