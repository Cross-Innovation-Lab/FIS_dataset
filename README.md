# MM-FIS 实验代码

本仓库仅包含可复现实验所需的 **`experiment/`** 与 **`scripts/`**（运行脚本）。

## 说明

- **数据与特征**：需自行准备；路径一般在配置（如 `experiment/configs`）中指定，不在本仓库内。
- **依赖**：按项目环境安装 PyTorch 等依赖后，在 `experiment/` 下执行训练/评估入口（参见各 `run_*.sh`）。

## 推送到 GitHub

1. 在 GitHub 新建空仓库（不要勾选添加 README）。
2. 在本目录执行：

```bash
cd /path/to/MM_FIS
git init
git add .gitignore README.md experiment scripts
git commit -m "Initial commit: experiment code only"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

若已 `git init`，只需 `git add` / `commit` / `remote` / `push`。
