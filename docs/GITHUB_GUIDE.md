# 上传 GitHub 与面试展示

## 上传哪些内容

提交项目根目录的 README、`src`、`configs`、`scripts`、`web`、`examples`、`tests`、`docs`、`results`、依赖文件、`pyproject.toml`、`.gitignore` 与 `.github`。

`.gitignore` 排除 `.idea`、Python 缓存、虚拟环境、凭据、数据和权重。不要把整个 Documents 或上级 `w` 文件夹上传。确认仓库根目录直接能看到 README 和 src。

本地的 `docs/RESUME_AND_REVIEW.md` 是给项目作者的审查与简历建议，由 `.gitignore` 排除，发布 ZIP 不包含该文件。

## 推荐方法：Git 命令上传

1. 登录 GitHub，在右上角 `+` → `New repository` 创建名为 `text2sql-spider-qlora` 的仓库。选择 Public；本地已有 README，新建时不勾选初始化 README、gitignore 或 license。
2. 复制仓库 HTTPS 地址，例如 `https://github.com/你的用户名/text2sql-spider-qlora.git`。
3. 在项目文件夹打开 PowerShell，运行：

```powershell
git init
git branch -M main
git status --short
git add .
git diff --cached --stat
```

检查待提交清单不含数据、权重、密钥或 IDE 配置，再执行：

```powershell
git commit -m "Add Text-to-SQL training and execution feedback repair system"
git remote add origin https://github.com/你的用户名/text2sql-spider-qlora.git
git push -u origin main
```

第一次推送按浏览器提示登录 GitHub。账户密码不能直接作为 HTTPS Git 密码；优先使用 Git 自带的浏览器登录流程。若 origin 已存在，先 `git remote -v` 确认，避免改错目标。

如果 Git 提示缺少作者身份，用你自己的信息在当前仓库设置：

```powershell
git config user.name "你的姓名或 GitHub 用户名"
git config user.email "你的 GitHub 邮箱或 GitHub 提供的 noreply 邮箱"
```

重新提交即可。

## 也可以使用网页上传

解压发布 ZIP，进入项目根目录。在空仓库中选择 `uploading an existing file`，已有仓库选择 `Add file` → `Upload files`，上传发布版文件和目录。网页方法容易漏掉 `.github`、`.gitignore`，Git 命令更适合后续维护。

GitHub 展示源代码与 README；README 里的 localhost 链接不会在 GitHub 上启动服务。演示时需要本地运行，或另行部署应用。

## 给面试官看的顺序

1. 打开 README，用架构图介绍问题输入、schema、SQL 生成、执行与错误反馈。
2. 打开 `prepare_data.py`，说明为什么按数据库拆分，展示处理后的 meta 文件。
3. 打开 `sft_encoding.py` 与 `train_qlora.py`，说明 prompt 的 `-100` label、完整 completion 和 LoRA 参数。
4. 打开 `evaluate.py`，说明字符串匹配、执行匹配与可执行率的区别。
5. 本地启动 Demo，演示初始字段错误、SQLite 错误、修复 SQL 与结果。准确说明当前为固定脚本演示，真实模型使用另一条启动命令。

在仓库主页设置 Description：`Qwen2.5-Coder + QLoRA Text-to-SQL training, SQLite execution feedback repair, and FastAPI demo`。

Topics 可填：`text-to-sql`、`qlora`、`qwen`、`spider`、`peft`、`fastapi`、`sqlite`。在个人主页 `Customize your pins` 中固定仓库，把仓库链接加到简历项目标题旁。

可录制约一分钟的本地操作视频：启动、选择示例库、执行问题、查看错误与修复。视频标题写“查询与修复流程演示”；只有真实模型运行的视频才写“Qwen 推理演示”。
