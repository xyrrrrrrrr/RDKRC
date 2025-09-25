#!/bin/bash

# Git自动上传脚本
# 使用方法：chmod +x git-auto-upload.sh 然后 ./git-auto-upload.sh

# 配置部分 - 根据需要修改
REMOTE_NAME="origin"       # 远程仓库名称
BRANCH_NAME="main"         # 分支名称
DEFAULT_COMMIT_MSG="自动提交: $(date +'%Y-%m-%d %H:%M:%S')"  # 默认提交信息

# 检查是否在Git仓库中
if [ ! -d .git ]; then
    echo "错误: 当前目录不是Git仓库"
    exit 1
fi

echo "===== 开始Git自动上传流程 ====="

# 检查是否有需要提交的更改
echo "检查更改..."
CHANGES=$(git status --porcelain)
if [ -z "$CHANGES" ]; then
    echo "没有需要提交的更改，退出脚本"
    exit 0
fi

# 提示用户输入提交信息，默认使用时间戳
read -p "请输入提交信息(默认: $DEFAULT_COMMIT_MSG): " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-$DEFAULT_COMMIT_MSG}

# 添加所有更改
echo "添加所有更改..."
git add .
if [ $? -ne 0 ]; then
    echo "错误: 添加文件失败"
    exit 1
fi

# 提交更改
echo "提交更改..."
git commit -m "$COMMIT_MSG"
if [ $? -ne 0 ]; then
    echo "错误: 提交失败"
    exit 1
fi

# 拉取远程最新代码，避免冲突
echo "拉取远程最新代码..."
git pull $REMOTE_NAME $BRANCH_NAME
if [ $? -ne 0 ]; then
    echo "警告: 拉取操作可能存在冲突，请手动处理后再推送"
    read -p "是否继续推送? (y/n) " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
fi

# 推送代码到远程仓库
echo "推送代码到远程仓库..."
git push $REMOTE_NAME $BRANCH_NAME
if [ $? -ne 0 ]; then
    echo "错误: 推送失败"
    exit 1
fi

echo "===== 所有操作完成 ====="
exit 0