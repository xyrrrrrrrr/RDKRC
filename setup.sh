#!/bin/bash
# 仅针对《Deep Learning of Koopman Representation for Control.pdf》的rdkrc工程
# 功能：自动配置工程环境，让Python能识别rdkrc核心包
# 使用方法：1. 将此脚本放在rdkrc工程根目录；2. 在终端执行 source setup_rdkrc_env.sh


# -------------------------- 核心逻辑：自动定位rdkrc工程根目录 --------------------------
# 1. 获取当前脚本（setup_rdkrc_env.sh）的绝对路径（确保在任何目录下source都能定位）
SCRIPT_ABS_PATH=$(realpath "${BASH_SOURCE[0]}")
# 2. 提取脚本所在目录——即rdkrc工程根目录（必须包含rdkrc核心包）
rdkrc_PROJECT_ROOT=$(dirname "$SCRIPT_ABS_PATH")
# 3. 提取脚本所在文件夹的父目录
SCRIPT_DIR=$(dirname "$SCRIPT_ABS_PATH")
SCRIPT_PARENT_DIR=$(dirname "$SCRIPT_DIR")
# echo -e "${SCRIPT_PARENT_DIR}"

# -------------------------- 校验工程结构（确保符合论文rdkrc要求） --------------------------
# 检查工程根目录下是否存在rdkrc核心包（论文算法的核心模块存放目录）
if [ ! -d "$rdkrc_PROJECT_ROOT" ]; then
    echo -e "\033[31m错误：未在当前目录下找到rdkrc核心包！\033[0m"
    echo -e "\033[33m请确认：1. 此脚本已放在rdkrc工程根目录；2. 工程根目录下存在rdkrc文件夹（含models/trainer/utils子目录）\033[0m"
    echo -e "\033[33m正确工程结构（参考论文）：\033[0m"
    echo -e "rdkrc工程根目录/"
    echo -e "├─ rdkrc/          （论文rdkrc算法核心包）"
    echo -e "│  ├─ models/     （论文MLP基函数网络：psi_mlp.py）"
    echo -e "│  ├─ trainer/    （论文损失函数：loss_functions.py）"
    echo -e "│  └─ utils/      （论文矩阵工具：matrix_utils.py）"
    echo -e "└─ setup_rdkrc_env.sh （当前脚本）"
    return 1  # source脚本时用return，避免退出终端
fi


# -------------------------- 配置Python环境变量（让Python识别rdkrc包） --------------------------
# 将rdkrc工程根目录添加到PYTHONPATH（Python模块搜索路径）
# 先检查是否已添加，避免重复配置
if [[ ":$PYTHONPATH:" != *":$SCRIPT_PARENT_DIR:"* ]]; then
    # export PYTHONPATH="$rdkrc_PROJECT_ROOT:$PYTHONPATH"    
    export PYTHONPATH="$SCRIPT_PARENT_DIR:$PYTHONPATH"
    echo -e "\033[32m成功：rdkrc工程根目录已添加到PYTHONPATH\033[0m"
    echo -e "工程根目录：$SCRIPT_PARENT_DIR"
else
    echo -e "\033[32m已配置：rdkrc工程根目录已在PYTHONPATH中\033[0m"
    echo -e "工程根目录：$SCRIPT_PARENT_DIR"
fi


# -------------------------- 验证配置（可选，快速确认是否生效） --------------------------
echo -e "\033[34m正在验证rdkrc包是否可导入...\033[0m"
python3 -c "
try:
    from rdkrc import PsiMLP, compute_total_loss, update_A_B
    print('\033[32m验证通过：rdkrc核心模块（PsiMLP/损失函数等）可正常导入！\033[0m')
except ModuleNotFoundError as e:
    print('\033[31m验证失败：rdkrc模块仍无法导入，错误信息：\033[0m', str(e))
except Exception as e:
    print('\033[31m其他错误：\033[0m', str(e))
"

echo -e "\033[34m使用说明：\033[0m"
echo -e "1. 后续可直接运行论文相关脚本（无需切换目录），例如："
echo -e "   python3 tests/test_psi_mlp.py （测试论文MLP基函数网络）"
echo -e "   python3 examples/run_inverted_pendulum.py （复现论文倒立摆实验）"
echo -e "2. 若打开新终端，需重新执行 source setup.sh 配置环境\033[0m"