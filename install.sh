
#!/bin/bash

reboot_needed=0

# bash fonts colors
red='\e[31m'
yellow='\e[33m'
gray='\e[90m'
green='\e[92m'
blue='\e[94m'
magenta='\e[95m'
cyan='\e[96m'
none='\e[0m'
_red() { echo -e ${red}$@${none}; }
_blue() { echo -e ${blue}$@${none}; }
_cyan() { echo -e ${cyan}$@${none}; }
_green() { echo -e ${green}$@${none}; }
_yellow() { echo -e ${yellow}$@${none}; }
_magenta() { echo -e ${magenta}$@${none}; }
_red_bg() { echo -e "\e[41m$@${none}"; }

is_err=$(_red_bg 错误：)
is_warn=$(_red_bg 警告：)
is_info=$(_red_bg 提示：)

err() {
    echo -e "\n$is_err $@\n" && exit 1
}

warn() {
    echo -e "\n$is_warn $@\n"
}

info() {
    echo -e "\n$is_info $@\n"
}

check_err() {
    if [[ $? != 0 ]]; then echo -e "\n$is_err $@\n" && exit 1; fi
}

if [[ $(lsb_release -rs) != "20.04" || $(lsb_release -is) != "Ubuntu" || $(uname -m) != "x86_64" ]]; then
    err "仅支持 ${yellow}(Ubuntu 20.04 和 x86_64 架构)${none}"
fi

configure_pip_source() {
    mkdir -p ~/.pip

    touch ~/.pip/pip.conf

    cat <<EOL > ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOL

    pip config list
}

install_basic_tools() {
    info "${yellow}开始安装Python3..."

    sleep 3

    # Update package list
    info "${yellow}更新软件包列表..."
    sudo apt update
    check_err "${yellow}更新软件包列表失败"

    # Install Python, pip, wget
    sudo apt install -y python3 python3-pip wget
    check_err "${yellow}安装Python, pip, wget失败"

    # Call the function
    configure_pip_source
    
}

install_nvidia_driver() {
    info "${yellow}开始安装NVIDIA 驱动..."

    sleep 3

    # Check if NVIDIA driver is already installed
    if command -v nvidia-smi >/dev/null 2>&1; then
        info "${yellow}NVIDIA 驱动已经安装"
        return
    fi

    # Add the official NVIDIA driver PPA
    info "${yellow}添加官方 NVIDIA 驱动 PPA..."
    sudo add-apt-repository ppa:graphics-drivers/ppa -y
    check_err "${yellow}添加官方 NVIDIA 驱动 PPA失败"

    sudo apt update
    check_err "${yellow}添加官方 NVIDIA 驱动 PPA失败"

    # List available NVIDIA driver versions
    available_drivers=$(ubuntu-drivers devices | grep -E 'nvidia-driver-[0-9]+')
    if [[ -z "$available_drivers" ]]; then
        err "${yellow}没有找到可用的 NVIDIA 驱动${none}."
        return
    fi

    # Display available drivers with numbers
    info "${yellow}可用的 NVIDIA 驱动版本:"
    PS3="请输入要安装的驱动版本的序号： "
    select driver in $(echo "$available_drivers" | awk '{print $3}'); do
        if [[ -n "$driver" ]]; then
            info "${yellow}您选择了 $driver"
            break
        else
            warn "${yellow}无效的选择，请重新选择"
        fi
    done

    # Install the selected NVIDIA driver version
    info "${yellow}安装 NVIDIA 驱动 $driver..."
    sudo apt install -y "$driver"
    check_err "${yellow}安装 NVIDIA 驱动 $driver失败"

    # Disable Nouveau driver if necessary
    info "${yellow}禁用 Nouveau 驱动..."
    sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nouveau.conf"
    sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nouveau.conf"
    sudo update-initramfs -u

    reboot_needed=1
}

conda_activate_pointfoot_legged_gym() {
    local anaconda_dir="$HOME/anaconda3"

    __conda_setup="$('$anaconda_dir/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        # Use eval to apply Conda setup if successful
        eval "$__conda_setup"
    else
        if [ -f "$anaconda_dir/etc/profile.d/conda.sh" ]; then
            # Source the conda.sh script if it exists
            . "$anaconda_dir/etc/profile.d/conda.sh"
        else
            # Fallback to adding Conda to PATH if other methods fail
            export PATH="$anaconda_dir/bin:$PATH"
        fi
    fi
    unset __conda_setup

    # Activate the newly created environment
    info "${yellow}激活 Conda 环境 pointfoot_legged_gym..."
    conda activate pointfoot_legged_gym
    check_err "${yellow}激活 Conda 环境 pointfoot_legged_gym失败"
}


install_anaconda() {
    info "${yellow}开始安装anaconda..."

    sleep 3

    # Define Anaconda installer URL and filename
    local anaconda_url="https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh"
    local installer_file="$HOME/$(basename $anaconda_url)"
    local anaconda_dir="$HOME/anaconda3"

    rm -rf "$installer_file"

    # Download Anaconda installer
    info "${yellow}下载 Anaconda 安装脚本..."
    cd ~ && wget "$anaconda_url"
    check_err "${yellow}下载 Anaconda 安装脚本失败"

    # Run the installer
    info "${yellow}运行 Anaconda 安装脚本..."
    bash "$installer_file"
    check_err "${yellow}运行 Anaconda 安装脚本"

    # Initialize Conda if not automatically configured
    info "${yellow}初始化 Conda 环境..."
    ~/anaconda3/bin/conda init
    check_err "${yellow}初始化 Conda 环境失败"

    # Create Conda environment for RL training
    info "${yellow}创建 Conda 环境用于 RL 训练..."
    __conda_setup="$('$anaconda_dir/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        # Use eval to apply Conda setup if successful
        eval "$__conda_setup"
    else
        if [ -f "$anaconda_dir/etc/profile.d/conda.sh" ]; then
            # Source the conda.sh script if it exists
            . "$anaconda_dir/etc/profile.d/conda.sh"
        else
            # Fallback to adding Conda to PATH if other methods fail
            export PATH="$anaconda_dir/bin:$PATH"
        fi
    fi
    unset __conda_setup
    conda create --name pointfoot_legged_gym python=3.8 -y
    check_err "${yellow}创建 Conda 环境用于 RL 训练失败"

    conda_activate_pointfoot_legged_gym

    info "${yellow}Anaconda 安装和 Conda 环境配置完成。"
}

install_pytorch() {
    info "${yellow}开始安装Pytorch..."

    sleep 3

    conda_activate_pointfoot_legged_gym

    # Install PyTorch, TorchVision, Torchaudio, and PyTorch CUDA
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    check_err "${yellow}开始安装Pytorch失败"

    info "${yellow}开始安装onnx, tensorboard, setuptools..."
    pip install onnx tensorboard==2.12.0 setuptools==59.5.0
    check_err "${yellow}安装onnx, tensorboard, setuptools失败"
}

install_isaac_gym() {
    info "${yellow}开始安装Isaac Gym..."

    sleep 3

    conda_activate_pointfoot_legged_gym

    local install_dir="$HOME/limx_rl"
    local isaac_gym_url="https://developer.nvidia.com/isaac-gym-preview-4"
    local installer_file="$install_dir/isaac-gym-preview-4.tar.gz"

    # Create the directory for training code
    info "${yellow}创建目录 $install_dir ..."
    mkdir -p "$install_dir"
    check_err "${yellow}创建目录 $install_dir 失败"

    # Download Isaac Gym if it is not already downloaded
    if [ ! -f "$installer_file" ]; then
        info "${yellow}下载 Isaac Gym 安装包..."
        wget "$isaac_gym_url" -O "$installer_file"
        check_err "${yellow}下载 Isaac Gym 安装包失败"
    fi

    # Extract the installation package
    info "${yellow}解压 Isaac Gym 安装包到 $install_dir ..."
    tar -xzvf "$installer_file" -C "$install_dir"
    check_err "${yellow}解压 Isaac Gym 安装包到 $install_dir"

    # Install Isaac Gym
    info "${yellow}安装 Isaac Gym ..."
    cd $install_dir/isaacgym/python
    pip install -e .
    check_err "${yellow}安装 Isaac Gym失败"

    # Fix NumPy compatibility issues
    sed -i 's/np.float/float/' isaacgym/torch_utils.py

    info "${yellow}Isaac Gym 安装和配置完成"
}

install_pointfoot_legged_gym() {
    info "${yellow}开始安装pointfoot-legged-gym..."

    sleep 3

    conda_activate_pointfoot_legged_gym

    local conda_env="pointfoot_legged_gym"
    local rl_dir="$HOME/limx_rl"
    local pointfoot_repo="https://github.com/limxdynamics/pointfoot-legged-gym.git"

    mkdir -p "$rl_dir"

    # Install pointfoot-legged-gym
    info "${yellow}安装 pointfoot-legged-gym 库 ..."
    cd "$rl_dir"
    if [ ! -d "$rl_dir/pointfoot-legged-gym" ]; then
        git clone "$pointfoot_repo"
        check_err "${yellow}pointfoot-legged-gym 库失败"
    fi
    cd "$rl_dir/pointfoot-legged-gym"
    pip install -e .
    check_err "${yellow}安装 pointfoot-legged-gym 库失败"
    info "${yellow}安装 pointfoot-legged-gym 库成功"
}

install_basic_tools

install_nvidia_driver

install_anaconda

install_pytorch

install_isaac_gym

install_pointfoot_legged_gym

if [ $reboot_needed -eq 1 ]; then
    read -p "安装完成需要重启才能生效，是否现在重启系统？(y/n): " answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        echo "请稍后手动重启系统。"
    fi
fi
