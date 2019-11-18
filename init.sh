

echo "############ 安装基础依赖 ############"

apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        git \
        wget \
        curl \
        # python-dev \
        # python-pip \
        vim


echo "############ 通过pip安装python模块 ############"

# pip freeze > requirements.txt
pip install -r requirements.txt

echo "############ 安装anaconda ############"

curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& chmod +x /miniconda.sh \
&& /miniconda.sh -b -p /miniconda \
&& rm /miniconda.sh


# echo "PATH=/miniconda/bin:$PATH" >> /etc/profile

# . /etc/profile

export PATH=/miniconda/bin:$PATH



echo "############ 设置时区 ############"

ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime


echo "############ 设置编码 ############"

LANG=C.UTF-8 LC_ALL=C.UTF-8



# conda create -n ai python=3.6

echo "############ 通过conda安装python模块 ############"


# conda env export > environment.yaml
# conda env create -f environment.yaml




# 启动docker入口文件docker-compose -f docker-compose.yml up
# 停止容器docker-compose -f docker-compo/	se.yml stop
# 进入容器docker exec -it faceLogin /bin/bash
# 或者docker exec -it faceLogin /bin/sh
# 保存容器里的改动: docker commit c9de3957530c aisrc:v1
# 进入刚才保存的镜像docker run -it aisrc:v1


