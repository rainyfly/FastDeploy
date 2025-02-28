[English](README.md) | 简体中文
# PaddleSegmentation 服务化部署示例

在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](../../../../../serving/README_CN.md)


## 启动服务

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/serving

#下载yolov5模型文件
wget  https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# 将模型文件放入 models/runtime/1目录下
mv PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer/model.pdmodel models/runtime/1/
mv PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer/model.pdiparams models/runtime/1/

# 拉取fastdeploy镜像(x.y.z为镜像版本号，需参照serving文档替换为数字)
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /serving 目录
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# 启动服务(不设置CUDA_VISIBLE_DEVICES环境变量，会拥有所有GPU卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```
>> **注意**: 当出现"Address already in use", 请使用`--grpc-port`指定端口号来启动服务，同时更改paddleseg_grpc_client.py中的请求端口号

服务启动成功后， 会有以下输出:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## 客户端请求

在物理机器中执行以下命令，发送grpc请求并输出结果
```
#下载测试图片
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

#安装客户端依赖
python3 -m pip install tritonclient[all]

# 发送请求
python3 paddleseg_grpc_client.py
```

发送请求成功后，会返回json格式的检测结果并打印输出:
```

```

## 配置修改

当前默认配置在CPU上运行ONNXRuntime引擎， 如果要在GPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](../../../../../serving/docs/zh_CN/model_configuration.md)
