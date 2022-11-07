import cv2
import os
import tempfile

import fastdeploy as fd
from fastdeploy.utils.example_resouce import get_detection_test_image


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path of PaddleDetection model directory")
    parser.add_argument(
        "--model_hub",
        default=None,
        help="Model name in model hub, the model will be downloaded automatically."
    )
    parser.add_argument(
        "--image", default=None, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 640, 640])
        option.set_trt_input_shape("scale_factor", [1, 2])
    return option


args = parse_arguments()

if args.model_dir is None and args.model_hub is None:
    model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')
elif args.model_dir is not None:
    model_dir = args.model_dir
else:
    model_dir = fd.download_model(name=args.model_hub)

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.detection.PPYOLOE(
    model_file, params_file, config_file, runtime_option=runtime_option)

# 预测图片检测结果
if args.image is None:
    image = get_detection_test_image()
else:
    image = args.image
im = cv2.imread(image)
result = model.predict(im.copy())
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
