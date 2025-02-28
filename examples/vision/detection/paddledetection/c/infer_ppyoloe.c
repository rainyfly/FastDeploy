// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>

#include "fastdeploy_capi/vision.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const char* model_dir, const char* image_file) {
  char model_file[100];
  char params_file[100];
  char config_file[100];
  int max_size = 99;
  snprintf(model_file, max_size, "%s%c%s", model_dir, sep, "model.pdmodel");
  snprintf(params_file, max_size, "%s%c%s", model_dir, sep, "model.pdiparams");
  snprintf(config_file, max_size, "%s%c%s", model_dir, sep, "infer_cfg.yml");

  FD_C_RuntimeOptionWrapper* option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapperUseCpu(option);

  FD_C_PPYOLOEWrapper* model = FD_C_CreatesPPYOLOEWrapper(
      model_file, params_file, config_file, option, PADDLE);

  FD_C_Mat im = FD_C_Imread(image_file);

  FD_C_DetectionResultWrapper* result_wrapper =
      FD_C_CreateDetectionResultWrapper();

  if (!FD_C_PPYOLOEWrapperPredict(model, im, result_wrapper)) {
    printf("Failed to predict.\n");
    return;
  }

  FD_C_DetectionResult* result =
      FD_C_DetectionResultWrapperGetData(result_wrapper);
  FD_C_Mat vis_im = FD_C_VisDetection(im, result, 0.5, 1, 0.5);

  FD_C_Imwrite("vis_result.jpg", vis_im);
  printf("Visualized result saved in ./vis_result.jpg\n");

  FD_C_DestroyRuntimeOptionWrapper(option);
  FD_C_DestroyPPYOLOEWrapper(model);
  FD_C_DestroyDetectionResultWrapper(result_wrapper);
  FD_C_DestroyDetectionResult(result);
  FD_C_DestroyMat(im);
  FD_C_DestroyMat(vis_im);
}

void GpuInfer(const char* model_dir, const char* image_file) {
  char model_file[100];
  char params_file[100];
  char config_file[100];
  int max_size = 99;
  snprintf(model_file, max_size, "%s%c%s", model_dir, sep, "model.pdmodel");
  snprintf(params_file, max_size, "%s%c%s", model_dir, sep, "model.pdiparams");
  snprintf(config_file, max_size, "%s%c%s", model_dir, sep, "infer_cfg.yml");

  FD_C_RuntimeOptionWrapper* option = FD_C_CreateRuntimeOptionWrapper();
  FD_C_RuntimeOptionWrapperUseGpu(option, 0);

  FD_C_PPYOLOEWrapper* model = FD_C_CreatesPPYOLOEWrapper(
      model_file, params_file, config_file, option, PADDLE);

  FD_C_Mat im = FD_C_Imread(image_file);

  FD_C_DetectionResultWrapper* result_wrapper =
      FD_C_CreateDetectionResultWrapper();

  if (!FD_C_PPYOLOEWrapperPredict(model, im, result_wrapper)) {
    printf("Failed to predict.\n");
    return;
  }

  FD_C_DetectionResult* result =
      FD_C_DetectionResultWrapperGetData(result_wrapper);
  FD_C_Mat vis_im = FD_C_VisDetection(im, result, 0.5, 1, 0.5);

  FD_C_Imwrite("vis_result.jpg", vis_im);
  printf("Visualized result saved in ./vis_result.jpg\n");

  FD_C_DestroyRuntimeOptionWrapper(option);
  FD_C_DestroyPPYOLOEWrapper(model);
  FD_C_DestroyDetectionResultWrapper(result_wrapper);
  FD_C_DestroyDetectionResult(result);
  FD_C_DestroyMat(im);
  FD_C_DestroyMat(vis_im);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf(
        "Usage: infer_demo path/to/model_dir path/to/image run_option, "
        "e.g ./infer_model ./ppyoloe_model_dir ./test.jpeg 0"
        "\n");
    printf(
        "The data type of run_option is int, 0: run with cpu; 1: run with gpu"
        "\n");
    return -1;
  }

  if (atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  }
  return 0;
}
