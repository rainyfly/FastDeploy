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

#include "fastdeploy/runtime/runtime.h"

#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/utils/utils.h"

#ifdef ENABLE_ORT_BACKEND
#include "fastdeploy/runtime/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "fastdeploy/runtime/backends/tensorrt/trt_backend.h"
#endif

#ifdef ENABLE_PADDLE_BACKEND
#include "fastdeploy/runtime/backends/paddle/paddle_backend.h"
#endif

#ifdef ENABLE_POROS_BACKEND
#include "fastdeploy/runtime/backends/poros/poros_backend.h"
#endif

#ifdef ENABLE_OPENVINO_BACKEND
#include "fastdeploy/runtime/backends/openvino/ov_backend.h"
#endif

#ifdef ENABLE_LITE_BACKEND
#include "fastdeploy/runtime/backends/lite/lite_backend.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "fastdeploy/runtime/backends/rknpu2/rknpu2_backend.h"
#endif

#ifdef ENABLE_SOPHGO_BACKEND
#include "fastdeploy/runtime/backends/sophgo/sophgo_backend.h"
#endif

namespace fastdeploy {

bool AutoSelectBackend(RuntimeOption& option) {
  auto iter0 = s_default_backends_by_format.find(option.model_format);
  if (iter0 == s_default_backends_by_format.end()) {
    FDERROR << "Cannot found a default backend for model format: "
            << option.model_format
            << ", please define the inference backend in RuntimeOption."
            << std::endl;
    return false;
  }

  auto iter1 = s_default_backends_by_device.find(option.device);
  if (iter1 == s_default_backends_by_device.end()) {
    FDERROR << "Cannot found a default backend for device: " << option.device
            << ", please define the inference backend in RuntimeOption."
            << std::endl;
    return false;
  }

  std::vector<Backend> candidates;
  for (const auto& b0 : iter0->second) {
    for (const auto& b1 : iter1->second) {
      if (b0 == b1) {
        candidates.push_back(b0);
      }
    }
  }

  if (candidates.size() == 0) {
    FDERROR << "Cannot found availabel inference backends by model format: "
            << option.model_format << " with device: " << option.device
            << std::endl;
    return false;
  }

  for (const auto& b : candidates) {
    if (IsBackendAvailable(b)) {
      option.backend = b;
      FDINFO << "FastDeploy will choose " << b << " to inference this model."
             << std::endl;
      return true;
    }
  }
  std::string debug_message = Str(candidates);
  FDERROR << "The candiate backends for " << option.model_format << " & "
          << option.device << " are " << debug_message
          << ", but both of them have not been compiled with current "
             "FastDeploy yet."
          << std::endl;
  return false;
}

bool Runtime::Init(const RuntimeOption& _option) {
  option = _option;
  // decrypt encrypted model
  if ("" != option.encryption_key_) {
#ifdef ENABLE_ENCRYPTION
    if (option.model_from_memory_) {
      option.model_file = Decrypt(option.model_file, option.encryption_key_);
      if (!(option.params_file.empty())) {
        option.params_file =
            Decrypt(option.params_file, option.encryption_key_);
      }
    } else {
      std::string model_buffer = "";
      FDASSERT(ReadBinaryFromFile(option.model_file, &model_buffer),
               "Fail to read binary from model file");
      option.model_file = Decrypt(model_buffer, option.encryption_key_);
      if (!(option.params_file.empty())) {
        std::string params_buffer = "";
        FDASSERT(ReadBinaryFromFile(option.params_file, &params_buffer),
                 "Fail to read binary from parameter file");
        option.params_file = Decrypt(params_buffer, option.encryption_key_);
      }
      option.model_from_memory_ = true;
    }
#else
    FDERROR << "The FastDeploy didn't compile with encryption function."
            << std::endl;
#endif
  }
  // Choose default backend by model format and device if backend is not
  // specified
  if (option.backend == Backend::UNKNOWN) {
    if (!AutoSelectBackend(option)) {
      return false;
    }
  }

  if (option.backend == Backend::ORT) {
    CreateOrtBackend();
  } else if (option.backend == Backend::TRT) {
    CreateTrtBackend();
  } else if (option.backend == Backend::PDINFER) {
    CreatePaddleBackend();
  } else if (option.backend == Backend::OPENVINO) {
    CreateOpenVINOBackend();
  } else if (option.backend == Backend::LITE) {
    CreateLiteBackend();
  } else if (option.backend == Backend::RKNPU2) {
    CreateRKNPU2Backend();
  } else if (option.backend == Backend::SOPHGOTPU) {
    CreateSophgoNPUBackend();
  } else if (option.backend == Backend::POROS) {
    FDASSERT(option.device == Device::CPU || option.device == Device::GPU,
             "Backend::POROS only supports Device::CPU/Device::GPU.");
    FDASSERT(option.model_format == ModelFormat::TORCHSCRIPT,
             "Backend::POROS only supports model format of "
             "ModelFormat::TORCHSCRIPT.");
    FDINFO << "Runtime initialized with Backend::POROS in " << option.device
           << "." << std::endl;
    return true;
  } else {
    FDERROR << "Runtime only support "
               "Backend::ORT/Backend::TRT/Backend::PDINFER/Backend::POROS as "
               "backend now."
            << std::endl;
    return false;
  }
  return true;
}

TensorInfo Runtime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

TensorInfo Runtime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

std::vector<TensorInfo> Runtime::GetInputInfos() {
  return backend_->GetInputInfos();
}

std::vector<TensorInfo> Runtime::GetOutputInfos() {
  return backend_->GetOutputInfos();
}

bool Runtime::Infer(std::vector<FDTensor>& input_tensors,
                    std::vector<FDTensor>* output_tensors) {
  for (auto& tensor : input_tensors) {
    FDASSERT(tensor.device_id < 0 || tensor.device_id == option.device_id,
             "Device id of input tensor(%d) and runtime(%d) are not same.",
             tensor.device_id, option.device_id);
  }
  return backend_->Infer(input_tensors, output_tensors);
}

bool Runtime::Infer() {
  bool result = backend_->Infer(input_tensors_, &output_tensors_, false);
  for (auto& tensor : output_tensors_) {
    tensor.device_id = option.device_id;
  }
  return result;
}

void Runtime::BindInputTensor(const std::string& name, FDTensor& input) {
  bool is_exist = false;
  for (auto& t : input_tensors_) {
    if (t.name == name) {
      is_exist = true;
      t.SetExternalData(input.shape, input.dtype, input.MutableData(),
                        input.device, input.device_id);
      break;
    }
  }
  if (!is_exist) {
    FDTensor new_tensor(name);
    new_tensor.SetExternalData(input.shape, input.dtype, input.MutableData(),
                               input.device, input.device_id);
    input_tensors_.emplace_back(std::move(new_tensor));
  }
}

FDTensor* Runtime::GetOutputTensor(const std::string& name) {
  for (auto& t : output_tensors_) {
    if (t.name == name) {
      return &t;
    }
  }
  FDWARNING << "The output name [" << name << "] don't exist." << std::endl;
  return nullptr;
}

void Runtime::ReleaseModelMemoryBuffer() {
  if (option.model_from_memory_) {
    option.model_file.clear();
    option.model_file.shrink_to_fit();
    option.params_file.clear();
    option.params_file.shrink_to_fit();
  }
}

void Runtime::CreatePaddleBackend() {
  FDASSERT(
      option.device == Device::CPU || option.device == Device::GPU ||
          option.device == Device::IPU,
      "Backend::PDINFER only supports Device::CPU/Device::GPU/Device::IPU.");
  FDASSERT(
      option.model_format == ModelFormat::PADDLE,
      "Backend::PDINFER only supports model format of ModelFormat::PADDLE.");
#ifdef ENABLE_PADDLE_BACKEND
  option.paddle_infer_option.model_file = option.model_file;
  option.paddle_infer_option.params_file = option.params_file;
  option.paddle_infer_option.model_from_memory_ = option.model_from_memory_;
  option.paddle_infer_option.device = option.device;
  option.paddle_infer_option.device_id = option.device_id;
  option.paddle_infer_option.enable_pinned_memory = option.enable_pinned_memory;
  option.paddle_infer_option.external_stream_ = option.external_stream_;
  option.paddle_infer_option.trt_option = option.trt_option;
  option.paddle_infer_option.trt_option.gpu_id = option.device_id;
  backend_ = utils::make_unique<PaddleBackend>();
  auto casted_backend = dynamic_cast<PaddleBackend*>(backend_.get());
  casted_backend->benchmark_option_ = option.benchmark_option;

  if (option.model_from_memory_) {
    FDASSERT(
        casted_backend->InitFromPaddle(option.model_file, option.params_file,
                                       option.paddle_infer_option),
        "Load model from Paddle failed while initliazing PaddleBackend.");
    ReleaseModelMemoryBuffer();
  } else {
    std::string model_buffer = "";
    std::string params_buffer = "";
    FDASSERT(ReadBinaryFromFile(option.model_file, &model_buffer),
             "Fail to read binary from model file");
    FDASSERT(ReadBinaryFromFile(option.params_file, &params_buffer),
             "Fail to read binary from parameter file");
    FDASSERT(casted_backend->InitFromPaddle(model_buffer, params_buffer,
                                            option.paddle_infer_option),
             "Load model from Paddle failed while initliazing PaddleBackend.");
  }
#else
  FDASSERT(false,
           "PaddleBackend is not available, please compiled with "
           "ENABLE_PADDLE_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::PDINFER in " << option.device
         << "." << std::endl;
}

void Runtime::CreateOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend_ = utils::make_unique<OpenVINOBackend>();
  backend_->benchmark_option_ = option.benchmark_option;
  FDASSERT(backend_->Init(option), "Failed to initialize OpenVINOBackend.");
#else
  FDASSERT(false,
           "OpenVINOBackend is not available, please compiled with "
           "ENABLE_OPENVINO_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::OPENVINO in " << option.device
         << "." << std::endl;
}

void Runtime::CreateOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend_ = utils::make_unique<OrtBackend>();
  backend_->benchmark_option_ = option.benchmark_option;

  FDASSERT(backend_->Init(option), "Failed to initialize Backend::ORT.");
#else
  FDASSERT(false,
           "OrtBackend is not available, please compiled with "
           "ENABLE_ORT_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::ORT in " << option.device << "."
         << std::endl;
}

void Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  option.trt_option.model_file = option.model_file;
  option.trt_option.params_file = option.params_file;
  option.trt_option.model_format = option.model_format;
  option.trt_option.gpu_id = option.device_id;
  option.trt_option.enable_pinned_memory = option.enable_pinned_memory;
  option.trt_option.external_stream_ = option.external_stream_;
  backend_ = utils::make_unique<TrtBackend>();
  backend_->benchmark_option_ = option.benchmark_option;
  FDASSERT(backend_->Init(option), "Failed to initialize TensorRT backend.");
#else
  FDASSERT(false,
           "TrtBackend is not available, please compiled with "
           "ENABLE_TRT_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::TRT in " << option.device << "."
         << std::endl;
}

void Runtime::CreateLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  backend_ = utils::make_unique<LiteBackend>();
  backend_->benchmark_option_ = option.benchmark_option;

  FDASSERT(backend_->Init(option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "LiteBackend is not available, please compiled with "
           "ENABLE_LITE_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::PDLITE in " << option.device
         << "." << std::endl;
}

void Runtime::CreateRKNPU2Backend() {
  FDASSERT(option.model_from_memory_ == false,
           "RKNPU2Backend don't support to load model from memory");
  FDASSERT(option.device == Device::RKNPU,
           "Backend::RKNPU2 only supports Device::RKNPU2");
  FDASSERT(option.model_format == ModelFormat::RKNN,
           "RKNPU2Backend only support model format of ModelFormat::RKNN");
#ifdef ENABLE_RKNPU2_BACKEND
  auto rknpu2_option = RKNPU2BackendOption();
  rknpu2_option.cpu_name = option.rknpu2_cpu_name_;
  rknpu2_option.core_mask = option.rknpu2_core_mask_;
  backend_ = utils::make_unique<RKNPU2Backend>();
  auto casted_backend = dynamic_cast<RKNPU2Backend*>(backend_.get());
  FDASSERT(casted_backend->InitFromRKNN(option.model_file, rknpu2_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "RKNPU2Backend is not available, please compiled with "
           "ENABLE_RKNPU2_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::RKNPU2 in " << option.device
         << "." << std::endl;
}

void Runtime::CreateSophgoNPUBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  backend_ = utils::make_unique<SophgoBackend>();
  FDASSERT(backend_->Init(option), "Failed to initialize Sophgo backend.");
#else
  FDASSERT(false,
           "SophgoBackend is not available, please compiled with "
           "ENABLE_SOPHGO_BACKEND=ON.");
#endif
  FDINFO << "Runtime initialized with Backend::SOPHGO in " << option.device
         << "." << std::endl;
}

Runtime* Runtime::Clone(void* stream, int device_id) {
  Runtime* runtime = new Runtime();
  if (option.backend != Backend::OPENVINO &&
      option.backend != Backend::PDINFER) {
    runtime->Init(option);
    FDWARNING << "Only OpenVINO/Paddle Inference support \
                  clone engine to  reduce CPU/GPU memory usage now. For "
              << option.backend
              << ", FastDeploy will create a new engine which \
                  will not share memory  with the current runtime."
              << std::endl;
    return runtime;
  }
  FDINFO << "Runtime Clone with Backend:: " << option.backend << " in "
         << option.device << "." << std::endl;
  runtime->option = option;
  runtime->backend_ = backend_->Clone(option, stream, device_id);
  return runtime;
}

// only for poros backend
bool Runtime::Compile(std::vector<std::vector<FDTensor>>& prewarm_tensors,
                      const RuntimeOption& _option) {
#ifdef ENABLE_POROS_BACKEND
  FDASSERT(
      option.model_format == ModelFormat::TORCHSCRIPT,
      "PorosBackend only support model format of ModelFormat::TORCHSCRIPT.");
  if (option.device != Device::CPU && option.device != Device::GPU) {
    FDERROR << "PorosBackend only supports CPU/GPU, but now its "
            << option.device << "." << std::endl;
    return false;
  }
  option.poros_option.device = option.device;
  option.poros_option.device_id = option.device_id;
  option.poros_option.enable_fp16 = option.trt_option.enable_fp16;
  option.poros_option.max_batch_size = option.trt_option.max_batch_size;
  option.poros_option.max_workspace_size = option.trt_option.max_workspace_size;

  backend_ = utils::make_unique<PorosBackend>();
  auto casted_backend = dynamic_cast<PorosBackend*>(backend_.get());
  FDASSERT(
      casted_backend->Compile(option.model_file, prewarm_tensors,
                              option.poros_option),
      "Load model from Torchscript failed while initliazing PorosBackend.");
#else
  FDASSERT(false,
           "PorosBackend is not available, please compiled with "
           "ENABLE_POROS_BACKEND=ON.");
#endif
  return true;
}

}  // namespace fastdeploy
