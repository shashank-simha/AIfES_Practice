#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "AIfESModel.h"
#include "../../utils/logger.h"

AIfESModel::AIfESModel(const ModelConfig& cfg,
                       const char* param_path,
                       FileAdapter* adapter_ptr)
    : ModelBase(cfg),
      parameter_memory(nullptr),
      inference_memory(nullptr),
      params_file_path(param_path),
      adapter(adapter_ptr)
{
}

AIfESModel::~AIfESModel() {
    free_parameter_memory();
    free_inference_memory();
    free_training_memory();
}

void AIfESModel::set_param_path(const char* path) {
    params_file_path = path;
}

bool AIfESModel::allocate_parameter_memory() {
    size_t size = aialgo_sizeof_parameter_memory(&model);
    parameter_memory = config.allocator_fn(size);
    if (!parameter_memory) {
        LOG_ERROR("AIfESModel: Failed to allocate parameter memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_distribute_parameter_memory(&model, parameter_memory, static_cast<uint32_t>(size));
    LOG_INFO("AIfESModel: Parameter memory allocated (%u bytes)", (unsigned)size);
    return true;
}

void AIfESModel::free_parameter_memory() {
    if (parameter_memory) {
        config.free_fn(parameter_memory);
        parameter_memory = nullptr;
        LOG_INFO("AIfESModel: Parameter memory freed");
    }
}

bool AIfESModel::allocate_inference_memory() {
    size_t size = aialgo_sizeof_inference_memory(&model);
    inference_memory = config.allocator_fn(size);
    if (!inference_memory) {
        LOG_ERROR("AIfESModel: Failed to allocate inference memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_schedule_inference_memory(&model, inference_memory, static_cast<uint32_t>(size));
    LOG_INFO("AIfESModel: Inference memory allocated (%u bytes)", (unsigned)size);
    return true;
}

void AIfESModel::free_inference_memory() {
    if (inference_memory) {
        config.free_fn(inference_memory);
        inference_memory = nullptr;
        LOG_INFO("AIfESModel: Inference memory freed");
    }
}

bool AIfESModel::allocate_training_memory(aiopti_t* optimizer) {
    size_t size = aialgo_sizeof_training_memory(&model, optimizer);
    training_memory = config.allocator_fn(size);
    if (!training_memory) {
        LOG_ERROR("AIfESModel: Failed to allocate training memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_schedule_training_memory(&model, optimizer, training_memory, static_cast<uint32_t>(size));
    aialgo_init_model_for_training(&model, optimizer);
    LOG_INFO("AIfESModel: Training memory allocated (%u bytes)", (unsigned)size);
    return true;
}

void AIfESModel::free_training_memory() {
    if (training_memory) {
        config.free_fn(training_memory);
        training_memory = nullptr;
        LOG_INFO("AIfESModel: Training memory freed");
    }
}

bool AIfESModel::load_model_parameters() {
    if (!adapter || !params_file_path) {
        LOG_WARN("AIfESModel: No adapter or parameter file path, skipping load");
        return true;
    }

    LOG_INFO("AIfESModel: Loading model parameters from %s", params_file_path);
    if (!adapter->open(params_file_path, FileAdapter::OpenMode::READ)) {
        LOG_INFO("AIfESModel: No saved parameters found, using defaults");
        return true;
    }

    size_t expected_bytes = aialgo_sizeof_parameter_memory(&model);
    size_t actual_bytes = adapter->size();
    if (actual_bytes != expected_bytes) {
        LOG_WARN("AIfESModel: Parameter file size mismatch (expected=%u, actual=%u)",
                 (unsigned)expected_bytes, (unsigned)actual_bytes);
        adapter->close();
        return true;
    }

    uint8_t* dst = reinterpret_cast<uint8_t*>(parameter_memory);
    const size_t CHUNK_SIZE = 1024;
    size_t total_read = 0;

    while (total_read < expected_bytes) {
        size_t to_read = std::min(CHUNK_SIZE, expected_bytes - total_read);
        size_t n = adapter->read(dst + total_read, to_read);
        if (n != to_read) {
            LOG_ERROR("AIfESModel: Read error (got %u, expected %u)", (unsigned)n, (unsigned)to_read);
            adapter->close();
            return false;
        }
        total_read += n;
    }

    adapter->close();
    LOG_INFO("AIfESModel: Parameters loaded successfully (%u bytes)", (unsigned)total_read);
    return true;
}

bool AIfESModel::store_model_parameters() {
    if (!adapter || !params_file_path) {
        LOG_WARN("AIfESModel: No adapter or parameter file path, skipping store");
        return false;
    }

    LOG_INFO("AIfESModel: Storing model parameters");
    char tmp_path[256];
    std::snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", params_file_path);

    if (!adapter->open(tmp_path, FileAdapter::OpenMode::WRITE)) {
        LOG_ERROR("AIfESModel: Failed to open temp file for writing");
        return false;
    }

    ailayer_t* current = model.input_layer;
    size_t total_bytes_written = 0;

    while (current) {
        for (int p = 0; p < current->trainable_params_count; ++p) {
            aitensor_t* tensor = current->trainable_params[p];
            if (!tensor || !tensor->data) continue;

            uint32_t total_elements = 1;
            for (uint8_t d = 0; d < tensor->dim; ++d) {
                total_elements *= tensor->shape[d];
            }

            size_t bytes = static_cast<size_t>(total_elements) * sizeof(float);
            size_t written = 0;
            const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(tensor->data);
            const size_t CHUNK_SIZE = 1024;

            while (written < bytes) {
                size_t to_write = std::min(CHUNK_SIZE, bytes - written);
                size_t ret = adapter->write(data_ptr + written, to_write);
                if (ret != to_write) {
                    LOG_ERROR("AIfESModel: Write error (got %u, expected %u)",
                              (unsigned)ret, (unsigned)to_write);
                    adapter->close();
                    adapter->remove(tmp_path);
                    return false;
                }
                written += ret;
                total_bytes_written += ret;
            }
        }

        if (current == model.output_layer) break;
        current = current->output_layer;
    }

    adapter->close();
    adapter->remove(params_file_path);
    if (!adapter->rename(tmp_path, params_file_path)) {
        LOG_ERROR("AIfESModel: Failed to rename temp file");
        adapter->remove(tmp_path);
        return false;
    }

    LOG_INFO("AIfESModel: Parameters stored (%u bytes)", (unsigned)total_bytes_written);
    return true;
}

bool AIfESModel::init() {
    LOG_INFO("AIfESModel: Initializing...");
    if (!build_model()) {
        LOG_ERROR("AIfESModel: build_model() failed");
        return false;
    }

    aialgo_compile_model(&model);

    if (!allocate_parameter_memory()) return false;

    // initialize parameters with default values
    aialgo_initialize_parameters_model(&model);

    // try loading if pre-trained params are available
    if (params_file_path && adapter) {
        if (!load_model_parameters()) return false;
    }

    if (!allocate_inference_memory()) return false;

    return true;
}
