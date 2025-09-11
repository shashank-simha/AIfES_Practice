#include "ClassificationModel.h"
#include "logger.h"
#include <algorithm>
#include <cstring>

ClassificationModel::ClassificationModel(const ModelConfig& cfg,
                                         const char* param_path,
                                         FileAdapter* adapter)
    : AIfESModel(cfg, param_path, adapter),
      loss(nullptr),
      optimizer(nullptr)
{
}

uint32_t ClassificationModel::infer(float* input_buffer) {
    if (!input_buffer) {
        LOG_ERROR("ClassificationModel: infer called with null input_buffer");
        return INVALID_CLASS;
    }

    if (config.input_shape.empty() || config.output_shape.empty()) {
        LOG_ERROR("ClassificationModel: input/output shape not set");
        return INVALID_CLASS;
    }

    // Forward pass using AIfES model
    aitensor_t input_tensor = create_tensor(1, config.input_shape, input_buffer);
    aitensor_t* output_tensor = aialgo_forward_model(&model, &input_tensor);
    if (!output_tensor || !output_tensor->data) {
        LOG_ERROR("ClassificationModel: forward pass failed");
        return INVALID_CLASS;
    }

    // Handle batch_size=1: output may be 2D [1, num_classes]
    float* out = reinterpret_cast<float*>(output_tensor->data);
    size_t out_len = 0;

    if (output_tensor->dim == 1) {
        out_len = output_tensor->shape[0];
    } else if (output_tensor->dim == 2 && output_tensor->shape[0] == 1) {
        out_len = output_tensor->shape[1];
    } else {
        LOG_ERROR("ClassificationModel: Expected 1D output or batch=1 2D output, got %uD", output_tensor->dim);
        return INVALID_CLASS;
    }

    // Argmax
    uint32_t pred_class = 0;
    float max_val = out[0];
    for (uint32_t i = 1; i < out_len; ++i) {
        if (out[i] > max_val) {
            max_val = out[i];
            pred_class = i;
        }
    }

    LOG_DEBUG("ClassificationModel: pred=%u, confidence=%.6f", pred_class, max_val);
    return pred_class;
}

void ClassificationModel::test(DatasetBase& ds, uint32_t num_samples) {
    LOG_INFO("Testing %u samples...", num_samples);

    // Buffers allocated once for input/target
    uint32_t total_elements = 1;
    for (auto d : config.input_shape) total_elements *= d;

    float* input_buffer = (float*) ps_malloc(total_elements * sizeof(float));
    uint32_t* target_buffer = (uint32_t*) ps_malloc(sizeof(uint32_t));

    if (!input_buffer || !target_buffer) {
        LOG_ERROR("ClassificationModel: buffer allocation failed");
        if (input_buffer) free(input_buffer);
        if (target_buffer) free(target_buffer);
        return;
    }

    uint32_t correct = 0;
    for (uint32_t i = 0; i < num_samples; i++) {
        ds.fetch_batch(1, input_buffer, target_buffer);

        uint32_t pred   = infer(input_buffer);
        uint32_t actual = target_buffer[0];
        if (pred == actual) correct++;

        LOG_DEBUG("ClassificationModel: pred: %d, actual: %d, correct: %s",
            pred, actual, (pred == actual) ? "YES" : "NO");

        if (((i + 1) * 100) / num_samples > (i * 100) / num_samples) {
            LOG_PROGRESS(i + 1, num_samples, "Samples");
        }
    }

    float acc = 100.0f * correct / num_samples;
    LOG_INFO("Accuracy: %u/%u (%.2f%%)", correct, num_samples, acc);

    free(input_buffer);
    free(target_buffer);
}

void ClassificationModel::train(DatasetBase& ds,
                                uint32_t num_samples,
                                uint32_t batch_size,
                                uint32_t num_epoch,
                                bool retrain,
                                bool early_stopping,
                                float early_stopping_target_loss) {
    LOG_INFO("Training on %u samples...", num_samples);

    // ---- Loss and Optimizer setup ----
    ailoss_crossentropy_f32_t crossentropy_loss;
    aiopti_sgd_f32_t sgd_opti = { .learning_rate = 0.001f };

    // ---- Loss setup ----
    if (!this->loss) {
        model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
        if (!model.loss) {
            LOG_ERROR("ClassificationModel: loss initialization failed");
            return;
        }
        LOG_INFO("ClassificationModel: default loss function (crossentropy) initialized");
    } else {
        model.loss = this->loss;
    }

    // ---- Optimizer setup ----
    aiopti_t* opti = nullptr;
    if (!this->optimizer) {
        opti = aiopti_sgd_f32_default(&sgd_opti);
        if (!opti) {
            LOG_ERROR("ClassificationModel: optimizer initialization failed");
            return;
        }
        LOG_INFO("ClassificationModel: default optimizer (sgd) initialized");
    } else {
        opti = this->optimizer;
    }


    // ---- Retrain option ----
    if (retrain) {
        LOG_INFO("Retraining: reinitializing model parameters");
        aialgo_initialize_parameters_model(&model);
    }

    // ---- Allocate training memory ----
    if (!allocate_training_memory(opti))
        return;

    // ---- Allocate batch buffers ----
    uint32_t input_elems = batch_size;
    for (auto d : config.input_shape) input_elems *= d;

    float* input_buffer  = (float*) ps_malloc(input_elems * sizeof(float));
    uint32_t* target_idx = (uint32_t*) ps_malloc(batch_size * sizeof(uint32_t));
    float* target_onehot = (float*) ps_malloc(batch_size * config.output_shape[0] * sizeof(float));

    if (!input_buffer || !target_idx || !target_onehot) {
        LOG_ERROR("ClassificationModel: buffer allocation failed");
        if (input_buffer) free(input_buffer);
        if (target_idx) free(target_idx);
        if (target_onehot) free(target_onehot);
        free_training_memory();
        return;
    }

    LOG_DEBUG("ClassificationModel: allocated input_buffer: %u bytes, target_idx: %u bytes, target_onehot: %u bytes",
              input_elems * sizeof(float),
              batch_size * sizeof(uint32_t),
              batch_size * config.output_shape[0] * sizeof(float));

    // ---- Training loop ----
    for (uint32_t epoch = 0; epoch < num_epoch; epoch++) {
        uint32_t steps = num_samples / batch_size;
        float epoch_loss = 0.0f;

        for (uint32_t step = 0; step < steps; step++) {
            // Fetch next batch
            ds.fetch_batch(batch_size, input_buffer, target_idx);

            // Convert labels → one-hot
            memset(target_onehot, 0, batch_size * config.output_shape[0] * sizeof(float));
            for (uint32_t i = 0; i < batch_size; i++) {
                uint32_t cls = target_idx[i];
                if (cls < config.output_shape[0]) {
                    target_onehot[i * config.output_shape[0] + cls] = 1.0f;
                }
            }

            // Create tensors
            aitensor_t input_tensor  = create_tensor(batch_size, config.input_shape, input_buffer);
            std::vector<uint32_t> target_shape = { batch_size, static_cast<uint32_t>(config.output_shape[0]) };
            aitensor_t target_tensor = create_tensor(1, target_shape, target_onehot);

            // Train step
            aialgo_train_model(&model, &input_tensor, &target_tensor, opti, batch_size);

            // Track loss
            float batch_loss;
            aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &batch_loss);
            epoch_loss += batch_loss;

            // Progress bar
            if (((step + 1) * 100) / steps > (step * 100) / steps) {
                char metric[32];
                snprintf(metric, sizeof(metric), "Steps (epoch %u/%u)", epoch + 1, num_epoch);
                LOG_PROGRESS(step + 1, steps, metric);
            }
        }

        // Print per-epoch loss
        epoch_loss /= steps;
        LOG_INFO("Epoch %u/%u - Loss: %.4f", epoch + 1, num_epoch, epoch_loss);
        store_model_parameters();

        if (early_stopping && (epoch_loss < early_stopping_target_loss)) {
            LOG_INFO("Early stopping: epoch loss %.4f < target %.4f",
                     epoch_loss, early_stopping_target_loss);
            break;
        }
    }

    LOG_INFO("Finished training");

    // ---- Cleanup ----
    free(input_buffer);
    free(target_idx);
    free(target_onehot);
    free_training_memory();
}
