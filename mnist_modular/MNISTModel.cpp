#include <cstdint>
#include "MNISTModel.h"
#include "logger.h"

MNISTModel::MNISTModel(const ModelConfig& cfg,
                       const char* param_file_path,
                       FileAdapter* adapter)
    : ClassificationModel(cfg, param_file_path, adapter)
{
}

bool MNISTModel::build_model() {
    if (config.input_shape.size() != 3 ||
        config.input_shape[0] != INPUT_CHANNELS ||
        config.input_shape[1] != INPUT_HEIGHT ||
        config.input_shape[2] != INPUT_WIDTH) {
        LOG_ERROR("MNISTModel: invalid input shape. Expected [%u,%u,%u], got [%zu,%zu,%zu]",
                INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                config.input_shape.size() > 0 ? config.input_shape[0] : 0,
                config.input_shape.size() > 1 ? config.input_shape[1] : 0,
                config.input_shape.size() > 2 ? config.input_shape[2] : 0);
        return false;
    }

    // ===== Layers =====
    static uint16_t input_shape[] = {1, config.input_shape[0], config.input_shape[1], config.input_shape[2]};
    static ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);
    static ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_A(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);
    static ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_A();
    static ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);
    static ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_A(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);
    static ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_A();
    static ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);
    static ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_A();
    static ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_A(DENSE1_SIZE);
    static ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_A();
    static ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_A(OUTPUT_SIZE);
    static ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_A();

    // Layer pointer to perform the connection
    ailayer_t *x = nullptr;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    if (!model.input_layer) {
        LOG_ERROR("MNISTModel: failed to create input layer");
        return false;
    }

    x = ailayer_conv2d_chw_f32_default(&conv1_layer, model.input_layer);
    if (!x) { LOG_ERROR("MNISTModel: conv1 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu1 init failed"); return false; }
    x = ailayer_maxpool2d_chw_f32_default(&pool1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: pool1 init failed"); return false; }

    x = ailayer_conv2d_chw_f32_default(&conv2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: conv2 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu2 init failed"); return false; }
    x = ailayer_maxpool2d_chw_f32_default(&pool2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: pool2 init failed"); return false; }

    x = ailayer_flatten_f32_default(&flatten_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: flatten init failed"); return false; }
    x = ailayer_dense_f32_default(&dense1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: dense1 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu3_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu3 init failed"); return false; }
    x = ailayer_dense_f32_default(&dense2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: dense2 init failed"); return false; }

    model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);
    if (!model.output_layer) {
        LOG_ERROR("MNISTModel: softmax init failed");
        return false;
    }

    return true;
}
