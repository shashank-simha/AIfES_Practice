#pragma once
#include "../../../core/DatasetBase.h"
#include "../../../core/FileAdapter.h"
#include "../../../models/aifes/ClassificationModel.h"

// ============================
// MNIST Model Configuration
// ============================

#define INPUT_CHANNELS      1       /**< Number of input channels (grayscale) */
#define INPUT_HEIGHT        28      /**< Input image height */
#define INPUT_WIDTH         28      /**< Input image width */

#define CONV1_FILTERS       4       /**< Number of filters in first conv layer */
#define CONV2_FILTERS       8       /**< Number of filters in second conv layer */
#define KERNEL_SIZE         {3, 3}  /**< Convolution kernel size */
#define STRIDE              {1, 1}  /**< Convolution stride */
#define PADDING             {1, 1}  /**< Convolution padding */
#define DILATION            {1, 1}  /**< Convolution dilation */
#define POOL_SIZE           {2, 2}  /**< Pooling window size */
#define POOL_STRIDE         {2, 2}  /**< Pooling stride */
#define POOL_PADDING        {0, 0}  /**< Pooling padding */
#define DENSE1_SIZE         32      /**< Units in first dense layer */

#define OUTPUT_SIZE         10      /**< Number of output classes */

/**
 * @brief MNIST model wrapper class.
 *
 * Defines and manages a small CNN model for MNIST digit classification.
 */
class MNISTModel : public ClassificationModel {
public:
    /**
     * @brief Construct MNISTModel (does not allocate memory).
     * @param cfg Model configuration (shapes, allocators, etc.).
     * @param param_path Optional path to parameter file. If nullptr, no automatic persistence is performed.
     * @param adapter File adapter implementation (abstracted for portability)
     */
    explicit MNISTModel(const ModelConfig& cfg,
                        const char* param_path = nullptr,
                        FileAdapter* adapter = nullptr);

    /**
     * @brief Destroy MNISTModel.
     */
    ~MNISTModel() = default;

private:
    /**
     * @brief Build the MNIST-specific model architecture.
     *
     * Overrides the base method to construct the layer sequence:
     * - Input layer: (1, 28, 28)
     * - Conv/Pool layers
     * - Fully-connected layers
     * - Softmax output layer with 10 classes
     *
     * @return true if the model was built successfully, false otherwise.
     */
    bool build_model() override;
};
