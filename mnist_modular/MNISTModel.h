#pragma once
#include <cstdint>
#include <cstddef>
#include "DatasetBase.h"
#include "ModelBase.h"
#include "FileAdapter.h"
#include <aifes.h>

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

static constexpr uint32_t MNIST_INVALID_CLASS = UINT32_MAX;

// ============================
// MNIST Model Class
// ============================

/**
 * @brief MNIST model wrapper class.
 *
 * Defines and manages a small CNN model for MNIST digit classification.
 * Inherits from ModelBase to follow the same design as datasets.
 */
class MNISTModel : public ModelBase {
public:
    /**
     * @brief Construct MNISTModel (does not allocate memory).
     * @param cfg Model configuration (shapes, allocators, etc.).
     * @param param_path Optional path to parameter file (nullptr = no persistence).
     * @param adapter File adapter implementation (abstracted for portability)
     */
    explicit MNISTModel(const ModelConfig& cfg,
                        const char* param_path = nullptr,
                        FileAdapter* adapter = nullptr);

    /**
     * @brief Destroy MNISTModel (frees allocated memory).
     */
    ~MNISTModel() override;

    /**
     * @brief Set parameter file path (for saving/loading model parameters).
     * @note Must be called before init(). If nullptr is set, persistence is disabled.
     * @param path Path string or nullptr.
     */
    void set_param_path(const char* path);

    /**
     * @brief Initialize the model (build layers, allocate memory).
     * @return true if initialization succeeded, false otherwise.
     */
    bool init();

    /**
     * @brief Run inference for a single input.
     * @param input_buffer Pointer to flattened input (float array).
     * @return Predicted label index.
     */
    uint32_t infer(float* input_buffer);

    /**
     * @brief Test the model on a dataset.
     * @param ds Reference to DatasetBase-derived dataset.
     * @param num_samples Number of samples to test.
     */
    void test(DatasetBase& ds, uint32_t num_samples);

    /**
     * @brief Train the model on a dataset.
     *
     * @param ds Dataset to train on.
     * @param num_samples Number of samples to use.
     * @param batch_size Training batch size.
     * @param num_epoch Number of training epochs.
     * @param retrain If true, retrain from scratch; otherwise, continue training.
     * @param early_stopping If true, stop early when target loss is reached.
     * @param early_stopping_target_loss Loss threshold for early stopping.
     */
    void train(DatasetBase& ds,
               uint32_t num_samples,
               uint32_t batch_size,
               uint32_t num_epoch,
               bool retrain,
               bool early_stopping,
               float early_stopping_target_loss);

private:
    // ============
    // Model state
    // ============

    aimodel_t model;              /**< AIfES model definition */
    void* parameter_memory;       /**< Memory for model parameters */
    void* training_memory;        /**< Memory for training state */
    void* inference_memory;       /**< Memory for inference state */

    const char* params_file_path; /**< Path to store/load parameters (user-specified) */
    FileAdapter* adapter;         /**< File adapter used to read data from storage */


    // ===================
    // Internal functions
    // ===================

    /** @brief Build the MNIST CNN model. */
    bool build_model();

    /** @brief Load model parameters from persistent storage. */
    bool load_model_parameters();

    /** @brief Store model parameters to persistent storage. */
    bool store_model_parameters();

    /** @brief Allocate memory for model parameters. */
    bool allocate_parameter_memory();

    /** @brief Free memory used for model parameters. */
    void free_parameter_memory();

    /**
     * @brief Allocate memory for training.
     * @param optimizer Optimizer instance.
     * @return true if allocation succeeded, false otherwise.
     */
    bool allocate_training_memory(aiopti_t* optimizer);

    /** @brief Free memory used for training. */
    void free_training_memory();

    /** @brief Allocate memory for inference. */
    bool allocate_inference_memory();

    /** @brief Free memory used for inference. */
    void free_inference_memory();
};
