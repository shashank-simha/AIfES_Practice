#pragma once
#include "AIfESModel.h"
#include "DatasetBase.h"
#include <cstdint>
#include <vector>

/**
 * @brief Generic base class for classification tasks.
 *
 * Provides train, test, and inference routines using AIfES runtime.
 * Derived classes only need to define the model architecture and set input/output shapes.
 */
class ClassificationModel : public AIfESModel {
public:
    /**
     * @brief Construct a ClassificationModel.
     *
     * @param cfg Model configuration (shapes, allocators).
     * @param param_path Optional path to parameter file for persistence (nullptr = disabled).
     * @param adapter FileAdapter instance for platform-agnostic I/O.
     */
    explicit ClassificationModel(const ModelConfig& cfg,
                                 const char* param_path = nullptr,
                                 FileAdapter* adapter = nullptr);

    /** Destructor */
    virtual ~ClassificationModel() = default;

    /**
     * @brief Run inference for one sample.
     *
     * Builds an input tensor from config.input_shape, runs a forward pass,
     * performs argmax on the output tensor and returns the predicted class index.
     *
     * @param input_buffer Pointer to normalized float input (C-order flattened)
     * @return Predicted class index on success, INVALID_CLASS on failure
     */
    uint32_t infer(float* input_buffer);

    /**
     * @brief Train the model on a dataset
     *
     * @param ds Dataset to train on
     * @param num_samples Number of samples to train
     * @param batch_size Batch size
     * @param num_epoch Number of epochs
     * @param retrain If true, reinitialize parameters
     * @param early_stopping If true, break when target loss achieved
     * @param early_stopping_target_loss Loss threshold for early stopping
     */
    void train(DatasetBase& ds,
               uint32_t num_samples,
               uint32_t batch_size,
               uint32_t num_epoch,
               bool retrain = false,
               bool early_stopping = false,
               float early_stopping_target_loss = 0.0f);

    /**
     * @brief Evaluate model on a dataset and report accuracy
     *
     * @param ds Dataset to evaluate
     * @param num_samples Number of samples to test
     */
    void test(DatasetBase& ds, uint32_t num_samples);

    /**
     * @brief Optional: set a custom loss function.
     *
     * @param loss_fn Pointer to custom loss function.
     * @note If not set, defaults to cross-entropy loss.
     */
    void set_loss(void* loss_fn);

    /**
     * @brief Optional: set a custom optimizer.
     *
     * @param optimizer Pointer to custom optimizer.
     * @note If not set, defaults to SGD optimizer.
     */
    void set_optimizer(void* optimizer);

protected:
    ailoss_t* loss = nullptr;       ///< Loss function pointer
    aiopti_t* optimizer = nullptr;  ///< Optimizer pointer

    static constexpr uint32_t INVALID_CLASS = 0xFFFFFFFF;

    /**
     * @brief Create an aitensor_t from a buffer and shape vector.
     *
     * @param batch_size Batch size
     * @param shape Vector of dimensions (excluding batch)
     * @param data Pointer to float data
     * @return Initialized aitensor_t
     */
    inline aitensor_t create_tensor(uint32_t batch_size, const std::vector<uint32_t>& shape, float* data)
    {
        std::vector<uint16_t> tensor_shape;
        tensor_shape.push_back(static_cast<uint16_t>(batch_size));
        for (auto d : shape) tensor_shape.push_back(static_cast<uint16_t>(d));

        aitensor_t tensor;
        tensor.dtype = aif32;
        tensor.dim = static_cast<uint8_t>(tensor_shape.size());
        tensor.shape = tensor_shape.data();
        tensor.tensor_params = 0;;
        tensor.data = data;

        return tensor;
    }
};
