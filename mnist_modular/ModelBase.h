#pragma once
#include <vector>
#include <functional>
#include <cstddef>
#include "logger.h"

/**
 * @brief Configuration for a model instance.
 *
 * Defines input/output shapes, allocation policy, and custom allocators.
 * Allocator functions are required; if not provided, defaults to std::malloc/free.
 */
struct ModelConfig {
    std::vector<uint32_t> input_shape;    /**< Shape of model input (e.g., {1,28,28}) */
    std::vector<uint32_t> output_shape;   /**< Shape of model output (e.g., {10}) */

    /** User-provided allocator function (must not be nullptr) */
    std::function<void*(size_t)> allocator_fn = nullptr;

    /** User-provided free function (must not be nullptr) */
    std::function<void(void*)> free_fn = nullptr;
};

/**
 * @brief Base class for ML models.
 *
 * Provides common configuration handling and allocator support.
 * Derived models can implement training, inference, and testing.
 */
class ModelBase {
public:
    /**
     * @brief Construct a ModelBase with the given configuration.
     *
     * Ensures allocator_fn and free_fn are valid. If not, falls back to malloc/free and logs a warning.
     *
     * @param cfg Model configuration.
     */
    explicit ModelBase(const ModelConfig& cfg);

    /** Virtual destructor */
    virtual ~ModelBase();

    /** @brief Access model configuration */
    const ModelConfig& get_config() const { return config; }

    /** @brief Return input shape */
    const std::vector<uint32_t>& get_input_shape() const { return config.input_shape; }

    /** @brief Return output shape */
    const std::vector<uint32_t>& get_output_shape() const { return config.output_shape; }

protected:
    ModelConfig config; /**< Model configuration (shapes, allocators) */
};
