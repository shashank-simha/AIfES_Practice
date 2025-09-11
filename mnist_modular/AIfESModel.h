#pragma once
#include "ModelBase.h"
#include "FileAdapter.h"
#include <aifes.h>

/**
 * @brief Task-agnostic base class for AIfES models.
 *
 * Handles AIfES runtime lifecycle, memory management, and parameter persistence.
 * Inherits from ModelBase. Does not assume classification or regression tasks.
 */
class AIfESModel : public ModelBase {
public:
    /**
     * @brief Construct an AIfESModel with the given configuration.
     *
     * @param cfg Model configuration (shapes, allocators).
     * @param param_path Optional path to parameter file for persistence (nullptr = disabled).
     * @param adapter FileAdapter instance for platform-agnostic I/O.
     */
    explicit AIfESModel(const ModelConfig& cfg,
                        const char* param_path = nullptr,
                        FileAdapter* adapter = nullptr);

    /** Virtual destructor frees allocated memory */
    virtual ~AIfESModel();

    /**
     * @brief Initialize the model.
     *
     * - Calls build_model() (virtual)
     * - Compiles model
     * - Allocates parameter & inference memory
     * - Loads parameters if path & adapter provided
     *
     * @return true on success, false on failure
     */
    virtual bool init();

    /**
     * @brief Set or update the parameter file path.
     * @param path Path string or nullptr
     */
    void set_param_path(const char* path);

protected:
    aimodel_t model;              /**< AIfES model structure */
    void* parameter_memory;       /**< Memory for model parameters */
    void* inference_memory;       /**< Memory for inference computations */
    void* training_memory;        /**< Memory for training computations */

    const char* params_file_path; /**< Parameter file path */
    FileAdapter* adapter;         /**< Platform-agnostic file adapter */

    /** @brief Allocate memory for model parameters */
    bool allocate_parameter_memory();

    /** @brief Free parameter memory */
    void free_parameter_memory();

    /** @brief Allocate memory for inference */
    bool allocate_inference_memory();

    /** @brief Free inference memory */
    void free_inference_memory();

    /** @brief Allocate memory for training with the given optimizer */
    bool allocate_training_memory(aiopti_t* optimizer);

    /** @brief Free training memory */
    void free_training_memory();

    /** @brief Load parameters from persistent storage */
    bool load_model_parameters();

    /** @brief Store parameters to persistent storage */
    bool store_model_parameters();

    /**
     * @brief Build the model layers and connections.
     *
     * Pure virtual: must be implemented in dataset-specific subclass.
     */
    virtual bool build_model() = 0;
};
