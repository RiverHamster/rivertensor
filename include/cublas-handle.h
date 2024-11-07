#pragma once

#include <cublas_v2.h>
#include "macros.h"

namespace ten {
// singleton object to manage a global cuBLAS handle
class CublasHandleManager {
public:
    static CublasHandleManager& getInstance() {
        static CublasHandleManager instance;
        return instance;
    }

    cublasHandle_t getHandle() const { return handle_; }

private:
    cublasHandle_t handle_;

    // Private constructor to initialize cuBLAS handle
    CublasHandleManager() {
        blasChkerr(cublasCreate(&handle_));
    }

    // Destructor to clean up cuBLAS handle
    ~CublasHandleManager() {
        cublasDestroy(handle_);
    }

    // Delete copy and move constructors to ensure singleton behavior
    CublasHandleManager(const CublasHandleManager&) = delete;
    CublasHandleManager& operator=(const CublasHandleManager&) = delete;
};

cublasHandle_t get_cublas_handle() {
    return CublasHandleManager::getInstance().getHandle();
}
}