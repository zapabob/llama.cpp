#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status) << '\n';
        return 1;
    }

    const char * expected_name = std::getenv("SM120_EXPECTED_GPU");
    if (expected_name == nullptr || *expected_name == '\0') {
        expected_name = "RTX 5060 Ti";
    }

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};
        const cudaError_t status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(status) << '\n';
            return 1;
        }
        std::cout << "device=" << device << " name=" << properties.name
                  << " compute_capability=" << properties.major << '.' << properties.minor << '\n';
        if (properties.major == 12 && properties.minor == 0 && std::strstr(properties.name, expected_name) != nullptr) {
            return 0;
        }
    }

    std::cerr << "no configured SM120 GPU matched expected name: " << expected_name << '\n';
    return 1;
}
