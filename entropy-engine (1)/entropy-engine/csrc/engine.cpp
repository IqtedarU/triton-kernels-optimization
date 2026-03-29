// engine.cpp — Bare-Metal CUBIN Launcher for Triton 3.x AOT Kernels

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

#define CUDA_DRIVER_CHECK(call, msg) do {                                    \
    CUresult res = (call);                                                   \
    if (res != CUDA_SUCCESS) {                                               \
        const char* err_str = nullptr;                                       \
        cuGetErrorString(res, &err_str);                                     \
        throw std::runtime_error(                                            \
            std::string(msg) + ": " + (err_str ? err_str : "unknown"));      \
    }                                                                        \
} while(0)

CUfunction load_cubin(const std::string& filepath, const std::string& func_name) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open .cubin file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read .cubin file: " + filepath);
    }

    CUmodule module;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&module, buffer.data()),
        "cuModuleLoadData failed for " + filepath);

    CUfunction function;
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&function, module, func_name.c_str()),
        "cuModuleGetFunction failed for '" + func_name + "' in " + filepath);

    return function;
}

// ABI struct matching PTX: 5 ptrs + 6 i32s + 2 ptrs = 13 params = 80 bytes
struct alignas(8) TritonABIArgs {
    void* d_a;          // param 0:  offset 0
    void* d_b;          // param 1:  offset 8
    void* d_c;          // param 2:  offset 16
    void* d_a_scale;    // param 3:  offset 24
    void* d_b_scale;    // param 4:  offset 32
    int32_t M;          // param 5:  offset 40
    int32_t N;          // param 6:  offset 44
    int32_t K;          // param 7:  offset 48
    int32_t stride_am;  // param 8:  offset 52
    int32_t stride_bk;  // param 9:  offset 56
    int32_t stride_cm;  // param 10: offset 60
    void* scratch_0;    // param 11: offset 64
    void* scratch_1;    // param 12: offset 72
    // Total: 80 bytes
};

class CubinEngine {
private:
    CUfunction int8_kernel_;
    int shared_mem_bytes_;
    int num_warps_;
    int block_m_;
    int block_n_;

public:
    CubinEngine(
        const std::string& cubin_path,
        const std::string& func_name = "gemm_int8_kernel",
        int shared_mem = 49152,
        int num_warps = 4,
        int block_m = 64,
        int block_n = 128
    ) : shared_mem_bytes_(shared_mem),
        num_warps_(num_warps),
        block_m_(block_m),
        block_n_(block_n)
    {
        static_assert(sizeof(TritonABIArgs) == 80, "TritonABIArgs size mismatch");
        static_assert(offsetof(TritonABIArgs, M) == 40, "M offset mismatch");
        static_assert(offsetof(TritonABIArgs, scratch_0) == 64, "scratch_0 offset mismatch");

        int8_kernel_ = load_cubin(cubin_path, func_name);

        CUDA_DRIVER_CHECK(
            cuFuncSetAttribute(int8_kernel_,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_mem_bytes_),
            "cuFuncSetAttribute (shared memory) failed");

        std::cout << "[CubinEngine] Loaded: " << cubin_path << std::endl;
        std::cout << "  func=" << func_name
                  << " shared=" << shared_mem_bytes_
                  << " warps=" << num_warps_
                  << " BLOCK_M=" << block_m_
                  << " BLOCK_N=" << block_n_ << std::endl;
        std::cout << "  sizeof(TritonABIArgs)=" << sizeof(TritonABIArgs) << std::endl;
    }

    torch::Tensor forward(
        torch::Tensor a_int8,
        torch::Tensor b_int8,
        torch::Tensor a_scale,
        torch::Tensor b_scale
    ) {
        TORCH_CHECK(a_int8.dtype() == torch::kInt8, "a must be int8");
        TORCH_CHECK(b_int8.dtype() == torch::kInt8, "b must be int8");
        TORCH_CHECK(a_scale.dtype() == torch::kFloat32, "a_scale must be fp32");
        TORCH_CHECK(b_scale.dtype() == torch::kFloat32, "b_scale must be fp32");
        TORCH_CHECK(a_int8.is_contiguous(), "a must be contiguous");
        TORCH_CHECK(b_int8.is_contiguous(), "b must be contiguous");

        int32_t M = a_int8.size(0);
        int32_t K = a_int8.size(1);
        int32_t N = b_int8.size(1);

        TORCH_CHECK(a_int8.size(1) == b_int8.size(0),
            "K mismatch: a is [", M, ",", K, "], b is [", b_int8.size(0), ",", N, "]");

        auto output = torch::empty({M, N},
            torch::TensorOptions().dtype(torch::kFloat16).device(a_int8.device()));

        TritonABIArgs args = {
            a_int8.data_ptr(),
            b_int8.data_ptr(),
            output.data_ptr(),
            a_scale.data_ptr(),
            b_scale.data_ptr(),
            M, N, K,
            (int32_t)a_int8.stride(0),
            (int32_t)b_int8.stride(0),
            (int32_t)output.stride(0),
            nullptr,
            nullptr,
        };

        size_t arg_size = sizeof(TritonABIArgs);
        void* launch_config[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, &args,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
            CU_LAUNCH_PARAM_END
        };

        int grid_m = (M + block_m_ - 1) / block_m_;
        int grid_n = (N + block_n_ - 1) / block_n_;
        int grid_x = grid_m * grid_n;
        int threads = num_warps_ * 32;

        CUDA_DRIVER_CHECK(
            cuLaunchKernel(int8_kernel_,
                grid_x, 1, 1,
                threads, 1, 1,
                shared_mem_bytes_,
                nullptr,
                nullptr,
                launch_config),
            "cuLaunchKernel failed");

        return output;
    }

    static CubinEngine from_metadata(
        const std::string& cubin_path,
        const std::string& meta_json_path,
        const std::string& func_name = "gemm_int8_kernel"
    ) {
        std::ifstream f(meta_json_path);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + meta_json_path);
        std::stringstream buf;
        buf << f.rdbuf();
        std::string json = buf.str();

        auto extract_int = [&](const std::string& key) -> int {
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
            pos = json.find(":", pos);
            auto start = json.find_first_of("0123456789", pos);
            auto end = json.find_first_not_of("0123456789", start);
            return std::stoi(json.substr(start, end - start));
        };

        int shared  = extract_int("shared");
        int warps   = extract_int("num_warps");
        int block_m = extract_int("BLOCK_M");
        int block_n = extract_int("BLOCK_N");

        std::cout << "[from_metadata] shared=" << shared
                  << " warps=" << warps
                  << " BLOCK_M=" << block_m
                  << " BLOCK_N=" << block_n << std::endl;

        return CubinEngine(cubin_path, func_name, shared, warps, block_m, block_n);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CubinEngine>(m, "CubinEngine")
        .def(pybind11::init<const std::string&, const std::string&, int, int, int, int>(),
            pybind11::arg("cubin_path"),
            pybind11::arg("func_name") = "gemm_int8_kernel",
            pybind11::arg("shared_mem") = 49152,
            pybind11::arg("num_warps") = 4,
            pybind11::arg("block_m") = 64,
            pybind11::arg("block_n") = 128)
        .def_static("from_metadata", &CubinEngine::from_metadata,
            pybind11::arg("cubin_path"),
            pybind11::arg("meta_json_path"),
            pybind11::arg("func_name") = "gemm_int8_kernel")
        .def("forward", &CubinEngine::forward);
}