cmake_minimum_required(VERSION 3.18)

# 设置项目名称和语言
project(CudaHelloWorld LANGUAGES CXX CUDA)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 11)

# 查找CUDA
find_package(CUDA REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)


# 添加可执行文件
add_executable(hello_world hello_world.cu)
add_executable(add add.cu)
add_executable(get_device_info get_device_info.cu)

# 设置CUDA属性
set_property(TARGET hello_world PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
set_property(TARGET add PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
set_property(TARGET get_device_info PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

# 设置编译选项
target_compile_options(hello_world PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# 链接CUDA库
target_link_libraries(hello_world ${CUDA_LIBRARIES})
target_link_libraries(add ${CUDA_LIBRARIES})
target_link_libraries(get_device_info ${CUDA_LIBRARIES})