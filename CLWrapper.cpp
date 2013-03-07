#include <fstream>
#include <iostream>
#include <iterator>

#include "CLWrapper.hpp"

CLWrapper::CLWrapper() {
  try {
    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    context = cl::Context(devices);

    // create command queue
    queue = cl::CommandQueue(context, devices[0]);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Kernel CLWrapper::compileKernel(const char *kernelFile, const char *kernelName) {
  try {
    // load opencl source
    std::ifstream cl_file(kernelFile);
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), 
                                                  cl_string.length() + 1));

    // create program
    cl::Program program(context, source);

    // compile opencl source
    program.build(devices);

    // load named kernel from opencl source
    return cl::Kernel(program, kernelName);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Buffer CLWrapper::uploadData(cl_mem_flags flags, size_t size, void *ptr) {
  try {
    // allocate device buffer to hold data
    std::cerr << "TODO: Change buffer uploading to be async" << std::endl;
    return cl::Buffer(flags, size, ptr);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CLWrapper::run(cl::Kernel kernel) {
  try {
    // execute kernel
    queue.enqueueTask(kernel);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}
