#include <fstream>
#include <iostream>
#include <iterator>

#include "CLWrapper.hpp"

CLWrapper::CLWrapper() {
  try {
    cl::Platform::get(&platforms);

    // create context
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

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
    try {
      program.build(devices);
    } catch(cl::Error e) {
      std::cerr << "Build Status:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
      std::cerr << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
      std::cerr << "Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
      throw e;
    }

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
    cl::Buffer buff(context, flags, size);
    queue.enqueueWriteBuffer(buff, CL_TRUE, 0, size, ptr, NULL);
    return buff;
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CLWrapper::downloadData(cl::Buffer buff, size_t size, void *ptr) {
  try {
    queue.enqueueReadBuffer(buff, CL_TRUE, 0, size, ptr, NULL);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CLWrapper::run(cl::Kernel kernel, cl::NDRange global, cl::NDRange local) {
  try {
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
  } catch(cl::Error e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CLWrapper::flush() {
  queue.finish();
}
