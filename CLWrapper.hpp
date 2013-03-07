#ifndef CLWRAPPER_HPP_
#define CLWRAPPER_HPP_

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class CLWrapper {
public:
  CLWrapper();
  cl::Kernel compileKernel(const char *kernelFile, const char *kernelName);
  cl::Buffer uploadData(cl_mem_flags flags, size_t size, void *ptr);
  void downloadData(cl::Buffer buff, size_t size, void *ptr);
  void run(cl::Kernel kernel, cl::NDRange global, cl::NDRange local);
  void flush();

private:
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  cl::Context context;
  cl::CommandQueue queue;
};

#endif /* CLWRAPPER_HPP_ */
