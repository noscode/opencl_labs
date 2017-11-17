#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <fstream>


int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0]);

      // load opencl source
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);


	  //read sizes
	  std::cout << "begin to read" << std::endl;
	  std::ifstream in("input.txt");
      size_t N, M; 
	  in >> N >> M;
	  std::cout << N << " " << M << std::endl;

      // compile opencl source
	  size_t const block_size = 16ul < N ? 16ul : N;
	  try
	  {
		  program.build(devices);
	  }
	  catch (cl::Error const & e)
	  {			
		  std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		  std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
		  std::cout << log_str;
		  return 0;
	  }

      // create a message to send to kernel
      size_t const matrix_size = N * N;
	  size_t const kernel_size = M * M;

      std::vector<float> a(matrix_size, 0);
      std::vector<float> b(kernel_size, 0);
      std::vector<float> c(matrix_size, 0);

	  std::cout << "start reading matrices" << std::endl;

      for (size_t i = 0; i < N; ++i)
      {
         for (size_t j = 0; j < N; ++j)
         {
            in >> a[i * N + j];
         }
      }
	  for (size_t i = 0; i < M; ++i)
      {
         for (size_t j = 0; j < M; ++j)
         {
            in >> b[i * M + j];
         }
      }

	  std::cout << "end reading matrices" << std::endl;

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * matrix_size);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * matrix_size);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * matrix_size, a.data());
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * matrix_size, b.data());

      // load named kernel from opencl source
	  cl::Kernel kernel(program, "convolution");
	  cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(N, N), cl::NDRange(block_size, block_size));
	  convolution(dev_a, dev_b, dev_c, (int) N, (int) M);

      

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * matrix_size, c.data());

	  std::cout << "start to write to file" << std::endl;
      std::ofstream out("output.txt");
//	  out << std::setprecision(3);
//	  std::cout << std::setprecision(3);
	  for (size_t i = 0; i < N; ++i)
	  {
		  for (size_t j = 0; j < N; ++j)
		  {
			  size_t index = i * N + j;
			  out << c[index] << " ";
			  std::cout << c[index] << " ";
          }
		  out << std::endl;
		  std::cout << std::endl;
	  }

      std::cout << "finished" << std::endl;
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}