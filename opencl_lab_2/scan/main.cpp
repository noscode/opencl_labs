#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <sstream>

size_t const block_size = 256;

std::vector<float> calculate_sums(const std::vector<float>& input, const std::vector<float>& sums, std::string name_of_function)
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	std::vector<cl::Kernel> kernels;

	size_t size = input.size();

	std::vector<float> output = std::vector<float>(size, 0);

	try {
		// create platform
		cl::Platform::get(&platforms);
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		// create context
		cl::Context context(devices);

		// create command queue
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		// load opencl source
		std::ifstream cl_file("scan.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

		// create program
		cl::Program program(context, source);

		// compile opencl source
		std::stringstream ss;
		ss << "-D BLOCK_SIZE=" << block_size;
		program.build(devices, ss.str().c_str());

		// allocate device buffer to hold message
		cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(float) * size);
		cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);
		cl::Buffer dev_sums;
		if (name_of_function == "aggregate_sums")
			dev_sums = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * sums.size());

		// copy from cpu to gpu
		queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * size, &input[0]);
		if (name_of_function == "aggregate_sums")
			queue.enqueueWriteBuffer(dev_sums, CL_TRUE, 0, sizeof(float) * sums.size(), &sums[0]);


		queue.finish();

		// load named kernel from opencl source
		cl::Kernel kernel_hs(program, name_of_function.c_str());
		cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(block_size));
		cl::Event event;
		if (name_of_function == "aggregate_sums")
			event = scan_hs(dev_input, dev_output, dev_sums);
		else
			event = scan_hs(dev_input, dev_output, cl::__local(sizeof(float) * block_size), cl::__local(sizeof(float) * block_size));
	 
		event.wait();
		cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		cl_ulong elapsed_time = end_time - start_time;

		queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * size, &output[0]);

		std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;
		
		return output;

	}
	catch (cl::Error e)
	{
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
		throw(e);
		
	}
}

int main()
{
	// read data
	std::ifstream input_fs("input.txt");
	size_t size;
	input_fs >> size;
	std::vector<float> input = std::vector<float>(size, 0);
	for (size_t i = 0; i < size; i++)
		input_fs >> input[i];

	// add 0 at the end
	// here some problems with std::max
	size_t original_size = input.size();
	size_t real_size = block_size >= input.size() ? block_size : input.size();
	while (real_size % block_size != 0)
		real_size++;

	while (input.size() < real_size)
		input.push_back(0);

	// calc sums and prepare to aggregate them
	std::vector<float> all_sums = calculate_sums(input, input, "scan_hillis_steele");
	std::vector<float> sums = std::vector<float> (1, 0);
	for (size_t i = block_size - 1; i < input.size(); i += block_size) {
		sums.push_back(sums.back() + all_sums[i]);
	}

	// get final result
	std::vector<float> result = calculate_sums(all_sums, sums, "aggregate_sums");
	// delete extra elements
	while (result.size() > original_size)
		result.pop_back();
	
	// print output
	std::ofstream output_fs("output.txt");
	output_fs << std::setprecision(3);
	for (float el : result)
		output_fs << el << " ";
}
