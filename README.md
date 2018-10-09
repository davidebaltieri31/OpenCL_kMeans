## OpenCL/SSE/Multithreaded optimized k-means<br>

Author Davide Baltieri<br>
License LGPLv3<br>

kmeans.h/.cpp is a direct implementation of the usual algorithm in OpenCL, it shows a modest speedup, because the original algorith was not designed with GPUs in mind. kmeans2.h/.cpp implements a series of optimizations for GPUs. What on paper (and on a CPU) looks like a slower algorithm turns out to be a lot faster then the usual one on a GPU

It would be awesome if you'd let me know if you use this code...<br>
<br>

## Currently only for Windows/Visual Studio

* Requires Microsoft's Parallel Patterns Library (included in VS2012/VS2013)
* Requires SSE support
* Requires OpenCL libs (Available with the NVIDIA CUDA Sdk or ATI equivalent, must link to OpenCL.lib and be able to include CL/cl.h)<br>




