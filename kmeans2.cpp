#include "kmeans2.h"
#include <windows.h>
#include <ppl.h>
#include <atomic>
#include <mutex>
#include <iostream>
#include <chrono>
#include <stdio.h>

#define USE_OCL_1_2
#undef min
#undef max

#pragma region KERNELS
const char* kernels_source = "\
#define __NUM_POINTS_EXPANDED__ %d\n\
#define __NUM_POINTS__ %d\n\
#define __NUM_CENTROIDS_EXPANDED__ %d\n\
#define __NUM_CENTROIDS__ %d\n\
#define __NUM_FEATURES_EXPANDED__ %d\n\
#define __NUM_FEATURES__ %d\n\
#define __TILE_SIZE__ %d\n\
__kernel void compute_dists(const __global float* points,\n\
							const __global float* centroids,\n\
							__global float* C) {\n\
	// Thread identifiers\n\
	const int row = get_local_id(0); // Local row ID (max: __TILE_SIZE__)\n\
	const int col = get_local_id(1); // Local col ID (max: __TILE_SIZE__)\n\
	const int globalRow = __TILE_SIZE__*get_group_id(0) + row; // Row ID of C (0..__NUM_POINTS_EXPANDED__)\n\
	const int globalCol = __TILE_SIZE__*get_group_id(1) + col; // Col ID of C (0..__NUM_CENTROIDS_EXPANDED__)\n\
	// Local memory to fit a tile of __TILE_SIZE__*__TILE_SIZE__ elements of A and B\n\
	__local float Asub[__TILE_SIZE__][__TILE_SIZE__];\n\
	__local float Bsub[__TILE_SIZE__][__TILE_SIZE__];\n\
	// Initialise the accumulation register\n\
	float acc = 0.0f;\n\
	// Loop over all tiles\n\
	const int numTiles = __NUM_FEATURES_EXPANDED__ / __TILE_SIZE__;\n\
	for (int t = 0; t<numTiles; t++) {\n\
		// Load one tile of A and B into local memory\n\
		const int tiledRow = __TILE_SIZE__*t + row;\n\
		const int tiledCol = __TILE_SIZE__*t + col;\n\
		Asub[col][row] = points[globalRow*__NUM_FEATURES_EXPANDED__ + tiledCol];\n\
		Bsub[col][row] = centroids[globalCol*__NUM_FEATURES_EXPANDED__ + tiledRow];\n\
		// Synchronise to make sure the tile is loaded\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		// Perform the computation for a single tile\n\
		for (int k = 0; k<__TILE_SIZE__; k++) {\n\
			float tmp = (Asub[k][row] - Bsub[col][k]);\n\
			acc += (tmp * tmp);\n\
		}\n\
		// Synchronise before loading the next tile\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	// Store the final result in C\n\
	C[globalRow*__NUM_CENTROIDS_EXPANDED__ + globalCol] = acc;\n\
}\n\
__kernel void find_membership(__global float* dists, __global int* membership) {\n\
	int idx = get_global_id(0);\n\
	int i = 0;\n\
	float min_distance = dists[idx*__NUM_CENTROIDS_EXPANDED__];\n\
	membership[idx] = 0;\n\
	for (i = 1; i < __NUM_CENTROIDS__; i++) {\n\
		float d = dists[idx*__NUM_CENTROIDS_EXPANDED__ + i];\n\
		if (min_distance > d) {\n\
			min_distance = d;\n\
			membership[idx] = i;\n\
		}\n\
	}\n\
}\n\
__kernel void compute_centroids_first_step(	__global float* points,\n\
											__global float* centroid_groups, __global int* centroid_groups_n,\n\
											__global int* membership, int group_size) {\n\
	float l_c_g[__NUM_CENTROIDS__*__NUM_FEATURES__];\n\
	int l_c_g_n[__NUM_CENTROIDS__];\n\
	int idx = get_global_id(0);\n\
	int p_id = idx*group_size;\n\
	__global float* t_from = points + (p_id*__NUM_FEATURES_EXPANDED__);\n\
	for(int i=0;i<__NUM_CENTROIDS__;i++) {\n\
		for(int k=0;k<__NUM_FEATURES__;k++) {\n\
			l_c_g[i * __NUM_FEATURES__ + k] = 0.0;\n\
		}\n\
		l_c_g_n[i] = 0;\n\
	}\n\
	for(int i=0;i<group_size;i++) {\n\
		if(p_id+i <__NUM_POINTS__) {\n\
			int c_idx = membership[p_id+i];\n\
			for(int k=0;k<__NUM_FEATURES__;k++) {\n\
				l_c_g[c_idx * __NUM_FEATURES__ + k] += t_from[i*__NUM_FEATURES_EXPANDED__ +k];\n\
			}\n\
			l_c_g_n[c_idx] += 1;\n\
		}\n\
	}\n\
	__global float* t_to = centroid_groups + (idx*__NUM_CENTROIDS__*__NUM_FEATURES__);\n\
	__global int* t_to_n = centroid_groups_n + (idx*__NUM_CENTROIDS__);\n\
	for(int i=0;i<__NUM_CENTROIDS__;i++) {\n\
		for(int k=0;k<__NUM_FEATURES__;k++) {\n\
			t_to[i * __NUM_FEATURES__ + k] = l_c_g[i * __NUM_FEATURES__ + k];\n\
		}\n\
		t_to_n[i] = l_c_g_n[i];\n\
	}\n\
}\n\
__kernel void compute_centroids_second_step(__global float* in_centroid_groups, __global int* in_centroid_groups_n,\n\
											__global float* out_centroid_groups, __global int* out_centroid_groups_n,\n\
											int group_size, int num_groups) {\n\
	float l_c_g[__NUM_CENTROIDS__*__NUM_FEATURES__];\n\
	int l_c_g_n[__NUM_CENTROIDS__];\n\
	int idx = get_global_id(0);\n\
	int group_step = idx*group_size;\n\
	for(int i=0;i<__NUM_CENTROIDS__;i++) {\n\
		for(int k=0;k<__NUM_FEATURES__;k++) {\n\
			l_c_g[i * __NUM_FEATURES__ + k] = 0.0;\n\
		}\n\
		l_c_g_n[i] = 0;\n\
	}\n\
	for(int i=0;(i<group_size) && ((group_step + i)<num_groups);i++) {\n\
		__global float* t_from = in_centroid_groups + ((group_step + i)*__NUM_CENTROIDS__*__NUM_FEATURES__);\n\
		__global int* t_from_n = in_centroid_groups_n + ((group_step + i)*__NUM_CENTROIDS__);\n\
		for(int j=0;j<__NUM_CENTROIDS__;j++) {\n\
			int v = t_from_n[j];\n\
			if(v != 0) {\n\
				for(int k=0;k<__NUM_FEATURES__;k++) {\n\
					l_c_g[j*__NUM_FEATURES__ + k] += t_from[j*__NUM_FEATURES__ + k];\n\
				}\n\
				l_c_g_n[j] += v;\n\
				t_from_n[j] = 0;\n\
			}\n\
		}\n\
	}\n\
	__global float* t_to = out_centroid_groups + (idx*__NUM_CENTROIDS__*__NUM_FEATURES__);\n\
	__global int* t_to_n = out_centroid_groups_n + (idx*__NUM_CENTROIDS__);\n\
	for(int i=0;i<__NUM_CENTROIDS__;i++) {\n\
		for(int k=0;k<__NUM_FEATURES__;k++) {\n\
			t_to[i * __NUM_FEATURES__ + k] = l_c_g[i * __NUM_FEATURES__ + k];\n\
		}\n\
		t_to_n[i] = l_c_g_n[i];\n\
	}\n\
}";
#pragma endregion
/*----< round up to nearest multiple of x >--------------------------------------------------------*/
__inline uint32_t round_up_multiple(uint32_t numToRound, uint32_t multiple)
{
	return ((numToRound + multiple - 1) / multiple) * multiple;
	//return (numToRound + multiple - 1) & -multiple;
}
/*----< get nearest (ceil) power of two >----------------------------------------------------------*/
__inline int next_power_2(int x) {
	int result = x - 1;
	result = result | (result >> 16);
	result = result | (result >> 8);
	result = result | (result >> 4);
	result = result | (result >> 2);
	result = result | (result >> 1);
	result += 1;
	return result;
}
/*----< get nearest (floor) power of two >---------------------------------------------------------*/
__inline int prev_power_2(int x) {
	return int(pow(2, floor(log(x) / log(2))));
}
/*----< fast float resqrt >------------------------------------------------------------------------*/
__inline float fast_resqrt(const float x)
{
	/* 32-bit version */
	union {
		float x;
		int  i;
	} u;
	float xhalf = 0.5f * x;
	/* convert floating point value in RAW integer */
	u.x = x;
	/* gives initial guess y0 */
	u.i = 0x5f3759df - (u.i >> 1);
	/*u.i = 0xdf59375f - (u.i>>1);*/
	/* two Newton steps */
	u.x = u.x * (1.5f - xhalf*u.x*u.x);
	u.x = u.x * (1.5f - xhalf*u.x*u.x);
	return u.x;
}
/*----< fast float sqrt >--------------------------------------------------------------------------*/
__inline float fast_sqrt(const float x)
{
	return (x < 1e-8) ? 0 : x * fast_resqrt(x);
}
/*----< multi-dimensional spatial Euclid distance square >-----------------------------------------*/
__inline float euclid_dist_2(const float *pt1, const float *pt2, const int numdims)
{
	int i;
	float ans = 0.0;
	//begin SSE
	__m128 v0, v1, dif, mul, sum;
	const float* _pt1 = pt1;
	const float* _pt2 = pt2;
	sum = _mm_set1_ps(0.0f);
	int sse_steps = numdims - numdims % 4;
	for (i = 0; i < sse_steps; i += 4) {
		v0 = _mm_loadu_ps(_pt1);
		v1 = _mm_loadu_ps(_pt2);
		dif = _mm_sub_ps(v0, v1);
		mul = _mm_mul_ps(dif, dif);
		sum = _mm_add_ps(sum, mul);
		_pt1 += 4;
		_pt2 += 4;
	}
	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);
	_mm_store_ss(&ans, sum);
	//end SSE
	//do standard pass if necessary
	for (; i < numdims; i++) {
		ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
	}
	return(ans);
}
/*----< find_nearest_point() >---------------------------------------------------------------------*/
__inline int find_nearest_point(const float  *pt, const int nfeatures, const  float *pts, const int npts, float* output_min_dist)
{
	int index, i;
	float min_dist = FLT_MAX;
	/* find the cluster center id with min distance to pt */
	{
		for (i = 0; i < npts; i++) {
			float dist = euclid_dist_2(pt, pts + nfeatures*i, nfeatures);  /* no need square root */
			index = (dist < min_dist) ? i : index;
			min_dist = (dist < min_dist) ? dist : min_dist;
		}
	}
	*output_min_dist = min_dist;
	return(index);
}
__inline int find_nearest_point(const float  *pt, const int nfeatures, const int nfeatures_expanded, const  float *pts, const int npts, float* output_min_dist)
{
	int index, i;
	float min_dist = FLT_MAX;
	/* find the cluster center id with min distance to pt */
	{
		for (i = 0; i < npts; i++) {
			float dist = euclid_dist_2(pt, pts + nfeatures_expanded*i, nfeatures);  /* no need square root */
			index = (dist < min_dist) ? i : index;
			min_dist = (dist < min_dist) ? dist : min_dist;
		}
	}
	*output_min_dist = min_dist;
	return(index);
}
/*----< rms_err(): calculates RMSE of clustering >-------------------------------------------------*/
float rms_err(const float  *feature_set, const int nfeatures, const int npoints, const float *cluster_centres, const int nclusters)
{
	float  sum_euclid = 0.0f;
	std::mutex err_sum_mutex;
	/* calculate and sum the sqaure of euclidean distance*/
	{
		Concurrency::parallel_for(0, npoints, [&](size_t i)
		{
			const float* pt_data = feature_set + nfeatures * i;
			float dist;
			int nearest_cluster_index = find_nearest_point(pt_data, nfeatures, cluster_centres, nclusters, &dist);
			err_sum_mutex.lock();
			sum_euclid += dist;
			err_sum_mutex.unlock();
		});
	}
	float ret = fast_sqrt(sum_euclid / npoints);
	return(ret);
}
float rms_err(const float  *feature_set, const int nfeatures, const int nfeatures_expanded, const int npoints, const float *cluster_centres, const int nclusters)
{
	float  sum_euclid = 0.0f;
	std::mutex err_sum_mutex;
	/* calculate and sum the sqaure of euclidean distance*/
	{
		Concurrency::parallel_for(0, npoints, [&](size_t i)
		//for (int i = 0; i < npoints; ++i)
		{
			const float* pt_data = feature_set + nfeatures * i;
			float dist;
			int nearest_cluster_index = find_nearest_point(pt_data, nfeatures, nfeatures_expanded, cluster_centres, nclusters, &dist);
			err_sum_mutex.lock();
			sum_euclid += dist;
			err_sum_mutex.unlock();
		//}
		});
	}
	float ret = fast_sqrt(sum_euclid / npoints);
	return(ret);
}
/*----< CPU clustering: computes points membership >-----------------------------------------------*/
void group_by_cluster(const float* points, float* centroids, int* membership, int num_points, int num_centroids, int num_features) {
	//for (int idx = 0; idx < num_points; ++idx) {
	Concurrency::parallel_for(0, num_points, [&](size_t idx) {
		float min_distance = euclid_dist_2(points + (idx * num_features), centroids, num_features);
		membership[idx] = 0;
		for (int i = 1; i < num_centroids; i++) {
			float d = euclid_dist_2(points + (idx * num_features), centroids + (i * num_features), num_features);
			if (min_distance > d) {
				min_distance = d;
				membership[idx] = i;
			}
		}
	});
}
/*----< CPU clustering: computes new cluster centroids >-------------------------------------------*/
void update_centroids(const float* points, int num_points, int* membership, float* centroids, int num_centroids, int num_features) {
	std::vector<float> accum;
	accum.assign(num_centroids*num_features, 0.0f);
	std::vector<int> accum_n;
	accum_n.assign(num_centroids, 0);
	
	/*for (int j = 0; j < num_points; j++) {
		int c_id = membership[j];
		for (int k = 0; k < num_features; k++) {
			accum[c_id*num_features + k] += points[j*num_features + k];
		}
		accum_n[c_id] += 1;
	}*/
	Concurrency::parallel_for(0, num_centroids, [&](size_t i) {
		for (int j = 0; j < num_points; j++) {
			if (membership[j] == i) {
				for (int k = 0; k < num_features; k++) {
					accum[i*num_features + k] += points[j*num_features + k];
				}
				accum_n[i] += 1;
			}
		}
	});

	//for (int j = 0; j < num_centroids; j++) {
	Concurrency::parallel_for(0, num_centroids, [&](size_t j) {
		for (int k = 0; k < num_features; k++) {
			centroids[j*num_features + k] = accum[j*num_features + k] / accum_n[j];
		}
	});
}
/*----< generate initial cluster centroid candidates using kmeans++ >------------------------------*/
void kmeans::kpp(const  float *pts, const int nfeatures, const int npts, float  *centers, int ncent)
{
	int j;
	int n_cluster;
	float sum;
	float *d = new float[npts];
	const float* p;
	m_random_float = std::uniform_real_distribution<float>(0.0f, 1.0f);
	//point p, c;
	//select first centroid at random
	m_random_int = std::uniform_int_distribution<int>(0, npts);
	int first = m_random_int(m_generator);
	memcpy((void*)(centers), (void*)(pts + first*nfeatures), sizeof(float)*nfeatures);

	for (n_cluster = 1; n_cluster < ncent; n_cluster++) 
	{
		sum = 0;
		for (j = 0, p = pts; j < npts; j++, p += nfeatures)
		{
			find_nearest_point(p, nfeatures, centers, n_cluster, d + j);
			sum += d[j];
		}
		sum = m_random_float(m_generator)*sum;
		for (j = 0, p = pts; j < npts; j++, p += nfeatures)
		{
			if ((sum -= d[j]) > 0) continue;
			memcpy((void*)(centers + n_cluster*nfeatures), (void*)(pts + j*nfeatures), sizeof(float)*nfeatures);
			break;
		}
	}
	delete[] d;
}
void kmeans::random(const  float *pts, const int nfeatures, const int npts, float  *centers, int ncent)
{
	std::vector<int> ids;
	ids.assign(npts, 0);
	for (int i = 0; i < npts; ++i) {
		ids[i] = i;
	}
	std::random_shuffle(ids.begin(), ids.end());

	for (int n_cluster = 0; n_cluster < ncent; n_cluster++) {
		memcpy((void*)(centers + n_cluster*nfeatures), (void*)(pts + ids[n_cluster] * nfeatures), sizeof(float)*nfeatures);
	}
}
/*----< initialize OpenCL >------------------------------------------------------------------------*/
int kmeans::initialize(const int use_gpu)
{
	if (cl_initialized) return 0;
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id available_platform_id[10];
	if (clGetPlatformIDs(10, available_platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }

	for (int i = 9; i >= 0; --i) {
		if (available_platform_id[i] == nullptr) continue;
		std::string resultdata;
		resultdata.resize(1000);
		size_t ret_size = 0;
		cl_int res = clGetPlatformInfo(available_platform_id[i],
			CL_PLATFORM_VERSION,
			10000,
			(void*)resultdata.data(),
			&ret_size);
		resultdata.resize(ret_size);
		printf("Trying platform %s in %s mode\n", resultdata.c_str(), use_gpu ? "GPU" : "CPU");
		cl_platform_id platform_id = available_platform_id[i];
		cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };
		device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
		context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, NULL);
		if (!context) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); }
		else break;
	}
	if (!context) { return -1; }

	// get the list of GPUs
	result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
	num_devices = (int)(size / sizeof(cl_device_id));

	if (result != CL_SUCCESS || num_devices < 1) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if (!device_list) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
	if (result != CL_SUCCESS) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_buffer_size, nullptr);
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE=%lld\n", max_buffer_size);

	clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%zd\n", max_work_group_size);

	clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &MAX_WORK_ITEM_DIMENSIONS, nullptr);
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS=%d\n", MAX_WORK_ITEM_DIMENSIONS);

	MAX_WORK_ITEM_SIZES.resize(MAX_WORK_ITEM_DIMENSIONS);
	size_t returned_size;
	clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*MAX_WORK_ITEM_SIZES.size(), &(MAX_WORK_ITEM_SIZES[0]), &returned_size);
	MAX_WORK_ITEM_SIZES.resize(returned_size/sizeof(size_t));
	for (int i = 0; i < MAX_WORK_ITEM_DIMENSIONS;++i)
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES[%d]=%zd\n",i, MAX_WORK_ITEM_SIZES[i]);

	clGetDeviceInfo(device_list[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &LOCAL_MEM_SIZE, 0);
	printf("CL_DEVICE_LOCAL_MEM_SIZE=%lld\n", LOCAL_MEM_SIZE);

	clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &MAX_COMPUTE_UNITS, nullptr);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS=%d\n", MAX_COMPUTE_UNITS);
	
	// create command queue for the first device
#ifndef USE_OCL_1_2
	cmd_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, NULL);
#else
	cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL); //deprecated
#endif
	if (!cmd_queue) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	cl_initialized = true;
	return 0;
}
/*----< shutdown OpenCL >--------------------------------------------------------------------------*/
int kmeans::shutdown()
{
	// release resources
	if (cmd_queue) clReleaseCommandQueue(cmd_queue);
	if (context) clReleaseContext(context);
	if (device_list) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	cl_initialized = false;
	return 0;
}
/*----< compile kernels >--------------------------------------------------------------------------*/
int kmeans::compile_kernels()
{
	//get kernels
	int sourcesize = 0;
	char* source = nullptr;
	sourcesize = int(strlen(kernels_source));
	
	m_kernel_source = new char[sourcesize * 2];
	sprintf_s(m_kernel_source, sourcesize * 2, kernels_source, m_num_points_expanded,m_num_points,m_num_clusters_expanded,m_num_clusters,m_num_features_expanded,m_num_features,m_tile_size);
	
	source = (char*)m_kernel_source;
	// compile kernel code
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	//cl_prog = clCreateProgramWithBuiltInKernels(context, 1, device_list, "group_by_cluster;sum_points;update_centroids;", &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(cl_prog, 0, NULL, NULL, NULL, NULL);
	//err = clBuildProgram(cl_prog, 0, NULL, "-g -s \"F:/Development/KMEANSCL/KMEANS/OpenCLFile.cl\"", NULL, NULL);
	{ // show warnings/errors
	  	static char log[65536]; memset(log, 0, sizeof(log));
	  	cl_device_id device_id = 0;
	  	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
	  	clGetProgramBuildInfo(cl_prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	  	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if (err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }

	char * kernel_compute_dists = "compute_dists";
	char * kernel_find_membership = "find_membership";
	char * kernel_compute_centroids_first_step = "compute_centroids_first_step";
	char * kernel_compute_centroids_second_step = "compute_centroids_second_step";
	//get/create kernels
	cl_kernel_compute_dists = clCreateKernel(cl_prog, kernel_compute_dists, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel(kernel_compute_dists) 0 => %d\n", err); return -1; }
	cl_kernel_find_membership = clCreateKernel(cl_prog, kernel_find_membership, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel(kernel_find_membership) 0 => %d\n", err); return -1; }
	cl_kernel_compute_centroids_first_step = clCreateKernel(cl_prog, kernel_compute_centroids_first_step, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel(kernel_compute_centroids_first_step) 0 => %d\n", err); return -1; }
	cl_kernel_compute_centroids_second_step = clCreateKernel(cl_prog, kernel_compute_centroids_second_step, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel(kernel_compute_centroids_second_step) 0 => %d\n", err); return -1; }
	clReleaseProgram(cl_prog);

	return 1;
}
/*----< allocate OpenCL arrays and stuff >---------------------------------------------------------*/
int kmeans::allocate() {
	cl_int err = 0;
	std::cout << "WARNING: local mem required " << (m_num_clusters*m_num_features*sizeof(float) + m_num_clusters*sizeof(int)) <<
		", local mem available " << LOCAL_MEM_SIZE << std::endl;

	//create GPU-side buffers
	cl_features = clCreateBuffer(context, CL_MEM_READ_WRITE, m_num_points_expanded * m_num_features_expanded * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_features (size:%zd) => %d\n", m_num_points_expanded * m_num_features_expanded* sizeof(float), err); return -1; }
	
	cl_centroids = clCreateBuffer(context, CL_MEM_READ_WRITE, m_num_clusters_expanded * m_num_features_expanded * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_centroids (size:%zd) => %d\n", m_num_clusters_expanded * m_num_features_expanded * sizeof(float), err); return -1; }
	
	cl_dists = clCreateBuffer(context, CL_MEM_READ_WRITE, m_num_points_expanded * m_num_clusters_expanded * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_dists (size:%zd) => %d\n", m_num_points_expanded * m_num_clusters_expanded * sizeof(float), err); return -1; }
	
	cl_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, m_num_points_expanded * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_membership (size:%zd) => %d\n", m_num_points_expanded * sizeof(int), err); return -1; }

	cl_n_parallel_units = MAX_COMPUTE_UNITS * 8;
	cl_n_group_size = cl_n_parallel_units;
	cl_n_groups = int(std::ceil(float(m_num_points_expanded) / float(cl_n_group_size)));
	allocated_cl_n_groups = cl_n_groups;
	
	cl_centroids_group_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_n_groups * m_num_clusters * m_num_features * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_centroids_group (size:%zd) => %d\n", cl_n_groups * m_num_clusters * m_num_features * sizeof(float), err); return -1; }

	cl_centroids_group_n_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_n_groups * m_num_clusters * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_centroids_group_n (size:%zd) => %d\n", cl_n_groups * m_num_clusters * sizeof(int), err); return -1; }

	cl_centroids_group_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_n_groups * m_num_clusters * m_num_features * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_centroids_group (size:%zd) => %d\n", cl_n_groups * m_num_clusters * m_num_features * sizeof(float), err); return -1; }

	cl_centroids_group_n_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_n_groups * m_num_clusters * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer cl_centroids_group_n (size:%zd) => %d\n", cl_n_groups * m_num_clusters * sizeof(int), err); return -1; }
	
	return 1;
}
/*----< deallocate OpenCL stuff >------------------------------------------------------------------*/
void kmeans::deallocateMemory()
{
	clReleaseKernel(cl_kernel_compute_dists);
	clReleaseKernel(cl_kernel_find_membership);
	clReleaseKernel(cl_kernel_compute_centroids_first_step);
	clReleaseKernel(cl_kernel_compute_centroids_second_step);
	clReleaseMemObject(cl_features);
	clReleaseMemObject(cl_centroids);
	clReleaseMemObject(cl_dists);
	clReleaseMemObject(cl_membership);
	clReleaseMemObject(cl_centroids_group_1);
	clReleaseMemObject(cl_centroids_group_n_1);
	clReleaseMemObject(cl_centroids_group_2);
	clReleaseMemObject(cl_centroids_group_n_2);
	//clReleaseProgram(prog);
}
/*----< set kernel arguments >---------------------------------------------------------------------*/
int kmeans::set_kernel_args() {
	cl_n_parallel_units = MAX_COMPUTE_UNITS * 8;
	cl_n_group_size = cl_n_parallel_units;
	cl_n_groups = int(std::ceil(float(m_num_points_expanded) / float(cl_n_group_size)));
	allocated_cl_n_groups = cl_n_groups;

	cl_int err = 0;
	//first kernel
	//__kernel void compute_dists(const __global float* points, \n\
	//							const __global float* centroids, \n\
	//							__global float* C)
	err = clSetKernelArg(cl_kernel_compute_dists, 0, sizeof(cl_mem), (void*)&cl_features);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_dists, 1, sizeof(cl_mem), (void*)&cl_centroids);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_dists, 2, sizeof(cl_mem), (void*)&cl_dists);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	//second kernel
	//__kernel void find_membership(__global float* dists, __global int* membership)
	err = clSetKernelArg(cl_kernel_find_membership, 0, sizeof(cl_mem), (void*)&cl_dists);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_find_membership, 1, sizeof(cl_mem), (void*)&cl_membership);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	//__kernel void compute_centroids_first_step(__global float* points, \n\
	//								__global float* centroid_groups, __global int* centroid_groups_n, \n\
	//								__global int* membership, int group_size)
	err = clSetKernelArg(cl_kernel_compute_centroids_first_step, 0, sizeof(void *), (void*)&cl_features);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_first_step, 1, sizeof(void *), (void*)&cl_centroids_group_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_first_step, 2, sizeof(void *), (void*)&cl_centroids_group_n_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_first_step, 3, sizeof(void *), (void*)&cl_membership);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_first_step, 4, sizeof(cl_int), (void*)&cl_n_group_size);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	//__kernel void compute_centroids_second_step(__global float* in_centroid_groups, __global int* in_centroid_groups_n, \n\
	//											__global float* out_centroid_groups, __global int* out_centroid_groups_n, \n\
	//											int group_size, int num_groups)
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 0, sizeof(void *), (void*)&cl_centroids_group_1);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 1, sizeof(void *), (void*)&cl_centroids_group_n_1);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 2, sizeof(void *), (void*)&cl_centroids_group_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 3, sizeof(void *), (void*)&cl_centroids_group_n_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 4, sizeof(cl_int), (void*)&cl_n_group_size);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 5, sizeof(cl_int), (void*)&cl_n_groups);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	return 1;
}
/*----< update kernel arguments >------------------------------------------------------------------*/
int kmeans::update_kernel_args(const int new_group_size, const int new_num_groups) {
	cl_int err = 0;
	
	cl_mem t1 = cl_centroids_group_1;
	cl_centroids_group_1 = cl_centroids_group_2;
	cl_centroids_group_2 = t1;

	cl_mem t1_n = cl_centroids_group_n_1;
	cl_centroids_group_n_1 = cl_centroids_group_n_2;
	cl_centroids_group_n_2 = t1_n;

	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 0, sizeof(void *), (void*)&cl_centroids_group_1);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 1, sizeof(void *), (void*)&cl_centroids_group_n_1);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 2, sizeof(void *), (void*)&cl_centroids_group_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 3, sizeof(void *), (void*)&cl_centroids_group_n_2);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 4, sizeof(int), (void*)&new_group_size);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	err = clSetKernelArg(cl_kernel_compute_centroids_second_step, 5, sizeof(int), (void*)&new_num_groups);
	if (err != CL_SUCCESS) { printf("ERROR: clSetKernelArg()=>%d failed\n", err); return -1; }
	return 1;
}
/*----< set dataset vars >-------------------------------------------------------------------------*/
void kmeans::set_dataset(const std::vector<float>& dataset, const int npts,
	const int nfeatures, const int tile_size) {
	m_num_points = npts;
	m_num_points_expanded = round_up_multiple(npts, tile_size);
	m_num_features = nfeatures;
	m_num_features_expanded = round_up_multiple(nfeatures, tile_size);
	m_tile_size = tile_size;
	m_dataset.clear();
	m_dataset.assign(m_num_points_expanded*m_num_features_expanded, 0.0f);
	for (int i = 0; i < m_num_points; ++i) {
		memcpy(m_dataset.data() + i*m_num_features_expanded,
			dataset.data() + i*m_num_features, sizeof(float)*m_num_features);
	}
	
}
/*----< set clusters vars >-------------------------------------------------------------------------*/
void kmeans::set_clusters(std::vector<float>& clusters, const int nclusters) {
	m_num_clusters = nclusters;
	m_num_clusters_expanded = round_up_multiple(nclusters, m_tile_size);
	m_clusters.clear();
	m_clusters.assign(m_num_clusters_expanded*m_num_features_expanded, 0.0f);
	for (int i = 0; i < m_num_clusters; ++i) {
		memcpy(m_clusters.data() + i*m_num_features_expanded,
			clusters.data() + i*m_num_features, sizeof(float)*m_num_features);
	}
}
/*----< upload feature set to gpu >----------------------------------------------------------------*/
int kmeans::upload_features() {
	cl_int err = 0;
	size_t write_size = 0;
	write_size = m_num_points_expanded * m_num_features_expanded * sizeof(float);
	err = clEnqueueWriteBuffer(cmd_queue, cl_features, 1, 0, write_size, (void*)(m_dataset.data()), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer cl_features (size:%zd) => %d\n", write_size, err); return -1; }
	return 1;
}
/*----< upload centroids set to gpu >--------------------------------------------------------------*/
int kmeans::upload_centroids() {
	cl_int err = 0;
	size_t write_size = 0;
	write_size = m_num_clusters_expanded * m_num_features_expanded * sizeof(float);
	err = clEnqueueWriteBuffer(cmd_queue, cl_centroids, 1, 0, write_size, (void*)(m_clusters.data()), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer cl_centroids (size:%zd) => %d\n", write_size, err); return -1; }
	return 1;
}
/*----< compute point membership on gpu >----------------------------------------------------------*/
int kmeans::compute_point_membership() {
	cl_int err = 0;
	size_t global_work_1[2] = { size_t(m_num_points_expanded), size_t(m_num_clusters_expanded) };
	size_t localWorkSize_1[2] = { m_tile_size, m_tile_size };
	err = clEnqueueNDRangeKernel(cmd_queue, cl_kernel_compute_dists, 2, 0, global_work_1, localWorkSize_1, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: second clEnqueueNDRangeKernel(cl_kernel_compute_dists)=>%d failed\n", err); return -1; }
	clFinish(cmd_queue);
	size_t global_work_2 = size_t(m_num_points);
	err = clEnqueueNDRangeKernel(cmd_queue, cl_kernel_find_membership, 1, 0, &global_work_2, 0, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: second clEnqueueNDRangeKernel(cl_kernel_find_membership)=>%d failed\n", err); return -1; }

	return 1;
}
/*----< read point membership data from gpu >------------------------------------------------------*/
int kmeans::read_membership(std::vector<int>& tmp_membership) {
	tmp_membership.clear();
	tmp_membership.assign(m_num_points, -1);
	cl_int err = 0;
	size_t read_size = m_num_points * sizeof(int);
	err = clEnqueueReadBuffer(cmd_queue, cl_membership, 0, 0, read_size, (void*)tmp_membership.data(), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out cl_centroids\n"); return -1; }
	return 1;
}
/*----< compute new centroids on the gpu >---------------------------------------------------------*/
int kmeans::compute_new_centroids(int num_division) {
	cl_int err = 0;

	size_t global_work_1 = size_t(cl_n_groups);
	err = clEnqueueNDRangeKernel(cmd_queue, cl_kernel_compute_centroids_first_step, 1, 0, &global_work_1, 0, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: second clEnqueueNDRangeKernel(cl_kernel_compute_centroids_first_step)=>%d failed\n", err); return -1; }
	while (cl_n_groups > cl_n_parallel_units) {
		int old_group_size = cl_n_groups;
		cl_n_groups = int(std::ceil(float(cl_n_groups) / float(num_division)));
		cl_n_group_size = num_division;
		update_kernel_args(cl_n_group_size, old_group_size);

		size_t global_work_2 = size_t(cl_n_groups);
		err = clEnqueueNDRangeKernel(cmd_queue, cl_kernel_compute_centroids_second_step, 1, 0, &global_work_2, 0, 0, 0, 0);
		if (err != CL_SUCCESS) { printf("ERROR: second clEnqueueNDRangeKernel(cl_kernel_compute_centroids_second_step)=>%d failed\n", err); return -1; }
	}

	size_t read_size = allocated_cl_n_groups * m_num_clusters * m_num_features * sizeof(float);
	std::vector<float> temp_centroids_groups;
	temp_centroids_groups.resize(read_size/ sizeof(float));
	err = clEnqueueReadBuffer(cmd_queue, cl_centroids_group_2, 1, 0, read_size, (void*)(temp_centroids_groups.data()), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out cl_membership\n"); return -1; }
	
	read_size = allocated_cl_n_groups * m_num_clusters * sizeof(int);
	std::vector<int> temp_centroids_groups_n;
	temp_centroids_groups_n.resize(read_size / sizeof(int));
	err = clEnqueueReadBuffer(cmd_queue, cl_centroids_group_n_2, 1, 0, read_size, (void*)(temp_centroids_groups_n.data()), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out cl_membership\n"); return -1; }

	for (int i = 1; i < allocated_cl_n_groups; ++i) {
		for (int j = 0; j < m_num_clusters; ++j) {
			if (temp_centroids_groups_n[(i*m_num_clusters) + j] != 0) {
				for (int k = 0; k < m_num_features; ++k) {
					temp_centroids_groups[j*m_num_features + k] += temp_centroids_groups[(i*m_num_clusters + j) * m_num_features + k];
				}
				temp_centroids_groups_n[j] += temp_centroids_groups_n[(i*m_num_clusters) + j];
				temp_centroids_groups_n[(i*m_num_clusters) + j] = 0;
			}
		}
	}
	memset(m_clusters.data(), 0, sizeof(float)*m_num_features_expanded*m_num_clusters_expanded);
	for (int j = 0; j < m_num_clusters; ++j) {
		if (temp_centroids_groups_n[j] != 0) {
			for (int k = 0; k < m_num_features; ++k) {
				m_clusters[j*m_num_features_expanded + k] = temp_centroids_groups[j*m_num_features + k] / temp_centroids_groups_n[j];
			}
		}
	}
	return 1;
}
/*----< do clustering on the gpu >-----------------------------------------------------------------*/
int kmeans::perform_clustering(int threshold, int nMaxInternalLoops, int num_divisions, std::vector<int>& membership) {
	set_kernel_args();

	upload_features();

	std::vector<int> tmp_membership;
	tmp_membership.assign(m_num_points_expanded, -1);
	membership.clear();
	membership.assign(m_num_points_expanded, -1);
	int loop = 0;
	int delta = 999;
	do {
		set_kernel_args();
		upload_centroids();
		compute_point_membership();
		read_membership(tmp_membership);
		compute_new_centroids(num_divisions);
		delta = 0;
		for (int i = 0; i < m_num_points; ++i) { //get num sample for each cluster
			int cluster_id = tmp_membership[i];
			if (cluster_id != membership[i]) {
				++delta;
			}
		}
		membership.swap(tmp_membership);
		std::cout << int(float(loop) / float(nMaxInternalLoops) * 100.0f) << "% ";
	} while ((delta > threshold) && (loop++ < nMaxInternalLoops));	/* makes sure loop terminates */
	std::cout << " iterated " << loop << " times ";
	return 1;
}

void kmeans::cpu_kmeans_clustering(const std::vector<float>& feature_set, const int nfeatures, const int npoints, const int nclusters,
	const int threshold, std::vector<int>& membership, std::vector<float>& initial_clusters, std::vector<float>& output_clusters, const int nMaxInternalLoops) {

	std::vector<float>  clusters;						/* out: [nclusters][nfeatures] */

	/* nclusters should never be > npoints that would guarantee a cluster without points */
	if (nclusters > npoints) {
		return;
	}
	/* allocate space for and initialize returning variable clusters[] */
	clusters.assign(nclusters * nfeatures, 0.0f);
	/* initialize the clusters */
	if (initial_clusters.size() == 0) {
		//kmeans++
		kpp(feature_set.data(), nfeatures, npoints, clusters.data(), nclusters);
		initial_clusters = clusters;
	}
	else {
		//else reuse previous cycle clusters
		memcpy(clusters.data(), initial_clusters.data(), sizeof(float)*nfeatures*nclusters);
	}
	/* initialize the membership to -1 for all */
	membership.clear();
	membership.assign(npoints, -1);
	/* iterate until convergence */
	int loop = 0;
	int delta = 999;
	do {
		std::vector<int> temp_membership;
		temp_membership.assign(npoints, -1);
		group_by_cluster(feature_set.data(), clusters.data(), temp_membership.data(), npoints, nclusters, nfeatures);
		update_centroids(feature_set.data(), npoints, temp_membership.data(), clusters.data(), nclusters, nfeatures);
		delta = 0;
		for (int i = 0; i < npoints; ++i) { //get num sample for each cluster
			int cluster_id = temp_membership[i];
			if (cluster_id != membership[i]) {
				++delta;
			}
			membership[i] = cluster_id;
		}
		std::cout << int(float(loop) / float(nMaxInternalLoops) * 100.0f) << "% ";
	} while ((delta > threshold) && (loop++ < nMaxInternalLoops));	/* makes sure loop terminates */
	std::cout << " iterated " << loop << " times ";
	output_clusters.swap(clusters);
}

int kmeans::cluster(const int use_gpu, 
					const int npoints, const int nfeatures, const std::vector<float>& feature_set,
					const int nclusters, std::vector<float>& cluster_centres, 
					const int threshold, const bool compute_rmse, float& out_rmse,
					const int nMaxInternalLoops, const bool outputMembership, std::vector<int>& bestMembership,
					const int tile_size, const int div_size, const bool use_kpp )
{
	float	min_rmse_ref = FLT_MAX;
	
	std::vector<int> membership;			
	membership.assign(npoints, 0);
	
	/* initialize open_cl stuff */
	if (initialize(use_gpu)<0) return -1;
	
	/* set dataset, expand it and init internals*/
	set_dataset(feature_set, npoints, nfeatures, tile_size);

	std::cout << "Computing " << nclusters << " clusters over " << npoints << " points" << std::endl;
	if (nclusters > npoints) {
		std::cout << "ERROR: cannot have more clusters than points!" << std::endl;
		return -1;	/* cannot have more clusters than points */
	}

	/* compute initials cluster centers if thei were not passed */
	std::vector<float> initial_cluster_centres;
	if (cluster_centres.size() == nclusters*nfeatures) {
		initial_cluster_centres = cluster_centres;
	}
	else {
		initial_cluster_centres.resize(nclusters*nfeatures);
		if (use_kpp) {
			kpp(feature_set.data(), nfeatures, npoints, initial_cluster_centres.data(), nclusters);
		}
		else {
			random(feature_set.data(), nfeatures, npoints, initial_cluster_centres.data(), nclusters);
		}
	}

	/* set clusters, expand them and init internals*/
	set_clusters(initial_cluster_centres, nclusters);

	/* customize kernels with dataset parameters and compile them */
	if (compile_kernels()<0) return -1;

	/* allocate device memory*/
	if (allocate()<0) return -1;

	/* do clustering */
	auto begin_time = std::chrono::system_clock::now();
	perform_clustering(threshold, nMaxInternalLoops, div_size, membership);
	auto end_time = std::chrono::system_clock::now();

	/* compute RMSE */
	if(compute_rmse) {
		out_rmse = rms_err(feature_set.data(), nfeatures, m_num_features_expanded, npoints, m_clusters.data(), nclusters);
		std::cout << " rms error=" << out_rmse << std::endl;
	}

	/* free device memory */
	deallocateMemory();	

	/* shutdown opencl */
	shutdown();

	/* copy outputs */
	cluster_centres.resize(m_num_clusters*m_num_features);
	for (int i = 0; i < m_num_clusters; ++i) {
		memcpy(cluster_centres.data() + i*m_num_features, m_clusters.data() + i*m_num_features_expanded, m_num_features*sizeof(float));
	}
	if (outputMembership)
		membership.swap(bestMembership);

	uint64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
	return int(elapsed_ms);
}

int kmeans::cpu_cluster(const int npoints, const int nfeatures, const std::vector<float>& feature_set,
						const int nclusters, std::vector<float>& cluster_centres,
						const int threshold,const  bool compute_rmse, float& out_rmse,
						const int nMaxInternalLoops, const bool outputMembership, std::vector<int>& bestMembership,
						const bool use_kpp)
{
	std::vector<int> membership;			
	membership.assign(npoints, 0);
	
	std::cout << "Computing " << nclusters << " clusters over " << npoints << " points" << std::endl;
	if (nclusters > npoints) {
		std::cout << "ERROR: cannot have more clusters than points!" << std::endl;
		return -1;	/* cannot have more clusters than points */
	}

	/* compute initials cluster centers if thei were not passed */
	std::vector<float> initial_cluster_centres;
	if (cluster_centres.size() == nclusters*nfeatures) {
		initial_cluster_centres = cluster_centres;
	}
	else {
		initial_cluster_centres.resize(nclusters*nfeatures);
		if (use_kpp) {
			kpp(feature_set.data(), nfeatures, npoints, initial_cluster_centres.data(), nclusters);
		}
		else {
			random(feature_set.data(), nfeatures, npoints, initial_cluster_centres.data(), nclusters);
		}
	}

	/* do clustering */
	auto begin_time = std::chrono::system_clock::now();
	cpu_kmeans_clustering(feature_set, nfeatures, npoints, nclusters, threshold, membership, initial_cluster_centres, cluster_centres, nMaxInternalLoops);		
	auto end_time = std::chrono::system_clock::now();

	/* compute RMSE */
	if (compute_rmse) {
		out_rmse = rms_err(feature_set.data(), nfeatures, npoints, cluster_centres.data(), nclusters);
		std::cout << " rms error=" << out_rmse << std::endl;
	}

	if (outputMembership)
		membership.swap(bestMembership);

	uint64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
	return int(elapsed_ms);
}