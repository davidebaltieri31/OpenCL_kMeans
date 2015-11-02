//stuff
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <float.h>
#include <vector>
#include <algorithm>
//parallel_for stuff
#include <ppl.h>
#include <atomic>
#include <mutex>
//opencl stuff
#include <windows.h>
#include <CL/cl.h>

#include "kmeans.h"

const char* kernel_source = "\
#ifndef FLT_MAX\n\
#define FLT_MAX 3.40282347e+38\n\
#endif\n\
__kernel void\n\
kmeans_kernel_c(__global float  *feature,\n\
				__global float  *clusters,\n\
				__global int    *membership,\n\
				int     npoints,\n\
				int     nclusters,\n\
				int     nfeatures,\n\
				int		offset,\n\
				int		size\n\
)\n\
{\n\
	unsigned int point_id = get_global_id(0);\n\
	int index = 0;\n\
	if (point_id < npoints)\n\
	{\n\
		float min_dist = FLT_MAX;\n\
		for (int i = 0; i < nclusters; i++) {\n\
			float dist = 0;\n\
			float ans = 0;\n\
			for (int l = 0; l<nfeatures; l++){\n\
				ans = feature[l * npoints + point_id] - clusters[i*nfeatures + l];\n\
				dist += (ans)*(ans);\n\
			}\n\
			if (dist < min_dist) {\n\
				min_dist = dist;\n\
				index = i;\n\
			}\n\
		}\n\
		membership[point_id] = index;\n\
	}\n\
	return;\n\
}\n\
__kernel void\n\
kmeans_swap(__global float  *feature,\n\
			__global float  *feature_swap,\n\
			int     npoints,\n\
			int     nfeatures)\n\
{\
	unsigned int tid = get_global_id(0);\n\
	for (int i = 0; i < nfeatures; i++)\n\
		feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];\n\
}\n\
";


/*---------------FORWARD DECLARATIONS----------------------------------------------------------*/
/*----< fast float resqrt >------------------------------------------------------------------------*/
float fast_resqrt(const float x);
/*----< fast float sqrt >--------------------------------------------------------------------------*/
float fast_sqrt(const float x);
/*----< multi-dimensional spatial Euclid distance square >----------------------------------------------------*/
float euclid_dist_2(const float *pt1, /* [numdims] float array */
					const float *pt2, /* [numdims] float array */
					const int numdims);
/*----< find_nearest_point() >-----------------------------------------------*/
int find_nearest_point(	const float  *pt,          /* origin, [nfeatures] float array */
						const int     nfeatures,
						const float  *pts,         /* points set, [npts][nfeatures] float array */
						const int     npts);
/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/
float rms_err(	const std::vector<float>& feature_set,         /* points set, [npoints][nfeatures] float vector */
				const int     nfeatures,
				const int     npoints,
				const std::vector<float>& cluster_centres,	/* cluster centers, [nclusters][nfeatures] float vector */
				const int     nclusters);
/*----< kmeans_clustering() a single block of kmeans iterations>---------------------------------------------*/
void kmeans_clustering(	const std::vector<float>& feature_set,    /* in: point set, [npoints][nfeatures] float vector*/
						const int nfeatures,
						const int npoints,
						const int     nclusters,
						const float   threshold,
						std::vector<int>& membership, /* out: [npoints], cluster membership of each point */
						const std::vector<float>& initial_clusters, /* in: [ncluster][nfeatures], initial cluster centers (empty vector if not known/not set) */
						std::vector<float>& output_clusters,  /* out: [ncluster][nfeatures], the NEW cluster centers */
						const int nMaxInternalLoops);
/*----< allocate OpenCL arrays and stuff >---------------------------------------------*/
int allocate(	const int n_points, 
				const int n_features, 
				const int n_clusters, 
				const std::vector<float>& feature); /* points set, [n_points][n_features] float vector */
/*----< deallocate OpenCL stuff >-------------------------------------------------------------*/
void deallocateMemory();
/*----< kmeansOCL: a single kmeans iteration via OpenCL >-------------------------------------------*/
int	kmeansOCL(	const std::vector<float>& feature_set, /* points set, [n_points][n_features] float array */
				const int n_features,
				const int n_points,
				const int n_clusters,
				std::vector<int>& membership,  /* out: [npoints], cluster membership of each point */
				const std::vector<float>& clusters,  /* in: [ncluster][nfeatures], the OLD cluster centers */
				std::vector<int>& new_centers_len, /* out: [ncluster] number of assigned points for each cluster*/
				std::vector<float>& new_centers); /* out: [ncluster][nfeatures], the NEW cluster centers */

/*--------------FUNCTIONS-------------------------------------------------------------------------*/

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
/*----< multi-dimensional spatial Euclid distance square >----------------------------------------------------*/
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
/*----< find_nearest_point() >-----------------------------------------------*/
__inline int find_nearest_point(const float  *pt, const int nfeatures,const  float *pts, const int npts)
{
	int index, i;
	float max_dist = FLT_MAX;
	/* find the cluster center id with min distance to pt */
	for (i = 0; i<npts; i++) {
		float dist = euclid_dist_2(pt, pts + nfeatures*i, nfeatures);  /* no need square root */
		index = (dist < max_dist) ? i : index;
		max_dist = (dist < max_dist) ? dist : max_dist;
	}
	return(index);

}
/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/
float rms_err(const std::vector<float>& feature_set, const int nfeatures, const int npoints, const std::vector<float>& cluster_centres, const int nclusters)
{
	float  sum_euclid = 0.0;		/* sum of Euclidean distance squares */
	std::mutex err_sum_mutex;
	/* calculate and sum the sqaure of euclidean distance*/
	//for (i = 0; i<npoints; i++) {
	//	nearest_cluster_index = find_nearest_point(feature_set[i], cluster_centres);
	//	sum_euclid += euclid_dist_2(feature_set[i], cluster_centres[nearest_cluster_index]);
	//}
	Concurrency::parallel_for(0, npoints, [&](size_t i)
	{
		const float* pt_data = feature_set.data() + nfeatures * i;
		const float* cluster_data = cluster_centres.data();
		int nearest_cluster_index = find_nearest_point(pt_data, nfeatures, cluster_data, nclusters);
		float dist = euclid_dist_2(pt_data, cluster_data + nfeatures*nearest_cluster_index, nfeatures);
		err_sum_mutex.lock();
		sum_euclid += dist;
		err_sum_mutex.unlock();
	});
	float ret = fast_sqrt(sum_euclid / npoints);
	return(ret);
}
/*----< kmeans_clustering() >---------------------------------------------*/
void kmeans_clustering(const std::vector<float>& feature_set, const int nfeatures, const int npoints, const int nclusters, const float threshold, std::vector<int>& membership, const std::vector<float>& initial_clusters, std::vector<float>& output_clusters, const int nMaxInternalLoops)
{
	std::vector<int> new_centers_len;					/* [nclusters]: no. of points in each cluster */
	float    delta;										/* if the point moved */
	std::vector<float>  clusters;						/* out: [nclusters][nfeatures] */
	std::vector<float>  new_centers;					/* [nclusters][nfeatures] */
	std::vector<int> initial;							/* used to hold the index of points not yet selected, prevents the "birthday problem" of dual selection (?) */
	int      initial_points;
	/* nclusters should never be > npoints that would guarantee a cluster without points */
	if (nclusters > npoints) {
		return;
	}
	/* allocate space for and initialize returning variable clusters[] */
	clusters.assign(nclusters * nfeatures, 0.0f);
	/* initialize the random clusters */
	initial.resize(npoints);
	for (int i = 0; i < npoints; ++i) {
		initial[i] = i;
	}
	initial_points = npoints;
	if (initial_clusters.size() == 0) {
		/* randomly pick cluster centers */
		for (int i = 0; i < nclusters && initial_points >= 0; ++i) {
			int n = (int)rand() % initial_points;
			for (int j = 0; j < nfeatures; ++j) {
				clusters[i*nfeatures + j] = feature_set[initial[n] * nfeatures +j];	// remapped
			}
			/* swap the selected index to the end (not really necessary, could just move the end up) */
			int temp = initial[n];
			initial[n] = initial[initial_points - 1];
			initial[initial_points - 1] = temp;
			initial_points--;
		}
	}
	else {
		for (int i = 0; i < nclusters; ++i) {
			for (int j = 0; j < nfeatures; ++j) {
				clusters[i*nfeatures + j] = initial_clusters[i*nfeatures + j];	// reuse input centers
			}
		}
	}
	/* initialize the membership to -1 for all */
	for (int i = 0; i < npoints; ++i) {
		membership[i] = -1;
	}
	/* allocate space for and initialize new_centers_len and new_centers */
	new_centers_len.assign(nclusters, 0);
	new_centers.assign(nclusters * nfeatures, 0.0f);
	/* iterate until convergence */
	int loop = 0;
	do {
		delta = 0.0;
		// CUDA
		delta = (float)kmeansOCL(	feature_set,	/* in: [npoints][nfeatures] */
									nfeatures,		/* number of attributes for each point */
									npoints,		/* number of data points */
									nclusters,		/* number of clusters */
									membership,		/* which cluster the point belongs to */
									clusters,		/* out: [nclusters][nfeatures] */
									new_centers_len,/* out: number of points in each cluster */
									new_centers		/* sum of points in each cluster */
									);
		/* replace old cluster centers with new_centers */
		for (int i = 0; i<nclusters; ++i) {
			if (new_centers_len[i] > 0) {
				for (int j = 0; j<nfeatures; ++j) { //TODO use memcpy
					clusters[i*nfeatures + j] = new_centers[i*nfeatures + j];	/* take average i.e. sum / MOVED INSIDE kmeansOCL*/
				}
			}
			else { /* cluster has 0 assigned points, reinitialize with random point */
				float sum = 0.0f;
				int n = (int)rand() % initial_points;
				for (int j = 0; j < nfeatures; ++j) {
					clusters[i*nfeatures + j] = feature_set[initial[n] * nfeatures + j];	
				}
				int temp = initial[n];
				initial[n] = initial[initial_points - 1];
				initial[initial_points - 1] = temp;
				initial_points--;
			}
		}
		std::cout << int(float(loop) / float(nMaxInternalLoops) * 100.0f) << "% ";
	} while ((delta > threshold) && (loop++ < nMaxInternalLoops));	/* makes sure loop terminates */
	//printf("iterated %d times\n", c);
	std::cout << " iterated " << loop << " times ";
	output_clusters.swap(clusters);
}
/*----< OPENCL STUFF >------------------------------------------------*/
// local variables, opencl states
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;
/*----< initialize opencl >------------------------------------------*/
static int initialize(const int use_gpu)
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }

	return 0;
}
/*----< shutdown opencl >--------------------------------------------*/
static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}
//opencl kernels and opencl data ids
cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem d_cluster;
cl_mem d_membership;
cl_kernel kernel_s;
cl_kernel kernel2;
int   *membership_OCL; //temp data
/*----< allocate OpenCL arrays and stuff >---------------------------------------------*/
int allocate(const int n_points, const int n_features, const int n_clusters, const std::vector<float>& feature)
{
	int sourcesize = strlen(kernel_source);
	char* source = (char*)kernel_source;

	// OpenCL initialization
	int use_gpu = 1;
	if (initialize(use_gpu)) return -1;

	// compile kernel
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	{ // show warnings/errors
		//	static char log[65536]; memset(log, 0, sizeof(log));
		//	cl_device_id device_id = 0;
		//	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		//	clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		//	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if (err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }

	char * kernel_kmeans_c = "kmeans_kernel_c";
	char * kernel_swap = "kmeans_swap";

	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	kernel2 = clCreateKernel(prog, kernel_swap, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }

	clReleaseProgram(prog);

	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1; }
	d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", n_points * n_features, err); return -1; }
	d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", n_clusters * n_features, err); return -1; }
	d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", n_points, err); return -1; }

	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), (char*)feature.data(), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1; }

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*)&d_feature);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*)&d_feature_swap);
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&n_points);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*)&n_features);

	size_t global_work[3] = { n_points, 1, 1 };
	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, NULL, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	membership_OCL = (int*)malloc(n_points * sizeof(int));
	return 1;
}
/*----< deallocate OpenCL stuff >-------------------------------------------------------------*/
void deallocateMemory()
{
	clReleaseMemObject(d_feature);
	clReleaseMemObject(d_feature_swap);
	clReleaseMemObject(d_cluster);
	clReleaseMemObject(d_membership);
	free(membership_OCL);

}
/*----< kmeansOCL: a single kmeans iteration via OpenCL >-------------------------------------------*/
int	kmeansOCL(const std::vector<float>& feature_set, const int n_features, const int n_points, const int n_clusters, std::vector<int>& membership, const std::vector<float>& clusters, std::vector<int>& new_centers_len, std::vector<float>& new_centers)
{
	int delta = 0;
	cl_int err = 0;
	size_t global_work[3] = { n_points, 1, 1 };
	//update opencl/gpu memory kernel clusters param
	err = clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters.data(), 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err); return -1; }
	int size = 0; int offset = 0;
	//set kernel params
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*)&d_feature_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*)&d_cluster);
	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*)&d_membership);
	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*)&n_points);
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*)&n_clusters);
	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*)&n_features);
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*)&offset);
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*)&size);
	//execute kernel
	err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	clFinish(cmd_queue); //wait kernel completion
	//get output data (feature membership)
	err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
	if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
	//compute new cluster centers
	delta = 0;
	new_centers_len.assign(n_clusters, 0);
	for (int i = 0; i < n_points; ++i) { //get num sample for each cluster
		int cluster_id = membership_OCL[i];
		++new_centers_len[cluster_id];
		//if (cluster_id<0 || cluster_id>n_points) std::cout << " ERRORE!! ";
		if (cluster_id != membership[i]) {
			++delta;
		}
		membership[i] = cluster_id;
	}
	new_centers.assign(n_clusters*n_features, 0.0f);
	//compute actual centers
	for (int i = 0; i < n_points; ++i) { // TODO can be parallelized?
		int cluster_id = membership[i];
		if (new_centers_len[cluster_id] > 0) {
			for (int j = 0; j < n_features; ++j) {
				//division done here, less precise but more numerically stable
				new_centers[cluster_id*n_features + j] += (feature_set[i*n_features + j] / float(new_centers_len[cluster_id]));
			}
		}
	}
	return delta;
}

/*---< THE FUNCTION >--------------------------------------------------------*/
/*---< cluster() >-----------------------------------------------------------*/
int cluster(const int npoints, const int nfeatures, const std::vector<float>& feature_set, const int min_nclusters, const int max_nclusters, const float threshold,
	int& best_nclusters, std::vector<float>& cluster_centres, float& min_rmse, const bool isRMSE, const int nExternalLoops, const bool reuseOldCLusterCenters, const int nMaxInternalLoops)
{
	float	min_rmse_ref = FLT_MAX;
	int		nclusters;						/* number of clusters k */
	int		index = 0;						/* number of iteration to reach the best RMSE */
	float	rmse;							/* RMSE for each clustering */
	std::vector<int> membership;			/* which cluster a data point belongs to */
	membership.assign(npoints,0);
	std::vector<int> chosen_membership;
	chosen_membership.assign(npoints, 0);
	std::vector<float> tmp_cluster_centres;	/* hold coordinates of cluster centers */
	cluster_centres.resize(0);
	int		i;
	/* sweep k from min to max_nclusters to find the best number of clusters */
	for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
		std::cout << "Computing " << nclusters << " clusters" << std::endl;
		if (nclusters > npoints) break;	/* cannot have more clusters than points */
		/* allocate device memory, invert data array */
		allocate(npoints, nfeatures, nclusters, feature_set);
		/* iterate nloops times for each number of clusters */
		for (i = 0; i < nExternalLoops; i++) {
			std::cout << "Iteration " << i << " of " << nExternalLoops << " : ";
			/* initialize initial cluster centers*/
			kmeans_clustering(feature_set, nfeatures, npoints, nclusters, threshold, membership, cluster_centres, tmp_cluster_centres, nMaxInternalLoops);
			/* find the number of clusters with the best RMSE */
			if (isRMSE) {
				rmse = rms_err(feature_set, nfeatures, npoints, tmp_cluster_centres, nclusters);
				std::cout << " rms error=" << rmse << std::endl;
				if (rmse < min_rmse_ref) {
					min_rmse_ref = rmse;			//update reference min RMSE
					min_rmse = min_rmse_ref;		//update return min RMSE
					best_nclusters = nclusters;	//update optimum number of clusters
					index = i;						//update number of iteration to reach best RMSE
					cluster_centres.swap(tmp_cluster_centres);
					chosen_membership.swap(membership);
					/*search if some clusters have 0 membersm
					split bigger cluster and ovewrite the zero one
					do only if not last external loop and if activated*/
					//TODO parallelize?
					if (((i+1)<nExternalLoops) && reuseOldCLusterCenters) {
						std::vector<std::pair<int, int>> temp_membership;
						temp_membership.resize(best_nclusters);
						for (int i = 0; i < temp_membership.size(); ++i) {
							temp_membership[i].first = i;
							temp_membership[i].second = 0;
						}
						for (int i = 0; i < npoints; ++i) {
							temp_membership[chosen_membership[i]].second++;
						}
						std::sort(temp_membership.begin(), temp_membership.end(), [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
							return left.second > right.second;
						});
						bool has_zero = false;
						if (temp_membership[temp_membership.size() - 1].second == 0) has_zero = true;
						else has_zero = false;
						while (has_zero) {
							//search best split axis
							int best_axis = -1;
							float best_ratio = 1.0f;
							int best_left = 0;
							int best_right = 0;
							for (int j = 0; j < nfeatures; ++j) {
								int toleft = 0;
								int total = 0;
								float ratio = 0.0f;
								for (int i = 0; i < npoints; ++i) {
									if (chosen_membership[i] == temp_membership[0].first) { //is a points of the cluster i have to split
										if (feature_set[i*nfeatures + j] < cluster_centres[temp_membership[0].first*nfeatures + j]) {
											toleft++;
										}
										total++;
									}
								}
								ratio = abs((float(toleft) / float(total)) - 0.5f);
								if (ratio < best_ratio) {
									best_axis = j;
									best_left = toleft;
									best_right = total - toleft;
								}
							}
							//compute splitted clusters centers
							std::vector<float> cluster_vec_center_A; int num_cluster_vec_center_A = 0;
							cluster_vec_center_A.assign(nfeatures, 0.0f);
							std::vector<float> cluster_vec_center_B; int num_cluster_vec_center_B = 0;
							cluster_vec_center_B.assign(nfeatures, 0.0f);
							for (int i = 0; i < npoints; ++i) {
								if (chosen_membership[i] == temp_membership[0].first) {
									if (feature_set[i*nfeatures + best_axis] < cluster_centres[temp_membership[0].first*nfeatures + best_axis]) {
										for (int j = 0; j < nfeatures; ++j) {
											cluster_vec_center_A[j] += feature_set[i*nfeatures + j];
											num_cluster_vec_center_A++;
										}
										chosen_membership[i] = temp_membership[0].first;
									}
									else {
										for (int j = 0; j < nfeatures; ++j) {
											cluster_vec_center_B[j] += feature_set[i*nfeatures + j];
											num_cluster_vec_center_B++;
										}
										chosen_membership[i] = temp_membership[temp_membership.size() - 1].first;
									}
								}
							}
							for (int j = 0; j < nfeatures; ++j) {
								cluster_vec_center_A[j] = cluster_vec_center_A[j] / float(num_cluster_vec_center_A);
								cluster_vec_center_B[j] = cluster_vec_center_B[j] / float(num_cluster_vec_center_B);
							}
							//reassign new cluster centers
							for (int j = 0; j < nfeatures; ++j) {
								cluster_centres[temp_membership[0].first*nfeatures + j] = cluster_vec_center_A[j];
								cluster_centres[temp_membership[temp_membership.size() - 1].first*nfeatures + j] = cluster_vec_center_B[j];
							}
							temp_membership[0].second = best_left;
							temp_membership[temp_membership.size() - 1].second = best_right;

							std::sort(temp_membership.begin(), temp_membership.end(), [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
								return left.second > right.second;
							});
							if (temp_membership[temp_membership.size() - 1].second == 0) has_zero = true;
							else has_zero = false;
						};
					}
				}
			}
			else {
				cluster_centres.swap(tmp_cluster_centres);
				std::cout << std::endl;
			}
		}
		deallocateMemory();							/* free device memory */
	}
	std::vector<int> temp_membership;
	temp_membership.assign(best_nclusters, 0);
	for (int i = 0; i < npoints; ++i) {
		temp_membership[chosen_membership[i]]++;
	}
	std::cout << "membership" << std::endl;
	for (int i = 0; i < temp_membership.size(); ++i) {
		std::cout << temp_membership[i] << ", ";
	}
	std::cout << std::endl;
	shutdown();
	return index;
}