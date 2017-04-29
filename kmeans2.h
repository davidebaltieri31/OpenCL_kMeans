/*
*	OpenCL/SSE/Multithreaded optimized k-means
*	Copyright 2017 Davide Baltieri
*	Author Davide Baltieri
*	License LGPLv3
*	It would be awesome if you'd let me know if you use this code...
*/
#pragma once

#include <random>
#include <windows.h>
#include <CL/cl.h>

class kmeans {
public:
	void kpp(const  float *pts, const int nfeatures, const int npts, float  *centers, int ncent);
	void random(const  float *pts, const int nfeatures, const int npts, float  *centers, int ncent);
	int initialize(const int use_gpu);
	int shutdown();
	int compile_kernels();
	int allocate();
	void deallocateMemory();
	int set_kernel_args();
	int update_kernel_args(const int new_group_size, const int new_num_groups);
	void set_dataset(const std::vector<float>& dataset, const int npts, const int nfeatures, const int tile_size);
	void set_clusters(std::vector<float>& clusters, const int nclusters);
	int upload_features();
	int upload_centroids();
	int compute_point_membership();
	int read_membership(std::vector<int>& tmp_membership);
	int compute_new_centroids(int num_division);
	int perform_clustering(int threshold, int nMaxInternalLoops, int num_divisions, std::vector<int>& membership);

	int cluster(const int use_gpu,
		const int npoints, const int nfeatures, const std::vector<float>& feature_set,
		const int nclusters, std::vector<float>& cluster_centres,
		const int threshold, const bool compute_rmse, float& out_rmse,
		const int nMaxInternalLoops, const bool outputMembership, std::vector<int>& bestMembership,
		const int tile_size, const int div_size, const bool use_kpp);

	/* CPU clustering */
	void cpu_kmeans_clustering(const std::vector<float>& feature_set, const int nfeatures, const int npoints, const int nclusters,
		const int threshold, std::vector<int>& membership, std::vector<float>& initial_clusters, 
		std::vector<float>& output_clusters, const int nMaxInternalLoops);

	int cpu_cluster(const int npoints, const int nfeatures, const std::vector<float>& feature_set,
		const int nclusters, std::vector<float>& cluster_centres,
		const int threshold, const  bool compute_rmse, float& out_rmse,
		const int nMaxInternalLoops, const bool outputMembership, std::vector<int>& bestMembership,
		const bool use_kpp);

protected:
	std::random_device	m_rd;
	std::mt19937_64		m_generator;
	std::uniform_real_distribution<float>	m_random_float;
	std::uniform_int_distribution<int>		m_random_int;
	
	std::vector<float> m_dataset;
	int m_num_points;
	int m_num_points_expanded;
	std::vector<float> m_clusters;
	int m_num_clusters;
	int m_num_clusters_expanded;
	int m_num_features;
	int m_num_features_expanded;
	int m_tile_size = 16;

	// local variables, opencl states
	char*				m_kernel_source;
	cl_context			context;
	cl_command_queue	cmd_queue;
	cl_device_type		device_type;
	cl_device_id   *	device_list;
	cl_int				num_devices;
	cl_ulong			max_buffer_size = 0;
	size_t				max_work_group_size = 0;
	int					num_points_per_cycle = 0;
	int					num_cycles = 0;
	int					workItems = 0;
	bool				cl_initialized = false;
	int					MAX_WORK_ITEM_DIMENSIONS;
	std::vector<size_t> MAX_WORK_ITEM_SIZES;
	cl_ulong			LOCAL_MEM_SIZE;
	cl_uint				MAX_COMPUTE_UNITS;
	int					allocated_cl_n_groups;
	int					cl_n_groups = 0;
	int					cl_n_group_size;
	int					cl_n_parallel_units;

	//opencl kernels and opencl data ids
	cl_kernel	cl_kernel_compute_dists;
	cl_kernel	cl_kernel_find_membership;
	cl_kernel	cl_kernel_compute_centroids_first_step;
	cl_kernel	cl_kernel_compute_centroids_second_step;
	cl_program	cl_prog;

	cl_mem cl_features;
	cl_mem cl_centroids;
	cl_mem cl_dists;
	cl_mem cl_membership;
	cl_mem cl_centroids_group_1;
	cl_mem cl_centroids_group_n_1;
	cl_mem cl_centroids_group_2;
	cl_mem cl_centroids_group_n_2;

};