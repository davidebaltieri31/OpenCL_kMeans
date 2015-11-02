# OpenCL_kMeans
OpenCL based k-Means clustering

int cluster(int      npoints,				/* number of data points */
			int      nfeatures,				/* number of attributes for each point */
			std::vector<float>& feature_set,/* matrix: [npoints][nfeatures] */
			int      min_nclusters,			/* range of min to max number of clusters */
			int		 max_nclusters,
			float    threshold,						/* loop terminating factor, it's an integer factor
													   a threshold of 1 means one dimension in one cluster center has moved*/
			int&     best_nclusters,				/* out: number between min and max with lowest RMSE */
			std::vector<float>& cluster_centres,	/* out: [best_nclusters][nfeatures] */
			float&	 min_rmse,						/* out: minimum RMSE */
			bool	 isRMSE,						/* calculate RMSE, should always be true */
			int		 nExternalLoops,				/* number of iteration for each number of clusters */
			bool	 reuseOldCLusterCenters,        /* each external loop reuse previously computed cluster centers as starting point*/
			int		 nMaxInternalLoops              /* maximun number of iterations for a single clustering 
													   job (usually 1000 should be enough)*/
			);
			
Currently only for Windows/Visual Studio (requirese Microsoft's Parallel Patterns Library)

