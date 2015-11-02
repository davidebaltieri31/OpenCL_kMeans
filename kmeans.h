/*
*	OpenCL/SSE/Multithreaded optimized k-means
*	Copyright 2015 Davide Baltieri
*	Author Davide Baltieri
*	License LGPLv3
*	It would be awesome if you'd let me know if you use this code...
*/
#pragma once

#include <vector>

/*THIS IS THE ONLY FUNCTION YOU HAVE TO CALL*/
//Perform KMEANS clustering
//this function is NOT thread safe!!
/*---< cluster() >-----------------------------------------------------------*/
int cluster(const int      npoints,					/* number of data points */
			const int      nfeatures,				/* number of attributes for each point */
			const std::vector<float>& feature_set,	/* matrix: [npoints][nfeatures] */
			const int      min_nclusters,			/* range of min to max number of clusters */
			const int		 max_nclusters,
			const float    threshold,				/* loop terminating factor, it's an integer factor
													   a threshold of 1 means one dimension in one cluster center has moved*/
			int&     best_nclusters,				/* out: number between min and max with lowest RMSE */
			std::vector<float>& cluster_centres,	/* out: [best_nclusters][nfeatures] */
			float&	 min_rmse,						/* out: minimum RMSE */
			const bool	 isRMSE,					/* calculate RMSE, should always be true */
			const int		 nExternalLoops,		/* number of iteration for each number of clusters */
			const bool	 reuseOldCLusterCenters,    /* each external loop reuse previously computed cluster centers as starting point*/
			const int		 nMaxInternalLoops      /* maximun number of iterations for a single clustering
													   job (usually 1000 should be enough)*/
			);



