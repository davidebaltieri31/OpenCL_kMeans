##OpenCL/SSE/Multithreaded optimized k-means<br>

Author Davide Baltieri<br>
License LGPLv3<br>
It would be awesome if you'd let me know if you use this code...<br>
<br>

##Currently only for Windows/Visual Studio

* Requirese Microsoft's Parallel Patterns Library
* Requires SSE
* Requires OpenCL libs (Available with the NVIDIA CUDA Sdk or ATI equivalent)<br>

 
##Interface:

int cluster(&emsp;int&emsp;npoints,&emsp;&emsp;<code>/ number of data points /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&emsp;nfeatures,&emsp;&emsp;<code>/ number of attributes for each point /</code> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;std::vector&lt;float&gt;&#38;&emsp;feature_set,&emsp;&emsp;<code>/ matrix: [npoints][nfeatures] /</code> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&emsp;min_nclusters,&emsp;<code>/ range of min to max number of clusters /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&emsp;max_nclusters,<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;float&emsp;threshold,&emsp;<code>/ loop terminating factor, it's an integer value</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<code>threshold of 1 means one dimension in</code> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<code>one cluster center has moved/</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&#38;&emsp;best_nclusters,&emsp;&emsp;<code>/ out: number between min and max with lowest RMSE /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;std::vector&lt;float&gt;&#38;&emsp;cluster_centres,&emsp;<code>/ out: [best_nclusters][nfeatures] /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;float&#38;&emsp;min_rmse,&emsp;<code>/ out: minimum RMSE /</code> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;bool&emsp;isRMSE,&emsp;&emsp;<code>/ calculate RMSE, should always be true /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&emsp;nExternalLoops,&emsp;<code>/ number of iteration for each number of clusters /</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;bool&emsp;reuseOldCLusterCenters,&emsp;<code>/ each external loop reuse previously</code><br> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<code>computed cluster centers as starting point/</code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;int&emsp;nMaxInternalLoops&emsp;&emsp;<code>/ maximun number of iterations for a single clustering </code><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <code>job (usually 1000 should be enough)/</code><br>
);
<br>
Returns number of iteration to reach best cluster set.


