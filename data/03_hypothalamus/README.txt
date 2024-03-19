Commented by Moritz Lampert:

Dataset URL: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248
Paper reference: J. R. Moffitt et al., “Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region,” Science, vol. 362, no. 6416, p. eaau5324, Nov. 2018, doi: 10.1126/science.aau5324.

The original dataset contains 1027848 cells from multiple mouse brains. This dataset only uses the first female brain data that has a "Bregma" value of -0.24 which contains 6412 cells.
(Wikipedia on "Bregma": The bregma is the anatomical point on the skull at which the coronal suture is intersected perpendicularly by the sagittal suture. (https://en.wikipedia.org/wiki/Bregma))

The directory contains the following files:
	cell_by_gene.parquet: The gene expression matrix
	cluster_assignment.parquet: Cluster assignment with cluster id and cell type
	cell_coords.parquet: The cell coordinates as x and y coordinates.
