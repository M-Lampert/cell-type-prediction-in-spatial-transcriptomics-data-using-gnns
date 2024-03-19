Commented by Moritz Lampert:

Dataset URL: https://datadryad.org/stash/dataset/doi:10.5061/dryad.jm63xsjb2
Paper reference: V. Petukhov et al., “Cell segmentation in imaging-based spatial transcriptomics,” Nat Biotechnol, vol. 40, no. 3, pp. 345–354, Mar. 2022, doi: 10.1038/s41587-021-01044-w.

The directory contains the following files:
	cell_by_gene.parquet: The gene expression matrix
	cluster_assignment.parquet: Cluster assignment with cluster id and cell type
	cell_coords.parquet: The cell coordinates as x and y coordinates.

README by the original authors:
The files above originate from the files of the original dataset described below as follows:
	cell_by_gene.parquet -> data_analysis/baysor_membrane_prior/segmentation_counts.csv
	cluster_assignment.parquet -> data_analysis/baysor_membrane_prior/cell_assignment.csv
	cell_coords.parquet -> data_analysis/baysor_membrane_prior/segmentation_cell_stats.csv
-----------------------------------------------------------------

File Organization for MERFISH Mouse Ileum Dataset (Baysor 2021)


Rosalind J. Xu, Moffitt Lab
Boston Children’s Hospital
May 2021


file_organization 

	definition of columns for some of the data files (see below) 



raw_data

	raw_data/dapi_stack.tif: dapi signal for select mouse ileum region

	raw_data/membrane_stack.tif: Na+/K+ - ATPase immunofluorescence signal for select mouse ileum region

		Each tif stack contains 9 z-planes spaced by 1.5 um apart, from 2.5 um (first frame) to 14.5 um (last frame) above the plane of the coverslip (0 um) 
		
	raw_data/molecules.csv: mRNAs (gene name, locations, area, brightness, quality score) for select mouse ileum region
		
		See file_organization/molecules.txt for column definitions
		



data_analysis

	data_analysis/baysor: Baysor (mRNA-only) segmentation of select mouse ileum region

	data_analysis/baysor_membrane_prior: Baysor (with Cellpose membrane segmentation prior) segmentation of select mouse ileum region

	data_analysis/cellpose: Cellpose (membrane or DAPI-based) segmentation of select mouse ileum region




data_analysis/baysor and data_analysis/baysor_membrane_prior are organized similarly. The contents are: 

	segmentation: Baysor segmentation outputs. See https://github.com/kharchenkolab/Baysor#outputs for detailed definitions 

		segmentation_counts.csv: Cell-by-gene matrix. Rows: genes; Cols: cells 

		segmentation_cell_stats.csv: Cell metadata. Each cell corresponds to each column in segmentation_counts.csv

		segmentation.csv: mRNA metadata. Each row corresponds to one mRNA in raw_data/molecules.csv

		segmentation_params.dump: Baysor segmentation parameters

		segmentation_diagnostics.html: Convergence of the Expectation - Maximization algorithm 

		poly_per_z.json: Polygon representation of Baysor cell boundaries

		segmentation_borders.html: Visualization of cell boundaries
		

	clustering: single-cell clustering results

		cell_assignment.csv: Assignment of each cell to cell type clusters. Each row corresponds to one cell in segmentation_cell_stats.csv (or one column in segmentation_counts.csv)
			
			Cells assigned as "Removed" are filtered out (not assigned to a cluster) during single-cell analysis 

		cluster_counts.csv: Counts of cells in each cluster

		marker_genes.csv: Marker gene statistics for each cluster 

			See file_organization/marker_genes.txt for column definitions




data_analysis/cellpose: 

	cell_boundaries: Cellpose cell boundaries

		training: Manually labeled cell boundaries for training Cellpose models 

		models: Cellpose models for segmenting DAPI and membrane stains

		results: Cellpose boundaries based on DAPI or Na+/K+ - ATPase immunofluorescence signal (membrane), where pixels covered by the same cell are masked with the same number across z-stacks
	
			The pixels corresponds to the pixels in raw_data/dapi_stack.tif and raw_data/membrane_stack.tif, as well as the pixel coordinates (x_pixel, y_pixel, z_pixel) in raw_data/molecules.csv

	
	segmentation: mRNA partitioning into cells based on Cellpose cell boundaries (derived from membrane stain)

		segmentation_counts.csv: Cell-by-gene matrix

		cell_coords.csv: x y coordinates in pixels for all cells, where each cell corresponds to one column in segmentation_counts.csv


	clustering: single-cell clustering results

		cell_assignment.csv: Assignment of cell to cell type clusters. Each row corresponds to one cell in segmentation_cell_stats.csv (or one column in segmentation_counts.csv)
			
			Cells assigned as "Removed" are filtered out (not assigned to a cluster) during single-cell analysis 

		cluster_counts.csv: Counts of cells in each cluster

		marker_genes.csv: Marker gene statistics for each cluster 

			See file_organization/marker_genes.txt for column definitions


		





		

		



		
