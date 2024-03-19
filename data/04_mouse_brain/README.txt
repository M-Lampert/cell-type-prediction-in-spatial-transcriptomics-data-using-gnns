Commented by Moritz Lampert:

Dataset URL: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/
Paper reference: M. Zhang et al., “Spatially resolved cell atlas of the mouse primary motor cortex by MERFISH,” Nature, vol. 598, no. 7879, pp. 137–143, Oct. 2021, doi: 10.1038/s41586-021-03705-x.

The dataset is a subset of the dataset. We use the data from mouse 1 sample 3 and slice 153 because it is the slice with the most cells.

The directory contains the following files:
	cell_by_gene.parquet: The gene expression matrix
	cluster_assignment.parquet: Cluster assignment with cluster id and cell type
	cell_coords.parquet: The cell coordinates as x and y coordinates.

README by the original authors:
The files above originate from the files of the original dataset described below as follows:
	cell_by_gene.parquet -> counts.h5ad
	cluster_assignment.parquet -> cell_labels.csv
	cell_coords.parquet -> ??
-------------------------------------------------------------------------------
This data collection contains spatially resolved single-cell transcriptomics datasets acquired using MERFISH on the mouse primary motor cortex (MOp) collected by the Xiaowei Zhuang Lab at Harvard University and Howard Hughes Medical Institute.

* The dataset contains MERFISH images of 12 experiments, which include 64 coronal slices of the MOp region (10 um thick slices, every 100um interval) collected from 2 biological replicates. For each mouse, 6 experiments were performed, and each experiment was named with the mouse id plus the sample id, e.g. mouse1_sample1, mouse2_sample3. For each experiment, multiple coronal slices (4-6 slices) were included on the same coverslip and were imaged together.

* In this dataset, a total of 258 genes were imaged. Among the 258 genes, 242 genes were imaged using MERFISH, which encodes individual genes with error-robust barcodes (22-bit binary codes in this case), imprints the barcodes onto the RNAs using combinatorial labeling with encoding probes, and measures the barcodes bit-by-bit using sequential hybridization of readout probes. The 22 bits are imaged in 11 hybridization rounds with two-color imaging each round. The remaining 16 genes were imaged by 8 sequential rounds of two-color FISH.

* Each of the subdirectory folders contains either the raw (e.g. mouse1_sample1_raw) or processed images (e.g. mouse1_sample1_processed) of one experiment. Each experiment contains many fields of view (FOVs) and each tiff file in the folder corresponds to the images of one FOV. The raw image files are named as aligned_images plus the FOV id (e.g. aligned_images0.tif); the processed image files are named as processed_images plus the FOV id (e.g. processed_images100.tif).

•	{sample_id}_raw: folders containing raw images. Each raw image file, corresponding to one FOV, is a stacked tiff file of multiple frames, each frame corresponding to one z-plane of one channel and each channel corresponding to one bit of the MERFISH imaging process, or one gene imaged in the sequential hybridization process, or the DAPI and polyT images used for cell segmentation. Seven z-planes are imaged for each channel. Images are aligned by fiducial bead fitting across each imaging round. The tiff stacks are ordered as channel 1 z-planes 1 through 7, channel 2 z-planes 1 through 7, …, channel 40 z-plane 1 through 7. See data_organization_raw.csv file for detailed channel information.

•	{sample_id}_processed: folders containing processed images. Each processed image file, corresponding to one FOV, is a stacked tiff file of multiple frames, each frame corresponding to one z-plane of one channel and each channel corresponding to one bit of the MERFISH imaging process, or the DAPI and polyT images used for cell segmentation. Seven z-planes are imaged for each channel. Images are aligned by fiducial bead fitting across each imaging round, and processed with a high pass filter and deconvolution. The tiff stacks are ordered the same as the raw images, except that images for the genes imaged by straight sequential hybridization are not included. See data_organization_processed.csv file for detailed channel information, and preprocessing.json file for parameters used in image processing. 
Note: For the 650 nm channels, a significant number of spots observed the first z-plane (i.e. at the coverslip surface) correspond to non-specific binding of the 650 nm dye to the coverslip surface, and the vast majority of these non-specific binding spots are decoded as invalid barcodes in the decoding process and are not used for subsequent analysis.  

* The processed_data folder contains the following files:
•	segmented_cells_{sample_id}.csv: Segmented cell boundary coordinates of each z-plane and the slice id that each cell belongs to for each experiment. Note that each experiment includes 4-6 tissue slices on a single coverslip, and the slice id gives the slice number that the cells belong to.
•	spots_{sample_id}.csv: Decoded spot location (x,y,z) in the unit of microns and their target gene identity for each experiment. The same coordinate system was used in segmented_cells_{sample_id}.csv and spots_{sample_id}.csv for each experiment, and hence the spots can be parsed into the segmented cells (and also the slices) according by comparison of their coordinates with the cell boundary locations.  
•	counts.h5ad: Cell by gene matrix of the whole dataset. The counts are normalized by the volume of each cell.
•	cell_labels.csv: Cluster label assignment for each cell.

We also provide these files that are associated to this dataset:
•	genes_combinatorial.txt: List of target genes that are imaged in combinatorial imaging rounds.
•	genes_sequential.txt: List of target genes that are imaged sequentially after the combinatorial imaging rounds.
•	probes.fasta: Provides sequences all encoding probes used for hybridization.
•	codebook.csv: Provides the barcodes that encode individual genes measured in the combinatorial imaging rounds.
•	data_organization_raw.csv: Provides information on how individual channels and z-planes are ordered in the multi-frame tiff file for each field of view of raw images.
•	data_organization_processed.csv: Provides information on how individual channels and z-planes are ordered in the multi-frame tiff file for each field of view of processed images.
•	microscope.json: Provides parameters of the microscope that was used with pixel size and orientation of images.
•	preprocessing.json: Provides values and filters used in generating the processed images.
•	fov_positions folder: contains the txt files of field of view positions for individual samples, named by its sample id.
