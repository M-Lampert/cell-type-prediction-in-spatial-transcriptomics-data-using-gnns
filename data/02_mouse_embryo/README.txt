Commented by Moritz Lampert:

Dataset URL: https://content.cruk.cam.ac.uk/jmlab/SpatialMouseAtlas2020/
Paper reference: T. Lohoff et al., “Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis,” Nat Biotechnol, vol. 40, no. 1, pp. 74–85, Jan. 2022, doi: 10.1038/s41587-021-01006-2.

The original dataset contains 57536 cells from 3 embryos. This dataset only uses the first embryo which contains 19451 cells.

The directory contains the following files:
	cell_by_gene.parquet: The gene expression matrix
	cluster_assignment.parquet: Cluster assignment with cluster id and cell type
	cell_coords.parquet: The cell coordinates as x and y coordinates.

README by the original authors:
The files above originate from the files of the original dataset described below as follows:
	cell_by_gene.parquet -> counts.Rds
	cluster_assignment.parquet -> MGA_joint_UMAP.Rds
	cell_coords.parquet -> metadata.Rds
------------------------------------------------------------------------------------------------------------------
This data contains the molecule, processed segmentation, processed expression, and 
imputed gene expression data from the seqFISH profiling of mouse embryos during 
organogenesis.

It is associated with the code in https://github.com/MarioniLab/SpatialMouseAtlas2020
Data can be visualised at https://marionilab.cruk.cam.ac.uk/SpatialMouseAtlas/

# Metadata

- metadata.Rds contains the associated information for each cell. uniqueID = unique 
cell ID, embryo, pos = field of view name, z = z-slice, x_global_affine = affine 
scaled x-position (largest values correspond to the tail region), y_global_affine = 
affine scaled y-position (lowest values correspond to the head region), embryo_pos, 
embryo_pos_z, Area = cell area in square pixels, UMAP1, UMAP2, celltype_mapped_refined, 
segmentation_vertices_x_global_affine = numericList of vertices to reconstruct 
segmentation, segmentation_vertices_y_global_affine =  = numericList of vertices to 
reconstruct segmentation.
- segmentation_vertices.Rds is a data frame containing the vertices to reconstruct 
segmentation.

# Gene expression

- mRNA.Rds contains the identified molecules for all genes. Additional information 
columns include: x = x-position within field of view, y = y-position within field of 
view, x_global = x-position within embryo, y_global = y-position within embryo
- counts.Rds contains the count matrix for all seqFISH genes in all identified cells.
- exprs.Rds contains the normalised gene expression matrix for all seqFISH genes in all 
identified cells.
- smFISH_counts.Rds contains the count matrix for all 36 smFISH genes in all identified cells.

# Imputed gene expression

- imputed.h5 contains the imputed gene expression matrix for whole-transcriptome in 
all identified cells.
- imputed_column_names.Rds contains the associated column (cell) names.
- imputed_row_names.Rds contains the associated row (gene) names.

# Cell-cell neighbourhood

- neighbourGraph_1.3.Rds is an R object containing igraph network of the cell-cell
neighbourhood, using an expansion factor of 1.3 on the cell segmentation.

# Joint integration with Mouse Gastrulation Atlas (MGA) data (Pijuan-Sala et al, 2019)

- MGA_joint_UMAP.Rds is a data.frame with columns for the cell name, joint UMAP 
coordinates, data type, embryo sample, joint subcluster, and cell type.
- MGA_joint_PCs.Rds is the corrected joint PC coordinates for both datasets,
rows correspond to cells and columns correspond to the 50 corrected PC coordinates.

# Shiny app input files

- Files in the `shiny_data` folder correspond to the `data` folder when running the Shiny app locally
