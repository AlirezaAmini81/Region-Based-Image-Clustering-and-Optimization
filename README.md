# Computational-Intelligence-Project


## Phase_1: Image Clustering and Classification

This project focuses on clustering and classifying image data using advanced computational intelligence techniques. The workflow involves feature extraction, clustering, and classification, with thorough analysis and optimization for performance enhancement.

### Project Objectives
- Perform feature extraction from images (color and spatial features).
- Cluster image pixels using DBSCAN and K-Means algorithms.
- Extract statistical and spatial features for further analysis.
- Construct feature histograms for images.
- Classify image regions using a Random Forest classifier.
- Evaluate model performance with metrics such as Precision, Recall, F1-Score, and confusion matrices.
- Optimize clustering parameters to enhance classification accuracy.

### Key Steps
1. **Data Loading**: Load image data and labels from pickle files.
2. **Feature Extraction**: Extract color (HSV) and spatial (x, y) features for each pixel.
3. **Clustering**: Group pixels using:
   - **DBSCAN**: Density-based clustering with varied radius and neighborhood size.
   - **K-Means**: Partition-based clustering with optimal K selection.
4. **Feature Engineering**:
   - Compute statistical features (mean, variance, skewness, kurtosis) for each cluster.
   - Derive spatial features (bounding box, area, orientation).
5. **Histogram Construction**: Create feature histograms for each image to summarize cluster information.
6. **Classification**: Classify images using Random Forest, leveraging the extracted features.
7. **Performance Evaluation**:
   - Compute metrics: Accuracy, Precision, Recall, F1-Score.
   - Analyze confusion matrices and identify misclassifications.
8. **Optimization**:
   - Experiment with different feature ratios and clustering parameters.
   - Evaluate clustering quality using Davies-Bouldin and Calinski-Harabasz indices.

### Results
- Best clustering results achieved using K-Means with \( K=5 \).
- Random Forest classification accuracy: **81.37%**.
- Identified key factors affecting misclassification, including insufficient color feature diversity.
- Suggested improvements: Additional feature extraction (e.g., shape-based features) and optimized sampling.


### Future Improvements
- Incorporate deep learning models for feature extraction.
- Use additional clustering algorithms (e.g., Mean-Shift, Spectral Clustering).
- Perform segmentation on larger datasets for scalability testing.
