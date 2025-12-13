Random Forest Classifier for Power System Anomaly Detection

This project implements a Random Forest-based classifier for detecting anomalies in power systems using GAN-augmented data. The system processes voltage and current measurements to classify three types of events: normal operations, cyber-attacks, and system faults.

The implementation begins by loading data from a CSV file named 'TrainData_GANData.csv', which must contain voltage measurements (V_gu1-4), current measurements (Idc1-4), and target labels (0: Normal, 1: Attack, 2: Fault). The data processing pipeline extracts six key features: mean and standard deviation of voltage and current signals, plus frequency-domain features obtained through Fourier transform analysis.

Before model training, the system performs comprehensive data visualization including feature distribution histograms, correlation heatmaps, and class distribution analysis. These visualizations are saved as high-quality PDF files for documentation and analysis purposes. The feature engineering process combines time-domain and frequency-domain characteristics to create a robust feature set for classification.

A Random Forest classifier is implemented with balanced class weighting to handle potential class imbalances. The model undergoes hyperparameter optimization through grid search, testing combinations of tree count (50, 100, 200), maximum depth (1, 10, 20), minimum samples split (2, 5, 10), and minimum samples leaf (1, 2, 4). Regularization is achieved through Gaussian noise addition rather than dropout layers.

The evaluation process generates multiple performance metrics including accuracy, precision (both macro and weighted averages), recall, F1-score, and ROC AUC. Visualization outputs include ROC curves with zoomed insets, confusion matrices, feature importance plots, and 3D scatter plots showing feature relationships across different classes.

The system produces several output files organized into two categories: visualization PDFs and data CSVs. The visualizations include histogram distributions, correlation matrices, ROC curves, confusion matrices, and 3D plots. The data files contain class distributions, correlation matrices, feature importance scores, performance metrics, ROC curve data points, and confusion matrix values for further analysis.

This implementation is designed for research and practical applications in power system monitoring and anomaly detection. The code includes comprehensive error handling, cross-validation procedures, and publication-quality visualization settings with proper font configurations for both Western and Chinese character sets.
