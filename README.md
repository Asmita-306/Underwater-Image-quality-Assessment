Underwater Image Quality Assessment Using Deep Learning
A deep-learning project for No-Reference Underwater Image Quality
Assessment (NR-UIQA). The system predicts a numerical
perceptual-quality score for an underwater image without requiring a
pristine reference image.

The project explores six ResNet-50-based architectures (M0--M5),
progressively investigating global features, patch-based learning,
attention, saliency, degradation awareness, and multi-scale feature
extraction.
Table of Contents
- Overview
- Research Motivation
- Dataset
- Methodology
- Model Architectures
- Training Configuration
- Evaluation Metrics
- Experimental Results
- Limitations
- Future Work
- Technology Stack
- Citation and Acknowledgement
Overview
Underwater images commonly suffer from:
- Colour distortion
- Low contrast
- Blurring
- Haze
- Reduced visibility caused by light absorption and scattering
These degradations make it difficult to assess image quality
consistently. This project investigates deep-learning approaches that
learn visual representations from underwater images and predict their
subjective quality scores.

Task: Regression
Input: RGB underwater image
Output: One numerical image-quality score
Backbone: ImageNet-pretrained ResNet-50
Dataset: UWIQA

This project is inspired by the research paper Deep Underwater Image
Quality Assessment With Explicit Degradation Awareness Embedding. It
does not claim to reproduce the complete EDANet architecture.
Instead, it investigates selected ideas through six ResNet-50-based
model variants.
Research Motivation
Traditional image-quality assessment methods may rely on handcrafted
features or require a reference image. No-Reference Image Quality
Assessment (NR-IQA) evaluates an image without requiring a corresponding
pristine reference.

The project studies whether global image features, local patches,
learned attention, saliency information, degradation-related features,
and multi-scale representations can help predict underwater image
quality.
Dataset
The experiments use the UWIQA dataset, containing 890 underwater
images with subjective quality-score annotations.

The reported split is:

  Subset         Number of images
  Training                    640
Validation                   72
Testing                     178
Total               890

The same split protocol is used for model comparisons. The test set is
reserved for final evaluation and is not used for training or model
selection.
Preprocessing
The paper describes the following preprocessing pipeline:
1. Convert images to RGB.
2. Resize images so that the smallest dimension is 256 pixels.
3. Apply centre cropping to obtain images of size 224 × 224.
4. Convert images to tensors.
5. Normalize using the ImageNet mean and standard deviation.
Dataset access: Obtain UWIQA from its authorized source. The
dataset is not included in this repository. Follow its licence and
usage conditions.

Methodology
All six models use an ImageNet-pretrained ResNet-50 backbone. The
original classification head is replaced with regression layers to
predict a quality score.

The overall pipeline is:
1. Load an underwater RGB image.
2. Apply preprocessing and normalization.
3. Extract features using ResNet-50.
4. Where applicable, divide the image into patches.
5. Apply the model-specific feature processing: averaging, attention,
   saliency, degradation-aware fusion, or multi-scale processing.
6. Aggregate features or local quality predictions.
7. Predict one image-quality score.
8. Evaluate predictions against the subjective annotations.
Model Architectures
M0 --- Baseline ResNet-50
- Processes the complete image.
- Extracts global features using ResNet-50.
- Uses a regression head consisting of a fully connected layer with
  512 neurons, ReLU, dropout of 0.3, and a final output neuron.
- Serves as the baseline for comparison.
M1 --- Patch Mean ResNet-50
- Divides the 224 × 224 image into four non-overlapping 112 × 112
  patches.
- Processes each patch using the shared ResNet-50 backbone.
- Extracts a 2048-dimensional feature vector from each patch.
- Averages the four feature vectors.
- Passes the aggregated representation to the regression network.
M2 --- Patch Attention ResNet-50
- Uses the same four-patch extraction strategy as M1.
- Extracts patch features using a shared ResNet-50 backbone.
- Uses an attention network with fully connected layers and Tanh
  activation to generate patch-importance scores.
- Applies Softmax to normalize the scores.
- Computes a weighted sum of patch features before regression.
M3 --- Saliency ResNet-50
- Generates a saliency map using the colour difference between each
  pixel and the image's mean colour.
- Divides the saliency map into four patches.
- Calculates the mean saliency score for each patch.
- Applies Softmax to obtain saliency weights.
- Extracts features from the corresponding original image patches.
- Combines patch features using the saliency weights and predicts the
  quality score.
M4 --- Triple-Branch Degradation-Aware Patch Network
M4 uses four 112 × 112 patches and a shared ResNet-50 backbone to
extract 2048-dimensional patch representations.

The representations are processed through three branches:
1. Quality branch: Predicts a local quality score for each patch
   using a regression network with sigmoid activation.
2. Saliency branch: Estimates the visual importance of each patch.
3. Degradation branch: Uses five sigmoid outputs to predict
   degradation attributes. The degradation scores are combined using
   beta coefficients and normalized using Softmax.
An importance-fusion network combines the branch information. The final
patch weights are used to aggregate local quality scores into one
image-quality prediction.
M5 --- Multi-Scale Degradation-Aware Patch Network
M5 extends M4 by processing the image at two scales:

  Scale       Grid         Patch size   Number of patches
  Coarse      2 × 2     112 × 112                   4
Fine        4 × 4       56 × 56                  16
Total                                        20

The patches are processed using a shared ResNet-50 backbone. Their
representations are passed through the quality, saliency, and
degradation branches used in M4. An importance-fusion network calculates
patch weights, and the local quality predictions are aggregated to
produce the final score.
Training Configuration
The paper reports the following training configuration:

  Parameter                        Setting
  Deep-learning framework          PyTorch
Backbone                         ResNet-50
Pretrained weights               ImageNet
Optimizer                        Adam
Loss function                    Mean Squared Error (MSE)
Maximum epochs                   50
Batch size                       8
Weight decay                     0.0001
Backbone learning rate           0.0001
Additional-layer learning rate   0.0003
Learning-rate scheduler          ReduceLROnPlateau
Early-stopping patience          8

The model checkpoint with the lowest validation loss is selected for
final testing.
Learning-Rate Screening
The experiments also report learning-rate screening using the first
three repetitions. The candidate learning rates are:
- 5 × 10⁻⁵
- 1 × 10⁻⁴
- 2 × 10⁻⁴
A learning rate is selected for each model using validation MSE. The
selected learning rates are then used for the remaining repetitions.

  Model     Selected learning rate
  M0                        0.0002
M1                       0.00005
M2                        0.0001
M3                        0.0001
M4                       0.00005
M5                        0.0001
Evaluation Metrics
The models are evaluated using five metrics:
- PLCC (Pearson Linear Correlation Coefficient): Measures linear
  correlation between predicted and subjective quality scores. Higher
  is better.
- SRCC (Spearman Rank-Order Correlation Coefficient): Measures
  agreement between predicted and subjective rankings. Higher is
  better.
- KRCC (Kendall Rank Correlation Coefficient): Measures pairwise
  ranking agreement. Higher is better.
- RMSE (Root Mean Squared Error): Measures prediction error in the
  score scale. Lower is better.
- MSE (Mean Squared Error): Measures average squared prediction
  error. Lower is better.
Experimental Results
The paper reports mean ± standard deviation over ten repetitions.
Before learning-rate tuning
  Model                 PLCC           SRCC           KRCC           RMSE
  M0            0.812269 ±   0.804495 ±   0.662618 ±     0.091347 ±
0.018412     0.022840     0.027443       0.009169

  M1              0.803230 ±     0.795134 ±     0.653911 ±     0.091801 ±
0.031082       0.033093       0.038298       0.008113

  M2              0.758976 ±     0.747899 ±     0.601765 ±     0.098756 ±
0.031703       0.037767       0.036963       0.006099

  M3              0.801394 ±     0.793179 ±     0.650431 ±   0.091118 ±
0.021180       0.020045       0.021479     0.004544

  M4              0.790264 ±     0.778870 ±     0.636454 ±     0.094050 ±
0.023306       0.029718       0.030058       0.004907
M5              0.794671 ±     0.784998 ±     0.644386 ±     0.093019 ±
0.033765       0.028631       0.027318       0.007292
After learning-rate tuning
  Model             PLCC         SRCC         KRCC         RMSE          MSE
  M0          0.8331 ±   0.8302 ±   0.6906 ±   0.0887 ±   0.0079 ±
0.0247     0.0286     0.0296     0.0069     0.0012

  M1            0.8167 ±     0.8089 ±     0.6711 ±     0.0912 ±     0.0084 ±
0.0349       0.0371       0.0394       0.0078       0.0014

  M2            0.7792 ±     0.7719 ±     0.6251 ±     0.0992 ±     0.0099 ±
0.0353       0.0321       0.0325       0.0070       0.0014

  M3            0.8234 ±     0.8167 ±     0.6773 ±     0.0899 ±     0.0082 ±
0.0333       0.0366       0.0367       0.0083       0.0015

  M4            0.8071 ±     0.7962 ±     0.6540 ±     0.0941 ±     0.0089 ±
0.0289       0.0345       0.0358       0.0063       0.0012
M5            0.8278 ±     0.8264 ±     0.6870 ±     0.0893 ±     0.0080 ±
0.0229       0.0277       0.0282       0.0054       0.0009
Key Findings
- M0 (global ResNet-50) achieves the best overall performance after
  tuning.
- M5 is the strongest proposed architecture among M1--M5 after tuning.
- M5 performs close to M0, but does not surpass it on the reported
  metrics.
- The results indicate that adding architectural complexity does not
  necessarily improve performance on this dataset.
- Learning-rate tuning improves the reported performance of the
  models.
The paper also includes ground-truth versus predicted-score plots,
training and validation loss curves, and residual plots to examine
prediction behaviour and generalization.
Limitations
The study identifies the following limitations:
1. Limited dataset size: UWIQA contains 890 images, which may not
   represent all underwater conditions.
2. Discrete quality annotations: The subjective scores are
   discrete, which may introduce uncertainty in regression.
3. Random-split dependence: Repeated random splits do not guarantee
   fully independent test sets.
4. Need for ablation studies: Further experiments are needed to
   isolate the contribution of patch learning, saliency, attention, and
   degradation-aware components.
5. Limited hyperparameter search: Learning-rate selection was
   screened using the first three repetitions to manage computational
   cost. Broader tuning may provide a more comprehensive comparison.
Future Work
Potential directions include:
- Expanding the dataset and evaluating on additional underwater
  image-quality datasets.
- Performing component-wise ablation studies.
- Investigating more extensive hyperparameter optimization.
- Improving robustness and generalization across different underwater
  environments.
Technology Stack
The paper identifies the following tools and libraries:
- Python
- PyTorch
- ResNet-50
- OpenCV
- Pillow (PIL)
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
Citation and Acknowledgement
This project is inspired by:

Q. Jiang, Y. Gu, Z. Wu, C. Li, H. Xiong, F. Shao, and Z. Wang, "Deep
Underwater Image Quality Assessment With Explicit Degradation Awareness
Embedding," IEEE Transactions on Image Processing, vol. 34,
pp. 1297--1310, 2025.

The project explores selected concepts from this work and does not claim
to be a complete implementation or reproduction of EDANet.
Repository Setup and Reproducibility Note
This README documents the methodology and results described in the
project paper. The paper does not specify the exact source-code entry
points, command-line arguments, dependency versions, or dataset download
URL. Add those details after checking the actual repository
implementation rather than assuming commands or file names.

Suggested information to add when available:
- Exact installation command and dependency file (requirements.txt
  or equivalent)
- Dataset download/access instructions
- Expected dataset directory structure
- Training commands for M0--M5
- Evaluation commands
- Random seeds and split-generation procedure
- Link to the actual project repository
