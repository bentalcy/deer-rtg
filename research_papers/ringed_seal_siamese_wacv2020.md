# ringed_seal_siamese_wacv2020


## Page 1

Siamese Network Based Pelage Pattern Matching for Ringed Seal
Re-identiﬁcation
Ekaterina Nepovinnykh Tuomas Eerola Heikki K ¨alvi¨ainen
Computer Vision and Pattern Recognition Laboratory (CVPRL)
Department of Computational and Process Engineering
Lappeenranta-Lahti University of Technology LUT, Lappeenranta, Finland
firstname.lastname@lut.fi
Abstract
In this paper we propose a method to match pelage
patterns of the Saimaa ringed seals enabling the re-
identiﬁcation of individuals. First, the pelage pattern is ex-
tracted from the seal’s fur using a method based on the Sato
tubeness ﬁlter . After this, the similarities of the pelage pat-
tern patches are computed using a siamese network trained
with a triplet loss function and a large dataset of manu-
ally selected patches. The similarities are then used to ﬁnd
the best matching patches from the images in the database
of known individuals. Furthermore, we employ the pro-
posed pattern matching method to build a full framework for
the ringed seal re-identiﬁcation, consisting of CNN-based
animal segmentation, patch correspondence detection, and
ranking the images in the database of known seal individu-
als based on the similarity to the query image. Our experi-
ments on challenging datasets of Saimaa ringed seals show
that the proposed method achieves promising identiﬁcation
results, providing a useful tool for the Saimaa ringed seal
monitoring.
1. Introduction
Automatic wildlife camera traps and crowd-sourced im-
age material provide novel possibilities to monitor endan-
gered animals species. However, massive image volumes
that these methods produce is overwhelming for biologists
to go through, which calls for automatic systems to perform
the analysis. The main task is to identify the animal individ-
uals in the images to provide the basis for the monitoring,
including population size estimation and animal migration
tracking.
Saimaa ringed seals (Pusa hispida saimensis) are endan-
gered due to various anthropogenic factors, such as random
bycatch and climate change. In addition, the risk of their
extinction is high due to low genetic diversity and small
population (currently around 400 individuals). The current
knowledge about this animal is mainly based on telemet-
ric studies with a relatively small number of individuals.
Photo-ID (photo-identiﬁcation) using camera traps is an ap-
proved and effective non-invasive method for studying and
monitoring the Saimaa ringed seals [14]. Ringed seals have
permanent pelage patterns that are unique to each individual
and can be used for the identiﬁcation (see Fig. 1).
Figure 1. Saimaa ringed seal identiﬁcation based on pelage pattern
patches.
Automatic methods for the re-identiﬁcation of individual
animals have been proposed for various species. However,
the re-identiﬁcation of ringed seals introduces some addi-
tional challenges. First, large variation in possible poses is
further exacerbated by the deformable nature of those an-
imals. This, in addition to the fact that the pelage pattern
is non-uniform and depends on the visible area of the ani-
mal, limits the size of the regions that could actually be used
for the identiﬁcation task. Second, the contrast between the
ring pattern and the rest of the pelage is low and the appear-
ance of the pattern varies between wet and dry fur. Finally,
image quality of automatic camera traps is typically low,
which might lead to the loss of details. These challenges
make the re-identiﬁcation considerably more difﬁcult for
25


## Page 2

ringed seals than, for example, for zebras with clearly visi-
ble pattern and limited variation in the pose of the torso.
In this paper, we propose a method for comparison and
matching of ringed seal pelage pattern patches. The method
starts by extracting the pelage patterns from the images by
utilizing the Sato tubeness ﬁlter based method [23]. This
ﬁlter can be used to detect continuous ridges, e.g. tubes,
wrinkles, rivers, or, in our case, pelage pattern. It calculates
the eigenvectors of the Hessian to compute the similarity of
an image region to tubes. This step gets rid of the irrele-
vant factors, such as illumination, and focuses the attention
of the algorithm on the actual pattern itself. This is crucial
since most of the data is collected with automatic camera
traps and the ringed seals tend to stay in the same region.
The same seal is often captured with the same camera, the
same background, and the similar illumination and pose.
This causes supervised methods to learn superﬁcial features
which might appear as good features in the training data, but
do not help to re-identify the seals in the real-world appli-
cation scenarios. To calculate similarities between patterns
and to match them to each other, a siamese network trained
with a triplet loss function is used.
We build a full framework for the identiﬁcation of
Saimaa ringed seal utilizing the pattern matching. The
framework starts with CNN based segmentation of the seal
from the background. This allows reliable pattern extrac-
tion. Pattern patches are then extracted from the query im-
age and the best matches for each patch are searched from
the database of patches from the known seals. The ﬁnal re-
identiﬁcation is done based on the similarity of the pattern
patches. Each patch of the query image is compared to all
patches from the database image, and a similarity heatmap
is built. Local maxima are used as candidates for this patch
projection. A geometrically aware algorithm then selects
suitable projection sets for the entire image and ranks the
comparative similarity.
In the experimental part of the work, we demonstrate
that the proposed pelage pattern matching provides high
matching accuracy outperforming siamese network applied
for unprocessed patches. Moreover, we show promising
identiﬁcation results, providing a useful semi-automatic re-
identiﬁcation tool for biologists. The system offers N pos-
sible candidates and the user manually chooses the corre-
sponding animal individual.
To summarize, this paper makes the following contribu-
tions: 1) a pattern extraction algorithm that works on im-
ages captured in the wild and reduces the impact of exter-
nal conditions such as lighting, weather and location, and
2) a novel end-to-end Saimaa ringed seal re-identiﬁcation
method that utilizes a siamese triplet network for compar-
ing distinct image regions and a matching algorithm that
checks topological consistency of corresponding points.
2. Related work
2.1. Animal re­identiﬁcation
Traditional tools for monitoring animals such as tagging
requires a physical contact with the animal which causes
stress and may change the behavior of the animal. To avoid
this, camera-based methods utilizing computer vision al-
gorithms have been developed for animal re-identiﬁcation.
Many of them are species-speciﬁc which limits their usabil-
ity [19, 22, 10].
There have also been research efforts towards creating
a uniﬁed approach applicable for identiﬁcation purposes
for several animal species. Wildbook [3] is a large-scale
project for the study, monitoring and identiﬁcation of ani-
mals with distinguishable marks on the body. Wildbook’s
computer vision based identiﬁcation methods are build on
the HotSpotter algorithm [7]. This algorithm is not species
speciﬁc and has been applied to Grevy’s and plain zebras,
giraffes, leopards, and lionﬁsh. HotSpotter uses viewpoint
invariant descriptors and a scoring mechanism that empha-
sizes the most distinctive keypoints and descriptors. In [26],
a species recognition algorithm based on sparse coding
spatial pyramid matching (ScSPM) was proposed. It was
shown that the proposed object recognition techniques can
be successfully used to identify animals on sequences of
images captured using camera traps in nature.
Due to the recent progress in deep learning, convo-
lutional neural networks (CNN) have also become popu-
lar tools for animal biometrics. For example, in [2], re-
identiﬁcation of the cattle using CNN approach combined
with k-NN classiﬁer was proposed. The method achieved
the accuracy of over 80% outperforming competing meth-
ods. The approach is, however, speciﬁc to muzzle patterns
of cattle. The muzzle patterns are obtained manually, pro-
viding consistent data that simpliﬁes the re-identiﬁcation.
A typical problem in the wildlife animal re-identiﬁcation
is that it is practically impossible to collect a large dataset
with large number of images for all individuals. Often the
method needs to be able to identify an individual with only
one or few previously collected examples. Moreover, the
animal re-identiﬁcation method should be able to recognize
if the query image contains an individual that is not in the
database of the known individuals. Recently, siamese neu-
ral network based approaches have gained popularity in the
task of animal re-identiﬁcation [13]. These methods pro-
vide a tool to classify objects based on only one example
image (one-shot learning) and to recognize if it belongs to
a class which the network has never seen. For example,
in [24], the effectiveness of siamese neural networks for re-
identiﬁcation of Human, Chimpanzee, Humpback Whale,
Fruit Fly, and Octopus was demonstrated.
In [20], natural markings of manta rays were used for
pose invariant re-identiﬁcation. The method uses a CNN ap-
26


## Page 3

proach with the semi-hard triplet mining strategy, the triplet
loss function, and an extensive geometric augmentation of
the input images. The method achieved 65 % Top-1 ac-
curacy and 97 % Top-10 accuracy. However, it should be
noted that the method requires the user input to localize the
region of interest.
In [15], a method that combines the CNN baseline with
pose-estimation to detect and re-identify Amur tigers was
presented. It achieved the Top-5 accuracy of 90%. In [18], a
three-module deep CNN architecture in order to learn com-
plementary, non-obvious features as well as obvious ones
was used for the Amur tiger identiﬁcation. The ﬁrst mod-
ule learns embeddings from an image as usual, the second
module utilizes the same architecture, but receives an image
with removed parts that correspond to areas of interest of the
ﬁrst network, and the third module combines their embed-
dings for the ﬁnal result. The method achieves the Top-5
accuracy of 91,6%. In [17], Amur tiger images were sepa-
rated into streams that are utilized by different embedding
networks: trunk (body) parts and limb parts. Trunk fea-
ture vectors are learned by a network with 8 vertical stripes,
while limb feature vectors are learned with multiple-branch
network for different limb components.The method demon-
strates the best result on the proposed dataset with the Top-5
accuracy of 95,3%.
In [27, 5, 21], the re-identiﬁcation of the Saimaa ringed
seals was considered. In [27], a superpixel based segmenta-
tion method and a simple texture feature based ringed seal
identiﬁcation method were presented. In [5], additional pre-
processing steps were proposed and two existing species
independent individual identiﬁcation methods were evalu-
ated. However, the identiﬁcation performance of neither of
the methods is good enough for most practical applications.
In [21], the re-identiﬁcation of the Saimaa ringed seals was
formulated as a classiﬁcation problem and was solved us-
ing transfer learning. While the performance was high on
the used test set, the method is only able to reliably perform
the re-identiﬁcation if there is a large set of examples for
each individual. Furthermore, the whole system needs to be
retrained if a new seal individual is introduced. Finally, it is
unclear if the high accuracy was due to the methods ability
to learn the necessary features from the fur pattern, or if it
also learned features such as pose, size, or illumination that
separated individuals in the used dataset, but do not provide
the means to generalize the methods to other datasets.
2.2. Siamese networks
The task of re-identiﬁcation of people and animals [16, 4,
8] could be formulated in terms of learning a distance metric
between individuals. The general strategy is to train a model
to discriminate between a collection of same/different pairs.
Since the model learns generic embeddings rather than a
rigid classiﬁer it is able to better generalize to new classes
that have not been used during the training.
The triplet neural network [12, 25] is an extension of the
Siamese Neural Network where three input samples are si-
multaneously considered in the loss function. The goal is
to learn an embedding such that the distance between sim-
ilar embedded samples is closer than the distance between
dissimilar samples.
During the training, three inputs are sampled and run
through the same embedding net: anchor xa, positive xp,
and negative xn samples. The loss function is calculated
using the embedded representations of those samples as fol-
lows:
Ltriplet(xa, xp, xn) = max(0, m+
∥ f (xa) − f (xp) ∥2
2 − ∥ f (xa) − f (xn) ∥2
2),
(1)
where f (·) is the embedding network, and m is some mar-
gin. This loss function turns to zero when the positive dis-
tance is smaller than the negative by more than a speciﬁed
margin. If the difference is smaller than the margin, or if
the positive distance is larger than the negative, the loss is
non-zero.
In [11], the concept of triplet mining is being discussed.
Triplet mining refers to a strategy of selecting triplets for
training. The issue is that the number of triplets grows cu-
bically with the size of the dataset, and most of those triplets
eventually become useless for learning. If one tries to learn
the concept of the ”same individual” then being shown pic-
tures of individuals with different clothes over and over
again does not improve the learning. However, being shown
similar-looking but different individuals, or the same in-
dividual in different poses, improves the learning dramat-
ically. Hard triplet mining strategy aims to achieve that by
only giving the network examples hard negative examples
(similar-looking but different) and hard positive examples
(differently-looking but similar). The issue with hard triplet
mining is that the network is essentially learning on a heav-
ily outlier-biased selection of triplets which can result in
poor performance on comparatively ”easy” tasks. Semi-
hard strategy alleviates that by including ”moderate” exam-
ples on either negative side, positive side, or both.
3. Saimaa ringed seal re-identiﬁcation
The proposed re-identiﬁcation process for the Saimaa
ringed seals is shown in Fig. 2. First, the seal is segmented
from the background. This step is crucial since most of the
images are obtained using static camera traps. Therefore,
the same seal is often captured with the same background
increasing the risk that the supervised identiﬁcation algo-
rithm learns to “identify” the background instead of the ac-
tual seal if the full image or the bounding box around the
seal is used. This may result in a system that is unable to
identify the seal in a new environment. After segmentation,
27


## Page 4

the seal is cropped to bounding box and the pelage pattern is
extracted. The region in the pattern image corresponding to
the seal segment is then divided into small patches. Finally,
the identiﬁcation is performed by ﬁnding the most similar
patches in the patch database of the known seals.
Figure 2. Proposed identiﬁcation algorithm.
3.1. Segmentation
To segment the seal from the background the Deeplab
model [6] is used. DeepLab is a state-of-the-art deep learn-
ing model for semantic image segmentation. It contains
three main advantages compared to the competing meth-
ods. First, It uses atrous convolution which is a powerful
tool in dense prediction tasks. Atrous convolution allows to
explicitly control the resolution at which feature responses
are computed within deep CNNs. Second, atrous spatial
pyramid pooling (ASPP) allows to robustly segment ob-
jects at multiple scales. ASPP probes an incoming convo-
lutional feature layer with ﬁlters at multiple sampling rates
and effective ﬁelds-of-views, thus capturing objects as well
as image context at multiple scales. Third, the localiza-
tion of object boundaries is improved by combining meth-
ods from deep CNNs and probabilistic graphical models.
The commonly deployed combination of max-pooling and
downsampling in deep CNNs achieves invariance, but has
a toll on localization accuracy. The method overcomes this
by combining the responses at the ﬁnal DCNN layer with a
fully connected Conditional Random Field (CRF) which is
shown both qualitatively and quantitatively to improve lo-
calization performance.
To make sure that the pelage pattern is fully covered in
the seal segment, two additional postprocessing steps are
applied to segmentation maps to close holes and to smooth
the borders. In order to close the holes in the pattern (both
internal and external) we apply sliding window convex hull
on a condition that pattern in a current window is not con-
nected. The image is broken down into many small heavily
overlapping square windows. Each is checked for connec-
tivity, and if some part of the current window is not con-
nected then we paint it over with convex hull. This way, the
overall smooth structure of the seal outline remains undis-
turbed, but deep external holes get closed. This allows us
to preserve a general concave outline of a seal, while clos-
ing potential open holes in the pattern. Smoothing of the
image is a simple two-pass Gaussian ﬁlter with threshold-
ing to keep the mask binary. Smoothing gets rid of blocky
artifacts produced by our hole closing algorithm.
3.2. Pelage pattern extraction
Images of the Saimaa ringed seals are very diverse. They
are taken from various distances and viewing angles. The
Saimaa ringed seals are characterized by low mobility, but
large variability of poses. The fur pattern covers the entire
body of the animal and does not have any speciﬁc location.
In addition, due to the fact that the images were obtained
from the camera traps in different weather conditions and
with different lighting, there is a signiﬁcant amount of ex-
cess noise.
In order to remove noise, to avoid learning of the su-
perﬁcial characteristics, and to reduce the amount of data
needed to train the identiﬁcation algorithm, the pelage pat-
tern is extracted from the segmented seal images. The pat-
tern extraction algorithm is based largely on the tubeness
ﬁlter with processing steps to increase ﬁdelity and consists
of the following steps:
1. Sato tubeness ﬁlter. This ﬁlter can be used to detect
continuous ridges (tubes, wrinkles, rivers, etc.). It is
well-suited for the Saimaa ringed seals pattern extrac-
tion since their patterns are mostly continuous ridges
that form rings and other shapes.
2. Unsharping using a mask with a radius of 5 and an in-
tensity of 25. This operation makes the results sharper.
3. Removing segmentation border which heavily inﬂu-
ences the Sato ﬁlter. This step is necessary since the
segmentation border is detected as a ”tube”, but it does
not belong to the pattern.
4. Morphological opening using a disk structuring ele-
ment with a radius of 3. This operation allows us to
remove small artifacts from the grayscale image.
5. Adaptive histogram normalization This operation is
performed in order to make the image brighter with-
out losing details.
6. Otsu’s thresholding and zeroing out pixels below it.
This makes pattern edges well-deﬁned while still keep-
ing pattern smooth.
7. Morphological opening using a disk structuring ele-
ment with a radius of 3. This operation is also re-
peated, and this time it removes artifacts left from
thresholding.
28


## Page 5

8. Unsharping using a mask with a radius of 5 and an
intensity of 2. This unsharping mask is weaker than
before. It only needs to slightly sharpen the image after
opening in order to keep the pattern well-deﬁned and
well-contrasted with black areas.
The result of pattern extraction is a grayscale image with
an explicit pattern outline as shown in Fig. 3.
Figure 3. Visualization of pattern extraction result. First row:
Steps 1–4 of the algorithm (from the left to the right). Second
row: Steps 5–9 of the algorithm (from the left to the right). Third
row: the source image (left) and the end result of pattern extraction
(right).
3.3. Patch matching
Another noticeable problem with the Saimaa ringed seal
re-identiﬁcation is that the visible part of the pelage pattern
varies greatly between different images of the same seal.
Therefore, the system should be robust to the seal pose and
angle of viewing. To enable this, the pattern image is di-
vided into patches that are then used to ﬁnd correspond-
ing patches in the known individuals. Dividing the pattern
into patches also helps to keep the size of the network used
for matching compact. Before the patch extraction, the pat-
tern segment is cropped and scaled to the common size to
make scale between images similar relative to the size of the
seal. Then overlapping patches with the common size (in
our case 160 × 160 pixels with 50% overlap) are extracted.
Finally, the pattern patches with less than 10% non-black
pixels are removed.
Triplet Neural Network [12] is used to calculate the sim-
ilarity between two patches. The network itself consists of 2
halves: a convolutional part and a fully-connected part. We
augment the network with rotation-invariance pass. Each
image is rotated using a set of predeﬁned angles (we use
-30, -20, -10, 0, 10, 20, 30 degrees). Then each rotated ver-
sion of the same image is passed through the convolutional
part of the network, and the results are summed together
before being passed to the fully connected part.
Let g be the convolutional part of the network, f is fully
connected part, Θ is the set of predeﬁned rotation angles
and xθ is the image rotated by the angle θ then the result of
rotation-invariance pass can be described as such:
y(x) = f (
∑
θ∈Θ
g(xθ)) (2)
For triplet networks, the training and the evaluation pro-
cesses differ considerably. During the training, the network
receives three samples at a time: anchor, positive, and nega-
tive. The anchor is a base image (pattern patch), the positive
is a sample of the same class (patch from the same pattern
of the same seal) as the anchor, and the negative is a sam-
ple of a different class (patch from a different seal) from
the anchor. The objective of the network is to encode these
samples in a way that the L2 metric distance between the
anchor and the positive is smaller (by a pre-deﬁned mar-
gin) than the distance between the anchor and the negative
(Fig. 4).
Figure 4. Triplet Network training for patch matching.
When the trained network is applied to the matching task
it takes a sample (pattern patch) as the input and produces
an encoding vector (feature vector). Our encoding vector
consists of 512 dimensions which is the same number as
the number of outputs of the CNN. These encodings can be
compared using L2 metric which makes it straightforward
and fast to compute distances between them. The patch cor-
respondences can be found by comparing the encodings of
patches from the query image to all the patches from the la-
beled images in the database. Those correspondences pro-
vide the basis for the re-identiﬁcation. It should be noted
that the encoding vectors need to be computed only once
for each patch enabling efﬁcient computation.
3.4. Individual re­identiﬁcation
The goal of the individual re-identiﬁcation algorithm is
to predict a seal identiﬁer (unique for each individual ani-
mal) given a query image and a gallery of known individ-
uals. The gallery contains a small number of distinct, high
quality pattern images of each seal captured from different
sides in order for us to be able to reliably perform compar-
isons with query images. It is used to construct a pattern
29


## Page 6

patch database which consists of the encoding vectors and
the seal identiﬁers for each patch.
The proposed re-identiﬁcation algorithm ranks the simi-
larity between two images (the query image and a gallery
image) and can be divided into the three main steps: 1)
patch-similarity heatmap generation to select candidates of
corresponding patches in the gallery image, 2) candidate ﬁl-
tering using topology-preserving projections, and 3) candi-
date ranking. The patch similarity heatmaps are generated
by dividing both images into patches and comparing each
query image with all gallery image patches. Local minima
of the heatmap (high similarity regions) are found and used
as projection candidates.
Candidate ﬁltering is performed by selecting the projec-
tion candidates that preserve topological relations between
original patches. We use a simple angle-based method to
calculate the topological consistency in the candidate pro-
jection. The algorithm calculates angles for each three con-
secutive projection points and compares them to the same
angles between patch center points. The total angle differ-
ence is the rank (lower is better) of the topological similar-
ity. Finally, the ranking is obtained by calculating the av-
erage weight of topologically similar projections and by se-
lecting the one with the lowest average weight. This weight
is the total rank of similarity between the query image and
the gallery image (see Fig. 5).
Figure 5. Examples of the region similarity heatmaps: the query
image (left) and gallery image (right). Heatmaps for a query im-
age highlight a single region that is being compared to the entire
gallery image. Heatmaps for the gallery image show regions which
are most similar to the highlighted region from the query image.
4. Experiments
The experiments were performed using a challenging
database of the Saimaa ringed seals collected using auto-
matic camera traps (see Fig. 6).
4.1. Segmentation
4.1.1 Data
A large annotated dataset is needed to train the Deeplab
model. Annotating the dataset of this size manually is very
labor-intensive, so a heuristical, semi-automatic approach
Figure 6. Examples of the seal images.
to generate segmentation ground truth was utilized. First, a
pretrained model able to segment similar animals was ap-
plied for the dataset of more than 308 846 seal images to
obtain a subset of well-segmented seals. The results were
checked ﬁrst automatically by discarding fully black im-
ages, the segmentation results with low pixel density, and
the segmentation results which took up more than 50% of
the image. Then, the rest were ﬁltered manually to re-
move incorrectly segmented images. In addition, 100 im-
ages were manually segmented to complement the dataset
with those images that were difﬁcult to segment for the
pretrained model. The resulting dataset contained nearly
100 000 well-segmented, high-deﬁnition images with bi-
nary segmentation masks. The dataset was split as follows:
56 000 images for training, 21 000 images for validation,
and 22 800 images for testing.
4.1.2 Results
The Deeplab model was trained using the Tensorﬂow deep
learning framework [1]. A model pretrained on the Pascal
VOC dataset [9] was used and transfer learning was applied
changing the last network level to separate only two classes:
the seal and the background. Examples of the segmentation
results are shown in Fig. 7.
Figure 7. Examples of segmentation results.
Intersection over Union (IoU) between the segmentation
result and the ground truth was used as a metric to evaluate
the results. The mean IoU over all images in the test set
30


## Page 7

was 82% without postprocessing the segments. With post-
processing the mean IoU of 91% was achieved. Fig. 8 illus-
trates the IoU distribution of the segmented images before
and after postprocessing. More than 75% of images have
IoU more than 90% and 20% have more than 95%.
50%-100% 60%-100% 70%-100% 80%-100% 90%-100%
0%
20%
40%
60%
80%
100%
Raw
Processed
Figure 8. The impact of postprocessing on segmentation quality.
The horizontal axis represents the IoU bucket while the vertical
axis is the percentage of images that fell into said IoU bucket after
segmentation with and without postprocessing.
4.2. Patch matching
4.2.1 Data
In order to train the triplet network for patch matching, a
dataset of 3000 different labeled patches belonging to 26
different classes was collected. Each class corresponds to
one manually selected location in the pelage pattern of one
seal, and each sample from one class was extracted from
different images of the same seal. Thus, the class is deﬁned
here as the spatially matching corresponding patches in the
images of the considered individual seal, not as the class of
the individual seal. The dataset was further augmented by
random rotations, scaling, and shifts (Fig. 9).
Figure 9. Examples of patches. Original patches (top two rows)
and the corresponding pattern patches (bottom two rows).
A different, unrelated set of patches was used to test the
method. The dataset of 1500 patches belonging to 28 differ-
ent classes was collected. None of the classes in the testing
dataset was encountered during the training phase. A patch
size of 160 pixels was used.
4.2.2 Results
To demonstrate the usefulness of the proposed pelage pat-
tern method, the triplet network was tested on both the orig-
inal patches extracted from the segmented images and the
processed patches extracted from pattern images. The re-
sults are presented in Table 1.
Table 1. Patch matching results.
Top-1 Top-2 Top-3 Top-4 Top-5
Pattern
patches
74.6% 79.2% 81.1% 84.5% 87.0%
Original
patches
66.4% 73.0% 76.9% 80.5% 82.5%
The results indicate that the pattern extraction clearly im-
proves the results. This can be explained by the noise and
large variation in appearance such as lighting conditions,
pattern visibility, and shadows that are unavoidable on non-
processed images. This noise prevents the network from
generalizing well during the training, and makes it harder
to extract relevant features during the evaluation.
Figure 10. Examples of patch comparison: query patch (left) and
best matches in the descending order of distance (right). The ﬁrst
example illustrates the case of robustness to the pattern deforma-
tion, the second example shows matching in the case of lost details,
and the third shows invariance to small rotations.
Fig. 10 shows examples of patches being compared
well despite differences in rotation thanks to the rotation-
invariance pass of the network. Patches on the right exhibit
different angles of rotation compared to the query patch,
and yet they show up in the top-5 comparison.
4.3. Re­identiﬁcation
4.3.1 Data
The dataset to evaluate the full re-identiﬁcation framework
consisted of 2 000 images of 46 unique seal individuals.
31


## Page 8

Each image contained one Saimaa ringed seal manually
identiﬁed by a biologist. The dataset was split into the
gallery set (the database of the known seals) and the query
set (the test set) as follows: 500 images in the gallery set (on
average 10 images per individual, with minimum of 3 and
maximum of 12), and the rest (1500 images) in the query
set (between 5 and 120 images per individual).
4.3.2 Results
The re-identiﬁcation framework was tested with several
variables. The ﬁrst is pattern extraction. We performed tests
with and without pattern extraction applied in order to quan-
tify its effect on re-identiﬁcation performance and training
quality. The second is network rotation invariance. Our pro-
posed rotation invariance pass was tested against the plain
network in order to see how useful it is in the real-world im-
ages outside of synthetic tests. Finally, we implemented two
different image re-identiﬁcation strategies based on patch
comparison. The ﬁrst strategy is a simple ”patch voting”
based on the k nearest neighbors (KNN) classiﬁcation. In
this algorithm, each patch is classiﬁed with KNN as belong-
ing to one of individual seals from the gallery set. Then
each patch ”votes” for the seal it belongs to, and the votes
are weighted according to conﬁdence metric. The second
strategy is the proposed heatmap-based topologically-aware
patch matching. The results are shown in Table 2.
Table 2. Re-identiﬁcation results with different method variables.
OG stands for original images, PA T for pattern extraction, ROT
and noROT signify rotation invariance, KNN is KNN-based patch
voting and TOP is topologically aware heatmaps
Top-1 Top-2 Top-3 Top-4 Top-5
OG-noROT-KNN 50.3% 56.2% 60.9% 63.4% 65.6%
PA T-noROT-KNN58.1% 64.6% 69.3% 74.0% 76.9%
PA T-noROT-TOP 62.7% 68.3% 72.1% 76.9% 83.7%
PA T-ROT-KNN 64.9% 70.5% 75.0% 80.1% 82.5%
PA T-ROT-TOP 67.8% 73.2% 77.2% 81.7% 88.6%
The results demonstrate that the pattern extraction makes
a signiﬁcant difference and increase the performance of
the whole re-identiﬁcation. Moreover, KNN-based patch
voting is worse than topologically aware heatmaps at re-
identiﬁcation, especially with non-perfect patch comparison
neural network. Finally, rotation invariance gives a small
boost in performance and helps to alleviate some of the ir-
regularities in the dataset.
Currently the task of identifying each seal is performed
manually by experts which takes considerable time and ef-
fort. Because of this, the Top-1 accuracy is not the only im-
portant metric for the re-identiﬁcation system. Limiting the
choice to a set of best matches during manual identiﬁcation
is going to speed up the process signiﬁcantly. High Top-
5 accuracy can help experts with identiﬁcation tasks while
still leaving them with a great degree of manual control. The
method can adequately present good potential matches from
which an expert can make an accurate conclusion much
faster. Fig. 12 shows examples of re-identiﬁcation results.
It is evident that the proposed method is capable of handling
complex cases when the pattern is only partially similar or
rotated.
Figure 11. Examples of re-identiﬁcation results: query images
(left) and top 4 matches (right). Correct matches (same individ-
ual) are highlighted with green and incorrect ones with red.
5. Conclusions
In this paper, we proposed a framework to re-identify
the Saimaa ringed seal individuals from camera-trap im-
ages for monitoring and conservation purposes. The frame-
work consists of seal segmentation using the state-of-the-
art Deeplab model, Sato tubeness ﬁlter based pelage pat-
tern extraction method, the siamese network based pattern
patch matching for ﬁnding patch correspondences, and a
re-identiﬁcation algorithm based on the patch similarities
and topology-preserving projections. Our results show that
the framework produces promising re-identiﬁcation results,
taking step towards fully automated re-identiﬁcation system
for the Saimaa ringed seals.The method provides a useful
tool for conservation biologist by reducing the amount of
manual work. The framework is species-agnostic and by re-
placing the pattern extraction step it can be applied to other
animal species with similar pelage or fur patterns.
Acknowledgments
The research was carried out in the CoExist project
(Project ID: KS1549) funded by the European Union, the
Russian Federation and the Republic of Finland via The
South-East Finland – Russia CBC 2014-2020 programme.
The authors would like to thank Vincent Biard, Marja
Niemi, and Mervi Kunnasranta from Department of Envi-
ronmental and Biological Sciences at University of Eastern
Finland for providing the data for the experiments and ex-
pert knowledge for identifying the individuals.
References
[1] M. Abadi, A. Agarwal, P . Barham, E. Brevdo, Z. Chen,
C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghe-
32


## Page 9

mawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y . Jia,
R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man ´e,
R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster,
J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P . Tucker,
V . V anhoucke, V . V asudevan, F. Vi´egas, O. Vinyals, P . War-
den, M. Wattenberg, M. Wicke, Y . Y u, and X. Zheng. Tensor-
Flow: Large-scale machine learning on heterogeneous sys-
tems, 2015. Software available from tensorﬂow.org.
[2] L. Bergamini, A. Porrello, A. C. Dondona, E. Del Negro,
M. Mattioli, N. D’alterio, and S. Calderara. Multi-views em-
bedding for cattle re-identiﬁcation. In 2018 14th Interna-
tional Conference on Signal-Image Technology & Internet-
Based Systems (SITIS), pages 184–191. IEEE, 2018.
[3] T. Y . Berger-Wolf, D. I. Rubenstein, C. V . Stewart, J. A.
Holmberg, J. Parham, S. Menon, J. Crall, J. V an Oast,
E. Kiciman, and L. Joppa. Wildbook: Crowdsourcing,
computer vision, and data science for conservation. arXiv
preprint arXiv:1710.08880, 2017.
[4] C.-A. Brust, T. Burghardt, M. Groenenberg, C. Kading, H. S.
Kuhl, M. L. Manguette, and J. Denzler. Towards automated
visual monitoring of individual gorillas in the wild. In Pro-
ceedings of the IEEE International Conference on Computer
Vision, pages 2820–2830, 2017.
[5] T. Chehrsimin, T. Eerola, M. Koivuniemi, M. Auttila,
R. Lev ¨anen, M. Niemi, M. Kunnasranta, and H. K ¨alvi¨ainen.
Automatic individual identiﬁcation of saimaa ringed seals.
IET Computer Vision, 12(2):146–152, 2018.
[6] L.-C. Chen, Y . Zhu, G. Papandreou, F. Schroff, and H. Adam.
Encoder-decoder with atrous separable convolution for se-
mantic image segmentation. arXiv:1802.02611, 2018.
[7] J. Crall, C. Stewart, T. Berger-Wolf, D. Rubenstein, and
S. Sundaresan. Hotspotter - patterned species instance recog-
nition. IEEE Workshop on Applications of Computer Vision
(WACV), pages 230–237, 2013.
[8] D. Deb, S. Wiper, S. Gong, Y . Shi, C. Tymoszek, A. Fletcher,
and A. K. Jain. Face recognition: Primates in the wild. In
2018 IEEE 9th International Conference on Biometrics The-
ory, Applications and Systems (BTAS) , pages 1–10. IEEE,
2018.
[9] M. Everingham, L. V an Gool, C. K. I. Williams, J. Winn,
and A. Zisserman. The PASCAL Visual Object Classes
Challenge 2012 (VOC2012) Results. http://www.pascal-
network.org/challenges/VOC/voc2012/workshop/index.html,
2012.
[10] K. M. Halloran, J. D. Murdoch, and M. S. Becker. Apply-
ing computer-aided photo-identiﬁcation to messy datasets:
a case study of thornicroft’s giraffe (giraffa camelopardalis
thornicrofti). African Journal of Ecology , 53(2):147–155,
2015.
[11] A. Hermans, L. Beyer, and B. Leibe. In defense of the
triplet loss for person re-identiﬁcation. arXiv preprint
arXiv:1703.07737, 2017.
[12] E. Hoffer and N. Ailon. Deep metric learning using triplet
network. In International Workshop on Similarity-Based
Pattern Recognition, pages 84–92. Springer, 2015.
[13] G. Koch, R. Zemel, and R. Salakhutdinov. Siamese neu-
ral networks for one-shot image recognition. In ICML deep
learning workshop, volume 2, 2015.
[14] M. Koivuniemi, M. Kurkilahti, M. Niemi, M. Auttila, and
M. Kunnasranta. A mark–recapture approach for estimating
population size of the endangered ringed seal (phoca hispida
saimensis). PloS one, 14(3):e0214269, 2019.
[15] S. Li, J. Li, W. Lin, and H. Tang. Amur tiger re-identiﬁcation
in the wild. arXiv preprint arXiv:1906.05586, 2019.
[16] W. Li, R. Zhao, T. Xiao, and X. Wang. Deepreid: Deep
ﬁlter pairing neural network for person re-identiﬁcation. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 152–159, 2014.
[17] C. Liu, R. Zhang, and L. Guo. Part-pose guided amur tiger
re-identiﬁcation. In The IEEE International Conference on
Computer Vision (ICCV) Workshops, Oct 2019.
[18] N. Liu, Q. Zhao, N. Zhang, X. Cheng, and J. Zhu. Pose-
guided complementary features learning for amur tiger re-
identiﬁcation. In The IEEE International Conference on
Computer Vision (ICCV) Workshops, Oct 2019.
[19] M. Matth ´e, M. Sannolo, K. Winiarski, A. Spitzen-van der
Sluijs, D. Goedbloed, S. Steinfartz, and U. Stachow. Com-
parison of photo-matching algorithms commonly used for
photographic capture–recapture studies. Ecology and evo-
lution, 7(15):5861–5872, 2017.
[20] O. Moskvyak, F. Maire, A. O. Armstrong, F. Dayoub, and
M. Baktashmotlagh. Robust re-identiﬁcation of manta rays
from natural markings by learning pose invariant embed-
dings. arXiv preprint arXiv:1902.10847, 2019.
[21] E. Nepovinnykh, T. Eerola, H. K ¨alvi¨ainen, and G. Rad-
chenko. Identiﬁcation of saimaa ringed seal individuals us-
ing transfer learning. In International Conference on Ad-
vanced Concepts for Intelligent Vision Systems , pages 211–
222. Springer, 2018.
[22] M. S. Norouzzadeh, A. Nguyen, M. Kosmala, A. Swanson,
M. S. Palmer, C. Packer, and J. Clune. Automatically iden-
tifying, counting, and describing wild animals in camera-
trap images with deep learning. Proceedings of the National
Academy of Sciences, 115(25):E5716–E5725, 2018.
[23] Y . Sato, S. Nakajima, N. Shiraga, H. Atsumi, S. Y oshida,
T. Koller, G. Gerig, and R. Kikinis. Three-dimensional
multi-scale line ﬁlter for segmentation and visualization of
curvilinear structures in medical images. Medical image
analysis, 2(2):143–168, 1998.
[24] S. Schneider, G. W. Taylor, S. Linquist, and S. C. Kre-
mer. Similarity learning networks for animal individual re-
identiﬁcation-beyond the capabilities of a human observer.
arXiv preprint arXiv:1902.09324, 2019.
[25] J. Wang, Y . Song, T. Leung, C. Rosenberg, J. Wang,
J. Philbin, B. Chen, and Y . Wu. Learning ﬁne-grained image
similarity with deep ranking. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition ,
pages 1386–1393, 2014.
[26] X. Y u, J. Wang, R. Kays, P . Jansen, T. Wang, and T. Huang.
Automated identiﬁcation of animal species in camera trap
images. EURASIP Journal on Image and Video Processing ,
2013(1):52, 2013.
[27] A. Zhelezniakov, T. Eerola, M. Koivuniemi, M. Auttila,
R. Lev ¨anen, M. Niemi, M. Kunnasranta, and H. K ¨alvi¨ainen.
Segmentation of saimaa ringed seals for identiﬁcation pur-
33


## Page 10

poses. In Proceedings of International Symposium on Visual
Computing, pages 227–236, Las V egas, USA, 2015.
34
