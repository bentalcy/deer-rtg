# patterned_animals_reid_2308.06335


## Page 1

Workshop Camera Traps, AI and Ecology
Brief Paper
Combining feature aggregation and
geometric similarity for re-identification of
patterned animals
Veikka Immonen1 Ekaterina Nepovinnykh1,∗ Tuomas Eerola1 Charles V. Stewart2 Heikki Kälviäinen1
1 Computer Vision and Pattern Recognition Laboratory, Department of Computational Engineering, School of Engineering Sciences,
Lappeenranta-Lahti University of Technology LUT, FI-53851 Lappeenranta, Finland
2 Department of Computer Science, Rensselaer Polytechnic Institute, Troy, NY 12180, USA
* E-mail: ekaterina.nepovinnykh@lut.fi
Abstract: Image-based re-identification of animal individuals allows gathering of information such as migration patterns of the ani-
mals over time. This, together with large image volumes collected using camera traps and crowdsourcing, opens novel possibilities
to study animal populations. For many species, the re-identification can be done by analyzing the permanent fur, feather, or skin
patterns that are unique to each individual. In this paper, we address the re-identification by combining two types of pattern simi-
larity metrics: 1) pattern appearance similarity obtained by pattern feature aggregation and 2) geometric pattern similarity obtained
by analyzing the geometric consistency of pattern similarities. The proposed combination allows to efficiently utilize both the local
and global pattern features, providing a general re-identification approach that can be applied to a wide variety of different pattern
types. In the experimental part of the work, we demonstrate that the method achieves promising re-identification accuracies for
Saimaa ringed seals and whale sharks.
1 Introduction
Automatic camera traps allow the collection of large volumes of
wildlife images in a non-invasive way. To fully utilize this data in
the research on animal populations, the analysis of the images needs
to be automated. The essential image analysis problem to be solved
is the re-identification of individual animals as it allows us to obtain,
e.g., information about the behavior and migration patterns, as well
as, estimate the population size through capture and recapture anal-
ysis. The re-identification can be done by utilizing permanent visual
traits such as fur pattern, scarring, or fin shape.
Re-identification has already been applied to study various ani-
mal species. For example, for Saimaa ringed seals ( Pusa hispida
saimensis), an endangered species native to Lake Saimaa in Finland,
image-based re-identification has been applied for conservation pur-
poses to study animal migration and behavior [12, 13, 15]. Recently,
many automated Saimaa ringed seal re-identification methods uti-
lizing the permanent ring pattern have been developed [23, 24, 27].
Similarly, population monitoring efforts of whale sharks (rhincodon
typus) have been tackled using re-identification and capture and
recapture models [31], and automatic methods for re-identification
have been proposed [1, 11]. These methods utilize the spot pat-
terns that are unique to each individual. Many of the existing
re-identification methods are specific to certain animal species or
types of patterns, limiting the breadth of this their usability. General
methods that can be applied to a wide variety of different animal
species would be preferable as they could be easily adapted for
various animal population studies.
In this paper, species-agnostic re-identification is addressed by
proposing a method that combines both local and global similari-
ties of the fur pattern (see Fig. 1). Local similarities are obtained
by aggregating CNN-based local pattern features over a processed
image using Fisher vectors. This results in compact vector presen-
tations of the pattern appearance for the query images enabling
efficient similarity comparison to the corresponding Fisher vec-
tors computed for images in the database of known individuals.
Fisher vector based aggregation does not, however, take into account
the geometric consistency of the pattern similarity. Therefore, the
similarity of the aggregated local pattern appearance is further com-
plemented with a geometric similarity providing the global context
for the pattern analysis. The geometric similarity is obtained by
Fig. 1: The proposed method combines aggregated pattern features
and geometric consistency of the point correspondences.
searching the local pattern feature correspondences between the
query and database images and analyzing how consistent they are
with the geometric transformation estimated using RANSAC.
In the experimental part of the work, we demonstrate the accuracy
of the proposed method on two species with very different types
of patterns: Saimaa ringed seals and whale sharks. We show that
the proposed method outperforms two earlier species-agnostic meth-
ods: HotSpotter [4] and NORPPA [24]. Furthermore, we show that
the combination of aggregated pattern feature-based similarity and
geometric similarity provides higher re-identification accuracy than
either of the similarity metrics alone. One notable benefit of the pro-
posed pattern feature extraction method is that it does not require
species-specific training but the same pre-trained keypoint detection
and feature descriptors can be used for both species, making it a
promising approach for generic animal individual re-identification.
Camera Traps, AI, and Ecology
© Copyright resides with the authors 1
arXiv:2308.06335v1  [cs.CV]  11 Aug 2023


## Page 2

2 Related work
Various animal species can be re-identified by different types of visu-
ally unique biological traits such as fur pattern, face, or fin shape.
Algorithmically, the methods can be divided into classification and
metric-based approaches [34]. The classification-based approaches
assume that the database of known individuals is known and finite.
The metric-based methods, on the other hand, aim to learn a simi-
larity metric between the input images. The re-identification is then
performed by matching based on the similarity which means that
metric-based approaches are not limited by the initial database and
can be applied to new individuals without retraining. A variety of
methods for animal re-identification exist. They have been success-
fully applied, for example, to Amur tigers (stripe pattern) [16], cattle
(muzzle shape) [14], giraffes (spot pattern) [18], humpback whales
(fluke shape) [35] and primates (face) [5].
A number of methods for the re-identification of Saimaa ringed
seals have been proposed [2, 3, 22–24, 27, 36]. Saimaa ringed seal
is especially challenging species for re-identification due to the fol-
lowing matters: (i) the large variation in possible poses which is
further exacerbated by the deformable nature of the seals, (ii) the
non-uniform pelage patterns, limiting the size of the regions that can
be used for the re-identification task, (iii) the low contrast between
the ring pattern and the rest of the pelage, and (iv) the extreme dataset
bias. These challenges have been addressed by proposing various
approaches to preprocess the images and to encode the pattern fea-
tures [3, 23, 24, 36]. The most successful methods employ the pattern
extraction step [23, 24] to construct a binary representation of the
pelage pattern and metric learning-based pattern encoding.
Individual whale sharks can be identified based on the spot pattern
on their skin. In [1], a blob detection was applied to find the individ-
ual spots, and a pattern-matching algorithm [6] originally developed
for astronomical images (star patterns) were used to compare the pat-
terns. In [11], a U-Net-based model was utilized for spot detection
and a metric learning-based approach generated pattern embeddings
for the re-identification of individuals.
While the majority of the existing methods are specific to species,
there has been also efforts towards species-agnostic re-identification
methods that can be applied to a wide variety of different type visual
traits. HotSpotter [4] is a SIFT-based algorithm that uses viewpoint
invariant descriptors and a scoring mechanism which emphasizes the
most distinctive key points, called “hot spots,” on an animal pattern.
PIE [21] is a deep learning-based method that receives shape embed-
ding and pose embedding separately and normalizes the shape to
match the individual regardless of the specific pose. In [32], vari-
ous similarity learning architectures are compared on chimpanzees,
humpback whales, fruit flies, and Siberian tigers.
3 Proposed method
The proposed method builds on the NORPPA method [24] devel-
oped for Saimaa ringed seals. We generalize the method to other
patterned animal species by extending the pattern similarity method
to address both the similarities in local appearance and the global
geometric consistency of the patterns. The method starts with the
segmentation of the animal from the background, after which the
pattern is extracted for further analysis. Regions of interest (patches)
are detected from the pattern images and the pattern image patches
are embedded. To measure the similarity of the local pattern appear-
ance, embeddings are aggregated to a single Fisher vector of a fixed
length. These vectors can be used to quantify the similarity of two
patterns using the cosine distance. Since Fisher vectors do not take
into account the global spatial structure of the pattern further geo-
metric similarity is assessed by analyzing the consistency matched
point correspondences to the homography found using RANSAC.
Finally, the two similarity metrics are combined and the most simi-
lar pattern is searched from the database of known individuals. The
whole pipeline is visualized in Fig. 2.
3.1 Preprocessing and segmentation
Preprocessing of the data consists of two steps: tone mapping and
segmentation. Images collected using camera traps and crowdsourc-
ing contain a large variation in the quality due to challenging illumi-
nation, weather conditions, and suboptimal quality of cameras. To
address this, we utilize tone mapping to balance the contrast levels,
making it easier to segment the animal and extract the pelage pat-
tern. For tone mapping, we utilize the method by Mantiuk et al. [17].
Various animals exhibit strong site-fidelity, meaning that they tend
to stay in the same regions. This together with the static camera
traps can lead to dataset bias issues as individuals are often cap-
tured with the same background increasing the risk that a supervised
recognition algorithm learns to identify the background instead of
the animal. This is why segmentation of the animal is recommended.
For segmentation, we utilize Mask R-CNN [7].
3.2 Aggregated local pattern appearance
To analyze the local pattern appearances, local features are extracted
and aggregated into comparable feature embeddings. First, the pat-
tern is extracted from the segmented images with a pre-trained
U-Net-based encoder-decoder model to emphasize the pattern
and discard irrelevant information. Next, affine covariant regions
are found and extracted from the pattern image by CNN-based
HesAffNet [20]. HesAffNet extracts local regions from images and
transforms them according to the estimated local affine transforma-
tion, generating affine-invariant patches of fur patterns.
The extracted patches are embedded into vectors of size 1 ×128
by using HardNet [19]. HardNet is trained to correctly match cor-
responding descriptors while avoiding false positives from similar
descriptors by using the triplet margin loss. After the HardNet
embedding, PCA is applied to the features to decorrelate them and
to reduce dimensionality.
Feature embeddings are aggregated using Fisher vectors [10,
28, 29]. A visual vocabulary is constructed by applying Gaussian
Mixture Model (GMM) to the features from the database. Then,
Fisher vectors are created for each image by computing the partial
derivatives of the log-likelihood function with respect to the GMM
parameters and concatenating them. Kernel PCA [33] is applied to
further reduce the dimensionality of the resulting image descriptors
which helps to reduce the storage requirements for the database, as
well as speed up the database search for the re-identification.
Finally, (dis)similarities between the Fisher vector for the query
image and the database images are calculated using the cosine
distance:
dL = 1 − Φq · Φdb
||Φq||2||Φdb||2
(1)
where Φq is the Fisher vector for query image and Φdb the Fisher
vector for a database image. This distance quantifies the dissimilarity
of the aggregated local pattern appearances between the images. For
more detailed description of the pattern feature aggregation, see [24].
3.3 Geometric similarity of patterns
The aggregated local pattern appearance does not take into account
the global spatial structure of the pattern. To further incorporate this
information to the pattern matching, the geometric consistency of the
local similarities are analyzed. This is done using a similar method
as the spatial reranking step of the HotSpotter algorithm [4] and
the object retrieval method proposed in [30]. The HardNet embed-
dings of the HesAffNet interest points are matched to find the patch
correspondences between query and database images. The matching
is done by computing cosine distances between the embeddings of
individual patch pairs.
The coordinates of the patch correspondences are then normal-
ized to have the zero mean and the maximum distance of 1 to the
origin. Outliers (and inliers) are detected by estimating the pro-
jective homography between the query image and database image
using RANSAC. The assumption is that if the patterns do not match,
Camera Traps, AI, and Ecology
2 © Copyright resides with the authors


## Page 3

Mask
R-CNN U-net
Interest points
Original image Segmentation Pattern extraction
Database
...............
Geometric similarity
Combined similarity
Feature aggregationCosine similarity
Aggregation local appearance similarity
HesAffNet
descriptors
RANSAC inliers Feature matching
+
DescriptorsFisher vector
HesAffNet
Fig. 2: The pipeline of the proposed method.
the inconsistency in the global arrangements of patch correspon-
dences causes a low amount of inliers. Therefore, the amount ( n)
and relative proportion (ω) of inliers are good metrics for geometric
similarity of patterns. It should be noted that due to the large pose
variation of animals, it is recommended to have a high inlier thresh-
old to ensure successful outlier detection in the case of matching
patterns.
3.4 Combined similarity
The final re-identification of the animal individual in the query image
is performed by searching the most similar pattern from the database
of known individuals. To compute the dissimilarity (distance) a novel
combination of the dissimilarity of aggregated local pattern appear-
ance and geometric dissimilarity of patterns is used. We propose two
combination rules:
dC = dL(1 − ω)a (2)
and
dC = dn
L. (3)
In the first combination rule, the geometric similarity, defined as a
ratio of inliers to all image points ω, has a polynomial influence on
the cosine distance d between the Fisher vectors (aggregated local
pattern appearance). The greater the value of a (a ≥ 0) is, the more
influence geometric consistency has in the final results. In the second
combination rule, the geometric consistency, defined as a number
of inliers n, has an exponential influence on the cosine distance
(dL ≤ 1). If the amount of individuals in the database is large, re-
identification can be made more efficient by using the aggregated
Fisher vector for quick database searches, and using the geometric
similarity only as the verification step.
4 Experiments
The proposed method was evaluated with two different datasets: the
SealID dataset [26] consisting of Saimaa ringed seals and the Whale
shark dataset provided by Wild Me [8, 9]. Saimaa ringed seal pat-
terns consist of local arrangements of ring-like shapes. The regions
enabling the re-identification often constitute a rather small portion
of the whole pattern, making it important to have representative
image features for local appearance. Whale shark patterns, on the
other hand, consist of small spots with similar appearance, making it
more important to be able to quantify the geometric arrangement of
the spots.
4.1 Datasets
4.1.1 SealID: The Saimaa Ringed Seal re-identification dataset
(SealID) from [26] is used for the experiments. The dataset con-
sists of 2080 images of 57 known Saimaa ringed seal individuals.
The dataset is divided into two subsets: the database and the query.
The database subset (N = 430) consists of high-quality and unique
images that are enough to cover the full body pattern of each indi-
vidual seal in the dataset. The database has been constructed by
prioritizing the images with the best quality. All training and prepa-
rations required for NORPPA are done using only the images from
the database subset. The query subset ( N = 1650) contains the
remaining images of the same seal individuals and these images
are used in re-identification experiments. Sample images from both
subset are shown in Fig. 3. The image distribution is shown in Fig. 4
4.1.2 Whale shark dataset: In the experiments, we utilized
an extended version of the whale shark identification dataset pro-
vided by Wild Me [8, 9]. The original dataset includes images and
corresponding labels in the Microsoft COCO format. Fig. 5 show-
cases example images from this dataset. Each image in the dataset is
accompanied by a bounding box delineating the torso portion of the
Camera Traps, AI, and Ecology
© Copyright resides with the authors 3


## Page 4

Fig. 3: Sample images from both query and database images of the
SealID dataset.
Fig. 4 : Image distributions for the database and query sets. For
example, in the query dataset (right) 25 individuals (y-axis) had less
than 20 images (x-axis). [26]
Fig. 5: Sample images from the Whaleshark dataset.
Fig. 6: Image distribution for the Whaleshark dataset
whale shark’s body, an individual identification tag, and the view-
point of the animal (right or left). The dataset comprises a total
of 5409 annotated sightings, specifically pertaining to 235 distinct
whale shark viewpoints. The image distribution is shown in Fig. 6
4.2 Segmentation and pattern extraction
Saimaa ringed seals were segmented using Mask R-CNN with
ResNet-101 backbone combined with Feature Pyramid Network
trained on ringed seals. For more information about the segmentation
method, see [25]. For whale sharks, the segmentation step was omit-
ted and the bounding box annotations were used. The evaluation,
therefore, focused only on the re-identification. Pattern extraction for
both datasets was performed using the U-Net-based model which
was pretrained on Saimaa ringed seals. Since both patterns consist
of white or light gray patterns on a dark background the same model
was found to be reasonably accurate also on whale sharks as it can
be seen from Fig. 7. Furthermore, a U-Net-based spot segmentation
method from [11] specifically developed for whale sharks was tested
(see Fig. 7(f)-(h)).
(a)
 (b)
(c)
 (d)
 (e)
(f)
 (g)
 (h)
Fig. 7: Visualisation of the pattern extraction step for Saimaa ringed
seals: (a)-(b), whale sharks using the segmentation model trained on
ringed seals: (a)-(b), and whale sharks using the model from [11]:
(f)-(h).
4.3 Saimaa ringed seal re-identification
The re-identification results for the SealID dataset are presented in
Table 1 and Fig. 8. NORPPA [24] corresponds to using only similar-
ity of aggregated local pattern appearances, that is dC = dL. When
only geometric similarity is used the distance metric is defined by
the number of inliers, that is dC = −n. The RANSAC inlier thresh-
old was set to 0.1 allowing a maximum displacement corresponding
to 5% of the size of the pattern image. A relatively large thresh-
old value was used to take into account the large pose variation
between images. The proposed combined pattern similarity mea-
sure outperformed the competing methods in the top-1 accuracy.
The exponential combination rule provided the best re-identification
accuracy. The re-identification results are illustrated in Fig. 9. The
green lines correspond to the inliers found during the computation
of geometric similarity. The red lines correspond to ouliers. In some
rare cases (16 out of 1650 query images) the combined similarity
failed to re-identify the individual even though NORPPA identified
the individual correctly. This means that the RANSAC method found
a notable amount of inliers despite the point correspondences being
incorrect. Few of these cases are illustrated in Fig. 10.
Camera Traps, AI, and Ecology
4 © Copyright resides with the authors


## Page 5

Table 1 Re-identification results on the SealID dataset.
Method top-1 top-3 top-5
HotSpotter [4] 61.9% 63.6% 64.4%
NORPPA [24]: dC = dL 77.2% 82.4% 85.0%
Only geometric similarity: dC = −n 79.4% 83.0% 84.7%
Proposed method: dC = dL(1 − ω)2 79.6% 83.2% 85.0%
Proposed method: dC = dn
L 83.4% 86.1% 87.6%
0 2 4 6 8 10 12 14 16 18 2060
64
68
72
76
80
84
88
92
top-k
accuracy (%)
NORPPA [24]:dC =dL
Only geometric similarity:dC =−n
Proposed:dC =dL(1−ω)2
Proposed:dC =dnL
HotSpotter [4]
Fig. 8 : Top- k re-identification results relative to k for SealID.
For HotSpotter, the results were only analyzed through the top-5
accuracy.
Fig. 9 : Five examples of the successful Saimaa ringed seal re-
identification by combination dC = dn
L. The query image is at the
top and the matched image from the database is at the bottom.
Detected inliers are labeled in green and outliers are labeled in red.
4.4 Whale shark re-identification
The whale shark dataset was not divided into the database and query
subsets. Instead, whale shark re-identification was performed using
a leave-one-out strategy. The RANSAC inlier threshold was set to
0.05. A smaller threshold value was used than for Saimaa ringed
seals due to the smaller variation in pose and the need to find finer
Fig. 10 : Three examples of the unsuccessful re-identification by
combination dC = dn
L. The query image is at the top and the
matched image from the database is at the bottom. Detected inliers
are labeled in green and outliers are labeled in red.
dissimilarities on spot arrangements. A general rule of thumb when
adjusting the threshold for new species is that it can be stricter for
cases where the transformation between the patterns is more linear
or suitable to be described with a homography, which is the case for
the whalesharks. The results are shown in Table 2 and Fig. 11. The
successful re-identification results are visualized in Fig. 12. As it
can be seen the proposed method outperforms both HotSpotter and
NORPPA. While the accuracies are notably lower than for Saimaa
ringed seals, the results can be considered promising since the same
pattern extraction and feature embedding models as for the ringed
seals were used without retraining or fine-tuning. By simply replac-
ing the pattern extraction model with the whale shark-specific spot
segmentation method [11], the top-1 accuracy increased to 73%.
Fully omitting the pattern extraction step lowered the top-1 accuracy
to 59% showing that even suboptimal pattern extraction method still
improves the accuracy.
Table 2 Re-identification results on the whale shark dataset.
Method top-1 top-3 top-5
HotSpotter [4] 52% 53% 53%
NORPPA [24]: dC = dL 53% 62% 69%
Only geometric similarity: dC = −n 55% 67% 72%
Proposed: dC = dL(1 − ω)2 56% 68% 72%
Proposed: dC = dn
L 61% 69% 73%
0 2 4 6 8 10 12 14 16 18 20
52
56
60
64
68
72
76
80
84
top-k
accuracy (%)
NORPPA [24]:dC =dL
Only geometric similarity:dC =−n
Proposed:dC =dL(1−ω)2
Proposed:dC =dnL
HotSpotter [4]
Fig. 11: Top-k re-identification accuracy relative tok for the Whale
shark dataset.
Camera Traps, AI, and Ecology
© Copyright resides with the authors 5


## Page 6

Fig. 12: Examples of the successful whale shark re-identifications
using combination dC = dn
L.
5 Conclusion
In this paper, the re-identification of animal individuals using unique
fur and skin patterns has been considered. The proposed method
combines both similarity of local visual appearances of the pat-
tern aggregated over the full image, as well as the global geometric
consistency of the pattern similarities. This provides a versatile pat-
tern similarity metric that can be used for re-identification on a
wide variety of patterned animals. We demonstrated the method on
two species with notably different types of patterns: Saimaa ringed
seals and whale sharks. Promising results were obtained with the
combined similarity providing higher accuracy than for the individ-
ual pattern similarity metrics. One notable benefit of the proposed
method is that species-specific training is not necessarily needed. To
demonstrate this, the same pre-trained models were used on both
species for pattern extraction and pattern feature embedding making
the proposed method species-agnostic without additional training.
At same time, the models are trainable making it possible to fine-
tune the method for a specific species. This has potential to further
increase the re-identification accuracy.
6 Acknowledgments
The authors would like to thank Vincent Biard, Piia Mutka, Marja
Niemi, and Mervi Kunnasranta from the Department of Environ-
mental and Biological Sciences at the University of Eastern Finland
(UEF) for providing the data of Saimaa ringed seals and their expert
knowledge of identifying each individual. The authors would like
to thank Maksim Kholiavchenko from the Department of Computer
Science, Rensselaer Polytechnic Institute, for providing additional
insight into their method.
7 References
1 Zaven Arzoumanian, Jason Holmberg, and Brad Norman. An astronomical
pattern-matching algorithm for computer-aided identification of whale sharks
rhincodon typus. Journal of Applied Ecology, 42(6):999–1011, 2005.
2 Tina Chehrsimin, Tuomas Eerola, Meeri Koivuniemi, Miina Auttila, Riikka Lev-
änen, Marja Niemi, Mervi Kunnasranta, and Heikki Kälviäinen. Automatic
individual identification of Saimaa ringed seals. IET Computer Vision , 12(2):
146–152, 2018.
3 Ilia Chelak, Ekaterina Nepovinnykh, Tuomas Eerola, Heikki Kälviäinen, and Igor
Belykh. EDEN: deep feature distribution pooling for saimaa ringed seals pattern
matching. In International Conference Cyber-Physical Systems and Control, pages
141–150, 2021.
4 Jonathan P Crall, Charles V Stewart, Tanya Y Berger-Wolf, Daniel I Rubenstein,
and Siva R Sundaresan. Hotspotter—patterned species instance recognition. In
IEEE Workshop on Applications of Computer Vision , pages 230–237, 2013.
5 Debayan Deb, Susan Wiper, Sixue Gong, Yichun Shi, Cori Tymoszek, Alison
Fletcher, and Anil K Jain. Face recognition: Primates in the wild. In IEEE Inter-
national Conference on Biometrics Theory, Applications and Systems, pages 1–10,
2018.
6 Edward J Groth. A pattern-matching algorithm for two-dimensional coordinate
lists. Astronomical Journal, 91:1244–1248, 1986.
7 Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In
IEEE International Conference on Computer Vision , pages 2961–2969, 2017.
8 Jason Holmberg. Sharkbook: Wildbook for sharks, 2023. URL https://www.
sharkbook.ai/. accessed 2023-06-18.
9 Jason Holmberg, Bradley Norman, and Zaven Arzoumanian. Estimating popula-
tion size, structure, and residency time for whale sharks rhincodon typus through
collaborative photo-identification. Endangered Species Research , 7(1):39–53,
2009.
10 David Hutchison, Takeo Kanade, Josef Kittler, Jon M. Kleinberg, Friedemann Mat-
tern, John C. Mitchell, Moni Naor, Oscar Nierstrasz, C. Pandu Rangan, Bernhard
Steffen, Madhu Sudan, Demetri Terzopoulos, Doug Tygar, Moshe Y . Vardi, Ger-
hard Weikum, Florent Perronnin, Jorge Sánchez, and Thomas Mensink. Improving
the Fisher Kernel for Large-Scale Image Classification. In European Conference
on Computer Vision, 2010.
11 Maksim Kholiavchenko. Comprehensive deep learning pipeline for whale shark
recognition. Master’s thesis, Rensselaer Polytechnic Institute, 2022.
12 Meeri Koivuniemi, Miina Auttila, Marja Niemi, Riikka Levänen, and Mervi Kun-
nasranta. Photo-id as a tool for studying and monitoring the endangered Saimaa
ringed seal. Endangered Species Research, 30:29–36, 2016.
13 Meeri Koivuniemi, Mika Kurkilahti, Marja Niemi, Miina Auttila, and Mervi
Kunnasranta. A mark–recapture approach for estimating population size of the
endangered ringed seal (Phoca hispida saimensis). PLOS ONE, 14:214–269, 2019.
14 Santosh Kumar, Amit Pandey, K. Sai Ram Satwik, Sunil Kumar, Sanjay Kumar
Singh, Amit Kumar Singh, and Anand Mohan. Deep learning framework for
recognition of cattle using muzzle point image pattern. Measurement, 116:1–17,
2018.
15 Mervi Kunnasranta, Marja Niemi, Miina Auttila, Mia Valtonen, Juhana Kammo-
nen, and Tommi Nyman. Sealed in a lake—biology and conservation of the
endangered Saimaa ringed seal: A review. Biological Conservation, 253:108908,
2021.
16 Ning Liu, Qijun Zhao, Nan Zhang, Xinhua Cheng, and Jianing Zhu. Pose-
Guided Complementary Features Learning for Amur Tiger Re-Identification. In
International Conference on Computer Vision Workshop, 2019.
17 Rafal Mantiuk, Karol Myszkowski, and Hans-Peter Seidel. A perceptual frame-
work for contrast processing of high dynamic range images. ACM Transactions on
Applied Perception, 3:286–308, 2006.
18 Vincent Miele, Gaspard Dussert, Bruno Spataro, Simon Chamaillé-Jammes,
Dominique Allainé, and Christophe Bonenfant. Revisiting animal photo-
identification using deep metric learning and network analysis.Methods in Ecology
and Evolution, 12(5):863–873, 2021.
19 Anastasiia Mishchuk, Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Working
hard to know your neighbor’s margins: Local descriptor learning loss. Advances
in Neural Information Processing Systems , 30, 2017.
20 Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Repeatability is not enough:
Learning affine regions via discriminability. InEuropean Conference on Computer
Vision, pages 284–300, 2018.
21 Olga Moskvyak, Frederic Maire, Feras Dayoub, Asia O. Armstrong, and Mahsa
Baktashmotlagh. Robust re-identification of manta rays from natural markings by
learning pose invariant embeddings. In International Conference on Digital Image
Computing: Techniques and Applications, 2021.
22 Ekaterina Nepovinnykh, Tuomas Eerola, Heikki Kälviäinen, and Gleb Radchenko.
Identification of Saimaa ringed seal individuals using transfer learning. In Inter-
national Conference on Advanced Concepts for Intelligent Vision Systems , pages
211–222, 2018.
23 Ekaterina Nepovinnykh, Tuomas Eerola, and Heikki Kalviainen. Siamese net-
work based pelage pattern matching for ringed seal re-identification. InIEEE/CVF
Winter Conference on Applications of Computer Vision Workshops , pages 25–34,
2020.
24 Ekaterina Nepovinnykh, Ilja Chelak, Tuomas Eerola, and Heikki Kälviäinen.
NORPPA: Novel ringed seal re-identification by pelage pattern aggregation.arXiv
preprint arXiv:2206.02498, 2022.
25 Ekaterina Nepovinnykh, Ilja Chelak, Andrei Lushpanov, Tuomas Eerola, Heikki
Kälviäinen, and Olga Chirkova. Matching individual ladoga ringed seals across
short-term image sequences. Mammalian Biology, 2022.
26 Ekaterina Nepovinnykh, Tuomas Eerola, Vincent Biard, Piia Mutka, Marja Niemi,
Mervi Kunnasranta, and Heikki Kälviäinen. SealID: Saimaa ringed seal re-
identification database. Sensors, 22, 2022.
27 Ekaterina Nepovinnykh, Antti Vilkman, Tuomas Eerola, and Heikki Kälviäinen.
Re-identification of saimaa ringed seals from image sequences. In Scandinavian
Conference on Image Analysis, pages 111–125, 2023.
28 Florent Perronnin and Christopher Dance. Fisher Kernels on Visual V ocabular-
ies for Image Categorization. In Conference on Computer Vision and Pattern
Recognition, 2007.
29 Florent Perronnin, Yan Liu, Jorge Sánchez, and Hervé Poirier. Large-scale image
retrieval with compressed fisher vectors. In IEEE Computer Society Conference
on Computer Vision and Pattern Recognition, pages 3384–3391, 2010.
30 James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, and Andrew Zisser-
man. Object retrieval with large vocabularies and fast spatial matching. In IEEE
Conference on Computer Vision and Pattern Recognition, pages 1–8, 2007.
31 Christoph A Rohner, Stephanie K Venables, Jesse EM Cochran, Clare EM Prebble,
Baraka L Kuguru, Michael L Berumen, and Simon J Pierce. The need for long-term
population monitoring of the world’s largest fish. Endangered Species Research,
47:231–248, 2022.
Camera Traps, AI, and Ecology
6 © Copyright resides with the authors


## Page 7

32 Stefan Schneider, Graham W Taylor, and Stefan C Kremer. Similarity learn-
ing networks for animal individual re-identification: an ecological perspective.
Mammalian Biology, pages 1–16, 2022.
33 Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller. Nonlinear
Component Analysis as a Kernel Eigenvalue Problem. Neural Computation, 10:
1299–1319, 1998.
34 Maxime Vidal, Nathan Wolf, Beth Rosenberg, Bradley Harris, and Alexander
Mathis. Perspectives on Individual Animal Identification from Biology and
Computer Vision. Integrative and Comparative Biology, 61:900–916, 2021.
35 Hendrik Weideman, Chuck Stewart, Jason Parham, Jason Holmberg, Kiirsten
Flynn, John Calambokidis, D Barry Paul, Anka Bedetti, Michelle Henley, Frank
Pope, et al. Extracting identifying contours for african elephants and humpback
whales using a learned appearance model. In IEEE/CVF Winter Conference on
Applications of Computer Vision, pages 1276–1285, 2020.
36 Artem Zhelezniakov, Tuomas Eerola, Meeri Koivuniemi, Miina Auttila, Riikka
Levänen, Marja Niemi, Mervi Kunnasranta, and Heikki Kälviäinen. Segmentation
of Saimaa ringed seals for identification purposes. In International Symposium on
Visual Computing, pages 227–236, 2015.
Camera Traps, AI, and Ecology
© Copyright resides with the authors 7
