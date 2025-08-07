# Sign Language Video Segmentation using Temporal Boundary Identification
Details of this research can be found in our accepted paper: https://aclanthology.org/2025.acl-srw.93/


https://github.com/user-attachments/assets/18a3792f-0f52-46e3-9a39-62f97956155f


<!-- https://github.com/user-attachments/assets/5d79764d-cb28-490e-875e-3d8ef9e7943b -->

## Objective

Addressing the persistent challenge of demanding and time-consuming temporal annotation in Sign Language (SL) videos, this project introduces a subtitle-level segmentation approach utilizing Beginning-Inside-Outside (BIO) tagging for precise boundary identification. 
<!--We train a Sequence-to-Sequence (Seq2Seq) model (with and without attention) on optical flow features from BOBSL and YouTube-ASL datasets. Our results demonstrate that the Seq2Seq model with attention significantly outperforms baseline methods, achieving improved segment percentage, F1-score, and IoU for subtitle boundary detection. An additional contribution includes a method for subtitle temporal resolution, designed to streamline manual annotation efforts. -->
## Implementation
### Datasets
* BOBSL:The aligned subtitle segments and pre-extracted optical flow features, is available for direct download at https://www.robots.ox.ac.uk/~vgg/data/bobsl/
* YouTube-ASL: Download raw videos 'download_replace_empty_subtitles.py' and extract optical flow features 'video_flow_estimation'. Dataset is available at: https://github.com/google-research/google-research/blob/master/youtube_asl/README.md

For BOBSL, videos are at 25 fps and pre-split into 40 training, 10 validation, and 10 test videos. Most clips are 30-60 minutes long. YouTube-ASL videos range from 40 seconds to 40 minutes, with data split into 70% training, 20% validation, and 10% testing.

### Model Architecture
Figure 1

<img src="https://github.com/user-attachments/assets/8b92e6bd-6172-49e2-a57c-b0974b2b7353" alt="attention_1" width="500">

### Proposed Method
* Input Features:
* **Input:** Optical Flow (RAFT)
ResNet-101 Feature Extraction
* Segmentation Model:
Core: Seq2Seq Architecture with & without Attention (Figure 1)
Encoders: BiLSTM & Autoregressive (Baselines)
Output: Beginning-Inside-Outside (BIO) for Boundaries.
* Subtitle Temporal Resolution:
Converts predictions to time-stamped SubRip Subtitle (.srt) Files
Uses Beam Search Inference
<!--
Encoder: BiLSTM encoder (2 layers, 128 hidden units,
dropout 0.2) to encode 375x2048 input sequences
from ResNet-101. 

Dcoder: (2 LSTM layers, 128 hidden units, dropout 0.1) uses an attention
mechanism to compute a weighted sum of the encoder outputs, forming a context vector (256 dimensions) at each decoding step. This context vector,
combined with the previous output embedding (128
dimensions), is used to generate logits via a fully
connected layer. A softmax operation is used to
normalize these logits into a probability distribution over the output segments. -->
### Training

![Screenshot From 2025-07-10 07-40-34](https://github.com/user-attachments/assets/788bd528-0b73-4a10-b65d-d261482ffae5)

### Inference
Algorithm to map model probabilities to subtitle boundaries.

![Screenshot From 2025-07-01 00-04-12](https://github.com/user-attachments/assets/c0e198e6-0df0-4b0f-b180-074a516d25e5)
<!-- ![Screenshot From 2025-07-01 00-05-22](https://github.com/user-attachments/assets/58ce94df-de3e-4f66-9603-824f1916670b) -->

### Results

<img src="https://github.com/user-attachments/assets/4fd63f9c-5e3d-4d15-9192-7981101b8193" alt="Screenshot From 2025-07-10 07-22-30" width="800">
<img src="https://github.com/user-attachments/assets/a771532d-d533-45cc-9a46-0085a9037fed" alt="Screenshot From 2025-07-10 07-22-46" width="800">


<!-- | Model    | Dataset  |   F1     |    IoU   |     %    | # Params | Time     |
|----------|----------|----------|----------|----------|----------|----------|
|Sequence Encoder |   BOBSL<br>YouTube-ASL | 0.58 <br> 0.56 | 0.60 <br>0.58|2.50 <br> 0.70|1.38M <br>1.18M|~14h<br>~15h|
|Autoregressive Encoder| BOBSL<br>YouTube-ASL| 0.55<br> 0.47| 0.51 <br> 0.50|  1.74<br>0.55 | 1.42M <br> 1.26M |~ 1d <br>~ 1d|

| Model    | Dataset  |   F1     |    IoU   |     %    | # Params | Time     |
|----------|----------|----------|----------|----------|----------|----------|
|Seq2Seq Encoder-Decoder w/o attention |   BOBSL<br>YouTube-ASL | 0.58 <br> 0.55 | 0.70 <br>0.58|2.16 <br> 0.87|3.1M <br>3.1M|~15h<br>~19h|
|Seq2Seq Encoder-Decoder w/ attention| BOBSL<br>YouTube-ASL| **0.60**<br> **0.60**| **0.74** <br> **0.62**| **1.03**<br>**0.95** | 7.8M <br> 3.0M |~ 2d <br>~ 2d| -->


* Successful cases:
  
<img src="https://github.com/user-attachments/assets/9f9759cf-37c2-4e4c-bb8f-67c65219fad0" alt="Screenshot From 2025-07-10 07-28-06" width="800">

<img src="https://github.com/user-attachments/assets/bba8182f-8438-45b0-a70f-f513e60f54af" alt="sample" width="800">

**Figure 1: Subtitle-level segmentation using BIO tags (B: green, I: light blue, O: white). Our attention-based model predicts these tags based on probabilities, delineating subtitle boundaries.**

* Failure cases:

<img src="https://github.com/user-attachments/assets/07277311-76f3-42ff-9146-a6ef27511c99" alt="asl_g+i" width="800">

<img src="https://github.com/user-attachments/assets/331180be-14d0-4ba7-a760-4c2783dd872f" alt="wrong_asl" width="800">


