# Sign Language Video Segmentation using Temporal Boundary Identification

https://github.com/user-attachments/assets/d3d2e8a4-1edd-407b-86f2-957551614aa5

## Objective
Addressing the persistent challenge of demanding and time-consuming temporal annotation in Sign Language (SL) videos, this project introduces a subtitle-level segmentation approach utilizing Beginning-Inside-Outside (BIO) tagging for precise boundary identification. 
<!--We train a Sequence-to-Sequence (Seq2Seq) model (with and without attention) on optical flow features from BOBSL and YouTube-ASL datasets. Our results demonstrate that the Seq2Seq model with attention significantly outperforms baseline methods, achieving improved segment percentage, F1-score, and IoU for subtitle boundary detection. An additional contribution includes a method for subtitle temporal resolution, designed to streamline manual annotation efforts. -->
## Implementation
### Datasets
* BOBSL consists of 60 manually-aligned videos featuring British Sign Language (BSL) interpretati
ons from BBC broadcasts, accompanied by English subtitles.
* YouTube-ASL dataset provides a comprehensive collection of American Sign Language (ASL) videos with corresponding English subtitles.

For BOBSL, videos are at 25 fps and pre-split into 40 training, 10 validation, and 10 test videos. Most clips are 30-60 minutes long. YouTube-ASL videos range from 40 seconds to 40 minutes, with data split into 70% training, 20% validation, and 10% testing.

### Model Architecture

<img src="https://github.com/user-attachments/assets/8b92e6bd-6172-49e2-a57c-b0974b2b7353" alt="attention_1" width="500">
![Screenshot From 2025-07-01 00-04-12](https://github.com/user-attachments/assets/815b457e-d02e-438e-a2f2-7be698fd2b7e)

Encoder: BiLSTM encoder (2 layers, 128 hidden units,
dropout 0.2) to encode 375x2048 input sequences
from ResNet-101. 

Dcoder: (2 LSTM layers, 128 hidden units, dropout 0.1) uses an attention
mechanism to compute a weighted sum of the encoder outputs, forming a context vector (256 dimensions) at each decoding step. This context vector,
combined with the previous output embedding (128
dimensions), is used to generate logits via a fully
connected layer. A softmax operation is used to
normalize these logits into a probability distribution over the output segments.
### Training
### Inference

### Results

| Model    | Dataset  |   F1     |    IoU   |     %    | # Params | Time     |
|----------|----------|----------|----------|----------|----------|----------|
|Sequence Encoder |   BOBSL<br>YouTube-ASL | 0.58 <br> 0.56 | 0.60 <br>0.58|2.50 <br> 0.70|1.38M <br>1.18M|~14h<br>~15h|
|Autoregressive Encoder| BOBSL<br>YouTube-ASL| 0.55<br> 0.47| 0.51 <br> 0.50|  1.74<br>0.55 | 1.42M <br> 1.26M |~ 1d <br>~ 1d|

| Model    | Dataset  |   F1     |    IoU   |     %    | # Params | Time     |
|----------|----------|----------|----------|----------|----------|----------|
|Seq2Seq Encoder-Decoder w/o attention |   BOBSL<br>YouTube-ASL | 0.58 <br> 0.55 | 0.70 <br>0.58|2.16 <br> 0.87|3.1M <br>3.1M|~15h<br>~19h|
|Seq2Seq Encoder-Decoder w/ attention| BOBSL<br>YouTube-ASL| **0.60**<br> **0.60**| **0.74** <br> **0.62**| **1.03**<br>**0.95** | 7.8M <br> 3.0M |~ 2d <br>~ 2d|

![Screenshot From 2025-07-01 00-04-12](https://github.com/user-attachments/assets/c0e198e6-0df0-4b0f-b180-074a516d25e5)
![Screenshot From 2025-07-01 00-05-22](https://github.com/user-attachments/assets/58ce94df-de3e-4f66-9603-824f1916670b)


* Successful cases:
<img src="https://github.com/user-attachments/assets/8c63f5f2-f19e-41a6-815e-f5164aaab091" alt="b_i_o(bobsl)" width="800">

**Figure 1: Example of Subtitle-level Segmentation**
