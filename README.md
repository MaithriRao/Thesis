# Sign Language Video Segmentation using Temporal Boundary Identification
## Objective
Addressing the persistent challenge of demanding and time-consuming temporal annotation in Sign Language (SL) videos, this project introduces a subtitle-level segmentation approach utilizing Beginning-Inside-Outside (BIO) tagging for precise boundary identification. We train a Sequence-to-Sequence (Seq2Seq) model (with and without attention) on optical flow features from BOBSL and YouTube-ASL datasets. Our results demonstrate that the Seq2Seq model with attention significantly outperforms baseline methods, achieving improved segment percentage, F1-score, and IoU for subtitle boundary detection. An additional contribution includes a method for subtitle temporal resolution, designed to streamline manual annotation efforts.
## Implementation
### Datasets
* BOBSL consists of 60 manually-aligned videos featuring British Sign Language (BSL) interpretations from BBC broadcasts, accompanied by English subtitles.
* YouTube-ASL dataset provides a comprehensive collection of American Sign Language (ASL) videos with corresponding English subtitles.

For BOBSL, videos are at 25 fps and pre-split into 40 training, 10 validation, and 10 test videos. Most clips are 30-60 minutes long. YouTube-ASL videos range from 40 seconds to 40 minutes, with data split into 70% training, 20% validation, and 10% testing.

### Model Architecture
### Training
### Inference
