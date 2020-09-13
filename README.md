# Fusical: Multimodal Fusion for Video Sentiment

***Update: Fusical has been accepted to ICMI 2020!!!*** 

Stay tuned!

## Introduction
Despite cultural diversity, emotions are universal. We will undertake the [EmotiW 2020 challenge](https://sites.google.com/view/emotiw2020), doing group-level sentiment recognition on videos from across the globe. Given short clips, the goal is to predict whether the video sentiment is positive, neutral, or negative. This problem is interesting because audio-visual sentiment analysis has implications in psychology and mental health.

## Dataset
We worked with the EmotiW 2020 dataset.

<div style="vertical-align:middle"><img src="images/example-classes.jpg" alt="Sample image" width="600px" text-align="center">
</div>

### Architecture
We ensembled models from four modalities: overall scene, pose, audio, and facial.

<div style="text-align:center"><img src="images/ensemble-architecture.jpg" alt="Ensemble Architecture" width="550px" align="center">
</div>

## Getting Started

To start, please check out our [presentation](https://drive.google.com/file/d/1catb9ng8glqUBi6AwJTKjNTnnoZjSwVq/view) and [slide deck](https://docs.google.com/presentation/d/1o5u3VHBYzWLeiQOaD43SvhYKbgfHvc6kYxQ322Xsl4o/edit).

### Code Layout
The code is organized as follows:
- src/ - preprocessing, generation, and classification code
- notebooks/ - notebooks for training and prediction

### Try it out
Run this [notebook](notebooks/ensemble/ensemble_fc_predictions_v2-with-laugh.ipynb) to see how our model works on the dataset.

### Model Emporium 
We provide many of the models we trained [here](model-emporium.md).

## Results

### Final Ensemble
<div style="text-align:center"><img src="images/final_fc_ensemble_conf.jpg" alt="Best Submission Confusion Matrix" width="400px" align="center">
</div>

### Ablation Study

<div style="text-align:center"><img src="images/ablation_conf.jpg" alt="Ablation Confusion Matrices" width="600px" align="center">
</div>

### Table - Individual Modalities

Results as reported based on the EmotiW 2020 validation dataset.

| Modality  | Accuracy | F1-Score
| ------------- | ------------- | ------------- |
| Scene  | 0.546  | 0.541 |
| Pose  | 0.486  | 0.489 |
| Audio|0.577|0.577|
| Face | 0.4 |0.348 |
| Image Captioning | 0.505 | 0.506 |

### Table - Final Ensemble
| Dataset  | Accuracy
| ------------- | -------------
| Validation | **0.640** |
| Test | **0.639** |

### "Saliency" Map
<div style="text-align:center"><img src="images/ensemble-saliency-2.jpg" alt="Ablation Confusion Matrices" width="400px" align="center">
</div>

## The Team
[Boyang Tom Jin](https://www.linkedin.com/in/tomjinvancouver/), [Leila Abdelrahman](http://leilaabdel.com/), [Cong Kevin Chen](https://www.linkedin.com/in/cong-kevin-chen-11544186/), [Amil Khanzada](https://www.linkedin.com/in/amilkhanzada)<br>
[CS231n - Stanford University](http://cs231n.stanford.edu/)

## Citing Our Work
```
@misc{2020fusical,
  author =       {Boyang Tom Jin and Leila Abdelrahman and Cong Kevin Chen and
                  Amil Khanzada},
  title =        {Fusical},
  howpublished = {\url{https://github.com/kevincong95/cs231n-emotiw}},
  year =         {2020}
}
```
