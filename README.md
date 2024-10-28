# OpenCV Image Super Resolution Example

Deep Learning based Super Resolution with OpenCV

## Steps

* Install the OpenCV with contrib modules (including dnn_superres)
* Download the pre-trained models
* Upscaling the image

## Set the super-resolution model

```python
import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(<model_path>)
sr.setModel(<model_name>)
```

## Upscaling the image

```python
image = cv2.imread('./input.png')
result = sr.upsample(image)
```

need to take care about the upsampling ratio and the model you use.

## Models

| Model Name | Performance | Model Size |
|----|----|----|
| EDSR | Best | Large |
| ESPCN | Normal | Small |
| FSRCNN | Normal | Small |
| LapSRN | Better | Medium |

ESPCN, FSRCNN are small, and good for inferencing. They can do real-time video upscaling.

## Model Download

* EDSR:  <https://github.com/Saafke/EDSR_Tensorflow/tree/master/models>
* ESPCN:  <https://github.com/fannymonori/TF-ESPCN/tree/master/export>
* FSRCNN:  <https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models>
* LapSRN:  <https://github.com/fannymonori/TF-LapSRN/tree/master/export>


