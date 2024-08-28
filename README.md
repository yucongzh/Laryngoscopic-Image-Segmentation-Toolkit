# Laryngoscopic-Image-Segmentation-Toolkit
This Github repo is an open-source toolkit and display of the laryngoscopic image segmentation system proposed in our work: 

## Introduction of the system
The system produces vocal folds and glottis masks from input laryngoscopic images. It has been trained on BAGLS dataset containing 59250-frame wise glottis annotations extracted from endoscopic high-speed videos (HSV) for glottis segmentation and larynx area object detection. The system then segments vocal folds by using image processing methods to extract prompts and appying prompt engineering methods for the prompt-based segmentation anything model (SAM).
## Segmentation demos
The following Figure shows several segmentation results on Fehling et al.'s dataset:
![Page 1](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/results.jpg)
We also improve the system to enable segmentation for laryngoscopic video. Here are some examples: 

![Page 2](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_1_tmp.gif)![Page 3](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_2_tmp.gif)![Page 4](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_3_tmp.gif)

![Page 5](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_1.gif)![Page 6](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_2.gif)![Page 7](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_3.gif)

## Model Checkpoints

## Using the code
We provide a jupyter notebook that enables users to input laryngoscopic images and obtain output images with glottis and vocal fold masks. You will need to adapt the paths for image and model loading. 
The following are requirements for runnning the notebook:
```
python==3.8
```
## License

## Citing the work

## Reference
