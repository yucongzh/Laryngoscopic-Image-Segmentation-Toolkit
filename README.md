# Laryngoscopic-Image-Segmentation-Toolkit
This Github repo is an open-source toolkit and display of the laryngoscopic image segmentation system proposed in our work: 

## Introduction of the system
The system produces vocal folds and glottis masks from input laryngoscopic images. It has been trained on BAGLS dataset containing 59250-frame wise glottis annotations extracted from endoscopic high-speed videos (HSV) for glottis segmentation and larynx area object detection. The system then segments vocal folds by using image processing methods to extract prompts and appying prompt engineering methods for the prompt-based segmentation anything model (SAM).
## Segmentation demos
The following Figure shows several segmentation results on Fehling et al.'s dataset:
![Page 1](https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/results.jpg)
We also improve the system to enable segmentation for laryngoscopic video. Here are some examples: 

<div align=center>
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_1_tmp.gif" style="max-width: auto; height: auto;">
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_2_tmp.gif" style="max-width: auto; height: auto;">
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_3_tmp.gif" style="max-width: auto; height: auto;">
</div>

<div align=center>
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_1.gif" style="max-width: auto; height: auto;">
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_2.gif" style="max-width: auto; height: auto;">
  <img src="https://github.com/EEugeneS/Laryngoscopic-Image-Segmentation-Toolkit/blob/main/Demos/gif/video_masked_3.gif" style="max-width: auto; height: auto;">
</div>

## Model Checkpoints

## Using the code
We provide a jupyter notebook that enables users to input laryngoscopic images and obtain output images with glottis and vocal fold masks. You will need to adapt the paths for image and model loading. 
The following are requirements for runnning the notebook:
```
python==3.8
segment-anything==1.0
```
## License

## Citing the work

## Reference
- Fehling, M. K., Grosch, F., Schuster, M. E., Schick, B., & Lohscheller, J. (2020). Fully automatic segmentation of glottis and vocal folds in endoscopic laryngeal high-speed videos using a deep Convolutional LSTM Network. PLOS ONE, 15(2), e0227791. https://doi.org/10.1371/journal.pone.0227791
  
- Gómez, P., Kist, A. M., Schlegel, P., Berry, D. A., Chhetri, D. K., Dürr, S., Echternach, M., Johnson, A. M., Kniesburges, S.Kunduk, M., Youri Maryn, Schützenberger, A., Verguts, M., & Döllinger, M. (2020). BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. Scientific Data, 7(1). https://doi.org/10.1038/s41597-020-0526-3

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., & Girshick, R. (2023). Segment Anything. Openaccess.thecvf.com. http://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html
