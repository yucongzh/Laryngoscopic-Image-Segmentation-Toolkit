[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# Laryngoscopic-Image-Segmentation-Toolkit
This Github repo is an open-source toolkit and display of the laryngoscopic image segmentation system proposed in our work: 

## Introduction of the system
The system produces vocal folds and glottis masks from input laryngoscopic images. It has been trained on BAGLS dataset [\[1\]](#ref-1) containing 59250-frame wise glottis annotations extracted from endoscopic high-speed videos (HSV) for glottis segmentation and larynx area object detection. The system then segments vocal folds by using image processing methods to extract prompts and appying prompt engineering methods for the prompt-based segmentation anything model (SAM) [\[2\]](#ref-2). 

As for the evaluation of the system's segmentation accuracy, we use Fehling et al.'s dataset [\[3\]](#ref-3), which is the only open-source dataset containing frame wise glottis and vocal fold annotations extracted from HSV.

## Segmentation demos
The following Figure shows several segmentation results on Fehling et al.'s dataset [\[3\]](#ref-3):
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

## Toolkit Instructions
1. Clone YOLO-v5 repository from [ultralytics/yolov5](https://github.com/ultralytics/yolov5), and put it in `Toolkit/`.
2. Download our trained U-Net and YOLO-v5 from [huggingface.co](https://huggingface.co/yucongzh/glottis_segmentation/tree/main), and save it to `Toolkit/checkpoints/`.
3. Download the SAM checkpoint for the`vit_h` model type from the official [Segment Anything GitHub repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints), and save it to `Toolkit/checkpoints/`.
4. Put your laryngoscopic images under `data/`.
5. Go to `Toolkit/` and run `python main.py --filename [IMAGE NAME]`.

## Acknowledgements
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## Citations
@article{zhang2024multimodal,  
  title={Multimodal Laryngoscopic Video Analysis for Assisted Diagnosis of Vocal Cord Paralysis}, 
  author={Zhang, Yucong and Zou, Xin and Yang, Jinshan and Chen, Wenjun and Liang, Faya and Li, Ming},  
  journal={arXiv preprint arXiv:2409.03597},  
  year={2024}  
}

## References
<a id="ref-1"></a>[1] Gómez, P., Kist, A. M., Schlegel, P., Berry, D. A., Chhetri, D. K., Dürr, S., Echternach, M., Johnson, A. M., Kniesburges, S.Kunduk, M., Youri Maryn, Schützenberger, A., Verguts, M., & Döllinger, M. (2020). BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. Scientific Data, 7(1). https://doi.org/10.1038/s41597-020-0526-3

<a id="ref-2"></a>[2] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., & Girshick, R. (2023). Segment Anything. Openaccess.thecvf.com. http://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html

<a id="ref-3"></a>[3] Fehling, M. K., Grosch, F., Schuster, M. E., Schick, B., & Lohscheller, J. (2020). Fully automatic segmentation of glottis and vocal folds in endoscopic laryngeal high-speed videos using a deep Convolutional LSTM Network. PLOS ONE, 15(2), e0227791. https://doi.org/10.1371/journal.pone.0227791


## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
