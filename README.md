
## Pedestron-Caltech

This work is based on [Pedestron](https://github.com/hasanirtiza/Pedestron) repository (that focuses on the advancement of research on pedestrian detection). This repository runs Pedestron repository with some [changes](https://github.com/hasanirtiza/Pedestron/pull/150) to be compatible with CUDA v11.7 to do training and testing specifically on the Caltech as a pedestrian detection dataset. This repository contains the environment, installation steps, run  demo, caltech dataset prepration, training and testing on the Caltech dataset.


### Environment
Debian 11, CUDA 11.7 and RTX3060Ti.


### Installation

conda create -name pedestron python==3.8<br/>
conda activate pedestron<br/>
pip install mmcv==0.2.10<br/>
pip install cython<br/>
pip install numpy<br/>
pip install instaboostfast<br/>
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia<br/>

cd Pedestron_caltech_dataset<br/>

Apply all [changes](https://github.com/hasanirtiza/Pedestron/pull/150)<br/>

```shell 
python setup.py install
```

## Getting Started

### Running a demo using pre-trained model on few images
Pre-trained model can be evaluated on sample images in the following way
```shell 
python tools/demo.py configs/elephant/cityperson/cascade_hrnet.py pretrained_models/epoch_5.pth.stu demo/ result_demo/ 
```
### Caltech Datasets Preparation
* I used [Caltech-images-convertor](https://github.com/Nazila-H/Caltech-images-convertor) which is based on [caltech-pedestrian-dataset-converter](https://github.com/mitmul/caltech-pedestrian-dataset-converter) repository to convert Caltech pedestrian videos into the format that is needed for Pedestron repository.

* The json file of annotations (train and test sets) are provided by [Pedestron](https://github.com/hasanirtiza/Pedestron/tree/master/datasets/Caltech)


### Training on the Caltech Pedestrian Dataset

```shell
python tools/train.py ${CONFIG_FILE}
```
For instance, training on the Caltech dataset:

```shell
python tools/train.py configs/elephant/caltech/cascade_hrnet.py
```
The results will be saved in `work_dirs` folder (with epoch_number.pth.stu format)

### Testing on the Caltech Pedestrian Dataset

1) Using the downloaded weight `epoch_14.pth.stu` (Cascade Mask R-CNN) as a starting checkpoint and making the prediction on the Caltech testset:
```shell
python tools/test_caltech.py configs/elephant/caltech/cascade_hrnet.py pretrained_models/epoch_ 14 15 --out result_caltech.json [--mean_teacher]
```

2) Using the result of training on the Caltech dataset (they are saving on `work_dirs` folder automatically) as a checkpoint (putting them on `pretrained_models` folder)  and making the prediction on the Caltech testset:
```shell
python tools/test_caltech.py configs/elephant/caltech/cascade_hrnet.py pretrained_models/epoch_ 20 21 --out result_caltech.json [--mean_teacher]
```

The result will be saved as `result_caltech.json14.json` (from 1).

Convert the result to txt by: 
```shell
python tools/caltech/convert_json_to_txt.py
```
The results will be saved at `tools/caltech/eval_caltech/Predestron_Results/result_caltech
                                                                                         |__set06
                                                                                            |__V000.txt 
                                                                                            |__... 
                                                                                            |__V018.txt
                                                                                         |__set07
                                                                                            |__V000.txt
                                                                                            |__... 
                                                                                            |__V011.txt
                                                                                         |__set08
                                                                                            |__V000.txt 
                                                                                            |__... 
                                                                                            |__V010.txt
                                                                                         |__set09
                                                                                            |__V000.txt
                                                                                            |__... 
                                                                                            |__V011.txt
                                                                                         |__set10
                                                                                            |__V000.txt
                                                                                            |__... 
                                                                                            |__V011.txt 

Chainging the name of `result_caltech` folder to `epoch_14`<br/>
cd Pedestron-Caltech/tools/caltech/eval_caltech<br/>
matlab -nodisplay -nosplash -nodesktop -r "run('dbEval.m');exit;"<br/>

The result will be saved in `eval_caltech/ResultEval/eval_newResonable.txt` for the value of MR.




### Caltech dataset folder structure 
```shell   
datasets/Caltech
|__test_images/   # 128419 train-images
   |__set06_V000_I00000.jpg
   |__set06_V000_I00001.jpg
   |__ ...
   |__set10_V011_I01733.jpg
|__train_images/   # 121465 test_images
   |__set00_V000_I00000.jpg 
   |__set00_V000_I00001.jpg
   |__ ...
   |__set05_V012_I01706.jpg   
|__test.json
|__train.json  

```


### Citation
[CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)
```
@InProceedings{Hasan_2021_CVPR,
    author    = {Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
    title     = {Generalizable Pedestrian Detection: The Elephant in the Room},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11328-11337}
}

```

[ArXiv2022](https://arxiv.org/pdf/2201.03176.pdf)
```
@article{hasan2022pedestrian,
  title={Pedestrian Detection: Domain Generalization, CNNs, Transformers and Beyond},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
  journal={arXiv preprint arXiv:2201.03176},
  year={2022}
}
```
