# Open-set Text Recognition via Character-Context Decoupling

The main part of the code is located in neko_2021_mjt/standardbench2_candidates

All data for training and most data for testing (except dictionary based close-set word recogntion) are uploaded to the Kaggle repo:

https://www.kaggle.com/vsdf2898kaggle/osocrtraining

The code has been tested on a fresh Manjaro (by 11-03-2022). If the code fails, please open an issue or contact me. 

Since we only use Arch-Linux-based systems (Arch and Manjaro), backward compatibility issues may occur. We will try our best to patch the code should this happen. However, we cannot guarantee a sure fix on backward compatibility issues. 

Also, if you have any questions or concerns about the paper or the code, feel free to open an issue so we can improve them. 


## Usage 


Please refer to the manul.pdf. 

[UPD] Dependency installation can be simplified with the scripts here:

https://github.com/lancercat/make_env/

## Trained Models 
All trained models are released in the Kaggle repo.

https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=models-release

### Naming 

#### Dataset
Dual_a: Open-set text recognition (Chinese-to-Japanese transfer)

Dual_b: Close-set text recognition (Conventional benchmarks)

Dual_ch: Zero-shot character recognition (CTW)

Dual_chhw: Zero-shot character recognition (Handwritten)

#### Method

Base model: Odancukmk7hnp_r45_C_trinorm_dsa3

DTA only: Odancukmk7hdtfnp_r45_C_trinorm_dsa3

Full model: Odancukmk8ahdtfnp_r45_C_trinorm_dsa3

Full model-Large: Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3

## Dataset
The dataset and dict files(templates from noto fonts) we use is the same as OSOCR
https://github.com/lancercat/OSOCR
We follow the OSOCR's convention on pre-processing and exclusion of data, which is detailed in their code and arxiv preprint.

https://www.techrxiv.org/articles/preprint/Towards_Open-Set_Text_Recognition_via_Label-to-Prototype_Learning/16910062

Most training datasets and evaluation datasets are now uploaded to 
https://www.kaggle.com/vsdf2898kaggle/osocrtraining


ALL models used in the paper are now released in the models-release folder (Having 100GB free space is so nice!)
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=models-release

Please follow the manual (manul.pdf) to proceed evaluation/training.


### Open-set text recognition
The japanese evaluation set(~4000 samples) can be found in the OSOCR repo, and both of the regular model and the large model are uploaded to dropbox. We ran 200k iterations to train this task. 

The training datasets are located in the ssddata folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=ssddata

The testing datasets are located in the ssddata_2 folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=ssddata_2

The models are located in the models-release folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=models-release

### Zero-shot text recognition
All models are trained for 50k iterations for the two datasets and total eight scenerios ((ctw,hwdb)\*(500,1000,1500,2000 training chars)) in this part.

The training datasets and testing datasets are both located in the ssddata_1 folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=ssddata_1

The models are located in the models-release folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=models-release


### Close-set benchmarks

We use the evaluation datasets shared by ClovaAI: 
https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0

Their repo: https://github.com/clovaai/deep-text-recognition-benchmark

We also offer a alternative copy in the ssddata_2 folder
https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=ssddata_2 



Despite being MJ-ST combination, there are a number of varients. We use the datasets provided in DAN and it's uploaded to the Kaggle repo.

https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=NIPS2014

https://www.kaggle.com/vsdf2898kaggle/osocrtraining?select=CVPR2016

You need to rearrange them like the following in the ssddata folder.

![image](https://user-images.githubusercontent.com/59994105/157666662-db003c7f-baff-4584-bbe9-db5d75d45c12.png)


Alternatively, you can use the DAN version, and the datasets can be found here:
https://github.com/Wang-Tianwei/Decoupled-attention-network


We ran 800k iterations for training model used on this task


### Measure and Statistic
The definition Line Accuracy(Word Accuracy) and Character Accuracy(1-NED) can be found here:

https://rrc.cvc.uab.es/?ch=14&com=tasks

## Infanstructure
Ablative models are trained on a server with 4 1080tis, 2 E5-2620 v4 CPUs, and 64Gib RAM.

Torch version is 1.7.1 (sry I previously mistook it as 1.7.0) and CUDA version is V10.1.243.
![image](https://user-images.githubusercontent.com/59994105/194462501-a830c9e4-7356-4c24-a72a-7721b2ed248c.png)

Evaluation is conducted on:
1. Laptop with RTX 2070 mobile(~7 Tflops) and I5-9400 (as a reference of low-energy-consumption deployment).
2. Server with one RTX 3090(~28 Tflops) and 2 E5-2620v3 CPUs (as a reference of speed-first deployment).

Generally, the regular model takes ~6 Gib to train and less than 1.8 Gib for evaluation, and the large model takes ~11 Gib to train and 2.4 Gib for evaluation.

## Extra Hints
It may take a while before we can find time to make a detailed tutorial on how to make your very own training set from raw images and fonts.

However, here are a few hints which may help you hack thru.
1) There are no extra hacks on lmdbs, what works with DAN should work here.  
2) There is a vertical filter in the dataloader. If you need vertical capability and are confident with your mods, remove it.
3) The dict (.pt files) used for training can be generated the same way described in the manual.
4) GLHF
