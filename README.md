This repo is modified from https://github.com/qiuqiangkong/panns_transfer_to_gtzan<br>
If you want view all pretrains, please go to https://github.com/qiuqiangkong/audioset_tagging_cnn<br>

# Tomofun 狗音辨識 AI 百萬挑戰賽 classification finetuned on pretrained audio neural networks (PANNs)
  Audio tagging is a task to classify audio clips into classes. Tomofun 狗音辨識 AI 百萬挑戰賽 is a competetion containing 1200 5-second audio clips with 6 classes ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking','Other']. In this codebase, we fine-tune PANNs [1] to build audio classification systems.

## TODO
- [ ] Add confusion matrix
- [ ] Plot Curve Notebooks
- [ ] Grad Cam

## Dataset
The dataset can be downloaded from https://tbrain.trendmicro.com.tw/Competitions/Details/15

## Run the code

**0. Prepare data** 

Download and upzip data, the data looks like:
<pre>
meta_train.csv

train
├── train_00001.wav
├── train_00002.wav
├── ...
└── train_01200.wav

public_test
├── public_00001.wav
├── public_00002.wav
├── ...
└── public_10000.wav

private_test
├── private_00001.wav
├── private_00002.wav
├── ...
└── private_20000.wav
</pre>

where meta_train.csv should in this format:

| Filename  | Label | Remark |
| - | - | - |
| train_00001  | 0  | Barking |
| train_00002  | 0  | Barking |
| ... | ... | ... |
| train_01200  | 5  | Dishes |

**1. Requirements** 

python 3.7.6
> \# in requirements.txt<br>
> matplotlib==3.4.2<br>
> dotmap==1.3.23<br>
> tensorflow==2.3.1<br>
> numpy==1.16.6<br>
> librosa==0.8.1<br>
> pandas==1.2.4<br>
> tqdm==4.61.1<br>

<pre>
pip install requirements.txt
bash download_cnn14.sh
</pre>

**2. Prepare Dataset**
1. Run all code in Reorder_File.ipynb # note : the file train_01046.wav is omitted.
2. <pre>bash prepare_hdf5.sh
# it cost about 2.5 hour in my machine
# if you only need to training, please do not run the last 2 lines.
</pre>

Note : if you want to try your own dataset, please modify following files
* Reorder_File.ipynb
* prepare_hdf5.sh
* ./utils/config.py

**3. Start Training & Evaluate**
1. <pre>bash train.sh 
# note : set augmentation to "none" is better in this dataset in our experiment</pre>
2. Run all code in UseFinetunedModelToPredict.ipynb # if you have modified the parameters, please correct them in the config section.
3. Check softmax_then_mean_from_panns_transfer_to_gtzan.csv

## Model
A 14-layer CNN of PANNs is fine-tuned. We use 10-fold cross validation for Tomofun 狗音辨識 AI 百萬挑戰賽 classification. That is, 1080 audio clips are used for training, and 120 audio clips are used for validation.

## Results
The system takes around 8 minutes to fit 600 mini-batch with a single card GeForce GTX 1080 Ti GPU card. Here is the result on 1nd fold. The results on different folds can be different.

<pre>
Sun, 20 Jun 2021 19:34:01 main.py[line:69] INFO Namespace(augmentation='mixup', batch_size=32, cuda=True, dataset_dir='./train_transfered', filename='main', freeze_base=False, holdout_fold='1', learning_rate=0.0001, loss_type='clip_nll', mode='train', model_type='Transfer_Cnn14', pretrained_checkpoint_path='./Cnn14_mAP=0.431.pth', resume_iteration=0, stop_iteration=600, suffix='_train', workspace='.')
Sun, 20 Jun 2021 19:34:01 main.py[line:72] INFO Using GPU.
Sun, 20 Jun 2021 19:34:03 main.py[line:85] INFO Load pretrained model from ./Cnn14_mAP=0.431.pth
Sun, 20 Jun 2021 19:34:14 main.py[line:151] INFO ------------------------------------
Sun, 20 Jun 2021 19:34:14 main.py[line:152] INFO Iteration: 10
Sun, 20 Jun 2021 19:34:15 main.py[line:157] INFO Validate accuracy: 0.250
Sun, 20 Jun 2021 19:34:15 main.py[line:158] INFO Validate loss: 0.29280
Sun, 20 Jun 2021 19:34:15 utilities.py[line:103] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics.pickle
Sun, 20 Jun 2021 19:34:15 utilities.py[line:104] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics_2021-06-20_19-34-03.pkl
Sun, 20 Jun 2021 19:34:15 main.py[line:168] INFO Train time: 6.956 s, validate time: 1.354 s
Sun, 20 Jun 2021 19:34:21 main.py[line:151] INFO ------------------------------------

............

Sun, 20 Jun 2021 19:36:40 main.py[line:151] INFO ------------------------------------
Sun, 20 Jun 2021 19:36:40 main.py[line:152] INFO Iteration: 200
Sun, 20 Jun 2021 19:36:41 main.py[line:157] INFO Validate accuracy: 0.892
Sun, 20 Jun 2021 19:36:41 main.py[line:158] INFO Validate loss: 0.05802
Sun, 20 Jun 2021 19:36:41 utilities.py[line:103] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics.pickle
Sun, 20 Jun 2021 19:36:41 utilities.py[line:104] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics_2021-06-20_19-34-03.pkl
Sun, 20 Jun 2021 19:36:41 main.py[line:168] INFO Train time: 6.185 s, validate time: 1.518 s
Sun, 20 Jun 2021 19:36:42 main.py[line:182] INFO Model saved to ./checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/200_iterations.pth
Sun, 20 Jun 2021 19:36:48 main.py[line:151] INFO ------------------------------------

............

Sun, 20 Jun 2021 19:39:16 main.py[line:151] INFO ------------------------------------
Sun, 20 Jun 2021 19:39:16 main.py[line:152] INFO Iteration: 400
Sun, 20 Jun 2021 19:39:17 main.py[line:157] INFO Validate accuracy: 0.925
Sun, 20 Jun 2021 19:39:17 main.py[line:158] INFO Validate loss: 0.04342
Sun, 20 Jun 2021 19:39:17 utilities.py[line:103] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics.pickle
Sun, 20 Jun 2021 19:39:17 utilities.py[line:104] INFO     Dump statistics to ./statistics/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/statistics_2021-06-20_19-34-03.pkl
Sun, 20 Jun 2021 19:39:17 main.py[line:168] INFO Train time: 6.161 s, validate time: 1.515 s
Sun, 20 Jun 2021 19:39:17 main.py[line:182] INFO Model saved to ./checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/400_iterations.pth
Sun, 20 Jun 2021 19:39:24 main.py[line:151] INFO ------------------------------------

............

</pre>

## Citation

[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." arXiv preprint arXiv:1912.10211 (2019).
