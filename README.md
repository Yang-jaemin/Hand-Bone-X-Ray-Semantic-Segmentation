# 🩻 HandBone X-Ray Segmentation


## 🔎 Project Overview


<img width="1079" alt="스크린샷 2023-06-30 오후 4 34 39" src="https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-10/assets/103094730/6cb931a2-f402-4772-b38d-0ae33c320017">

의료분야에서 Segmentation task는 진단 및 치료 계획을 개발하는 데 필수적이다. Bone Segmentation은 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있다. 또한 수술 계획을 세우거나 의료 장비에 필요한 정보를 제공하고 교육 목적으로도 사용될 수 있다. 이번 프로젝트를 통해 뼈를 정확하게 Segmentation하는 모델을 개발함으로써 의료 분야에 다양한 목적으로 도움이 되고자 했다.

<br/>

## 👨‍👨‍👧‍👦 Members


| 이름 | github | 맡은 역할 |
| --- | --- | --- |
| 김보경 &nbsp;| [github](https://github.com/bogeoung) | Baseline code 리팩토링, smp Model 구현 및 실험, Loss weight 실험, Optuna 구현,    inference ensemble 구현|
| 김정주 | [github](https://github.com/Kim-Jeong-Ju) | Augmentation 실험, Model 설계/성능 측정, Loss 설계/실험, Visualization 구축 |
| 양재민 | [github](https://github.com/Yang-jaemin) | Augmentation 실험, smp Model 구현 및 실험 Scratch Model 구현, 실험 |
| 임준표 | [github](https://github.com/anonlim) | Augmentation 실험, Model 구현 및 실험, inference augmentation, 학습 속도 가속화 |
| 정남교 | [github](https://github.com/jnamq97) | kfold 구현, Optuna 구현, ensemble 구현, smp Model 구현 및 실험, augmentation 실험 |
<br/>

## 📷 Dataset


- 이미지 크기 : (2048, 2048)

<img width="453" alt="스크린샷 2023-06-30 오후 4 50 32" src="https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-10/assets/103094730/33cac93a-58cf-4597-9b31-9286501105db">


- 29 classes : 손가락 / 손등 / 팔로 구성
    - finger-1, finger-2, finger-3, finger-4, finger-5,finger-6, finger-7, finger-8, finger-9, finger-10,finger-11, finger-12, finger-13, finger-14, finger-15,finger-16, finger-17, finger-18, finger-19, Trapezium,Trapezoid, Capitate, Hamate, Scaphoid, Lunate,Triquetrum, Pisiform, Radius, Ulna
    
<br/>

## 🗂️ Structure


```python

input/
|-- code
|   |-- adamp.py
|   |-- alarm.py
|   |-- app.py
|   |-- app_utils.py
|   |-- data_eda.ipynb
|   |-- dataset.py
|   |-- ensemble.py
|   |-- inference.py
|   |-- inference_kfold.py
|   |-- loss.py
|   |-- model.py
|   |-- requirements.txt
|   |-- train.py
|   |-- train_optuna.py
|   `-- wbf_ensemble.py
`-- data
    |-- test
    |   |-- DCM
    |   |   |-- ID040
    |   |   |   |-- image1.png
    |   |   |   `-- image2.png
    |   |   |-- ID041
            ...
    |   |   |-- ID550
    `-- train
        |-- DCM
        |   |-- ID001
        |   |   |-- image1.png
        |   |   `-- image2.png
        |   |-- ID002
            ...
        |   `-- ID548
        `-- outputs_json
            |-- ID001
            |   |-- image1.json
            |   `-- image2.json
            |-- ID002
            ...
            `-- ID548
```   

<br/>

## 💡 Result


**최종 제출 모델**

| Model | Dice Score |
| --- | --- |
| FCN_ResNet101 | 0.9605 |
| HRNetV2_W48 | 0.9707 |
| DeepLabPlus_HRNet | 0.9699 |
| MANet | 0.9510 |
| Ensemble | 0.9721 |

### 리더보드 결과

Public : 0.9721 -> Private : 0.9728

![Untitled](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-10/assets/103094730/28012864-9d85-4cc8-8abd-c9f3cde5d878)

<br/>

## ❓ How to use


**Install Requirements**

```python
pip install -r requirement.txt
```

**train**

```python
python train.py --seed {seed} --epochs {epochs} --batch_size {batch_size} --wandb {1:save, 0:not save}} --lr {learning rate} --val_every {random int} --saved_dir {save directory} --model {model name} --loss {BCE rate, Dice rate, IoU rate}
```

**inference**

```python
python inference.py/inference_kfold.py --saved_dir {save directory} --model {model name}
```
