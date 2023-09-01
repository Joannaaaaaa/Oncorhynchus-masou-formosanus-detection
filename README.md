# Oncorhynchus-masou-formosanus-detection
偵測櫻花鉤吻鮭

利用自動化辨識及追蹤去計算櫻花鉤吻鮭的平均個數以及移動趨勢


## Environment

tracking_result:
Python version==3.7  

## Dataset

6000張（1000手動label 5000利用資料增強翻轉）:
網址

Data Augmentation:
run albumentation.py:

```
python albumentation.py
```

## Train

get the training weights

run train.py:

```
python train.py
```

Use weights to see the result

run detect_test.py:

```
python detect_test.py
```
tracking and identifing the fish by this code
run tracking_result.py:

```
python tracking_result.py
```

## result
result_video:
https://youtu.be/ph4j2CFaBL4?si=o6jOGF1SFQizl2JA

result image: 

<img src="https://github.com/Joannaaaaaa/Oncorhynchus-masou-formosanus-detection/assets/98182630/aba721a6-3a73-4553-b30c-02a09f7f3137" width="450">
