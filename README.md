# MFN

## Train
Download complete training dataset in this [site](https://pan.baidu.com/s/1Q-Y6YO9Zh7H1MDM7CmRXeQ). Code：w33i.
***
Run 
```
  python train_sr.py --opt=options/train_infraffn.json
```

## Evaluation
Download trained models and complete testing dataset in thie [site](https://pan.baidu.com/s/1Q-Y6YO9Zh7H1MDM7CmRXeQ). Code：w33i.

Setting up the following directory structure:

    .
    ├── model_zoo                   
    |   ├──InfraFFN_x4.pth         # X4
    |   ├──InfraFFN_x3.pth         # X3
    |   |——InfraFFN_x2.pth          # X2 
    
***
Run 
```
  python my_main_test_infraffn.py
```

## Acknowledgement
Thanks to [Kai Zhang](https://scholar.google.com.hk/citations?user=0RycFIIAAAAJ&hl) for his work.
