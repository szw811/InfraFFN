# MFN

## Train
Download complete training dataset in [site](https://pan.baidu.com/s/1OTmUT6bswoNyshrWS9rXVQ) Code：cuek.
***
Run 
```
  python train_sr.py --opt=options/train_mfnet.json
```

## Evaluation
Download trained models and complete testing dataset in [site](https://pan.baidu.com/s/1OTmUT6bswoNyshrWS9rXVQ) Code：cuek.

Setting up the following directory structure:

    .
    ├── model_zoo                   
    |   ├──MFNet_x4.pth         # X4
    |   ├──MFNet_x3.pth         # X3
    |   |——MFNet_x2.pth          # X2 
    
***
Run 
```
  python my_main_test_mfnet.py
```

## Acknowledgement
Thanks to [Kai Zhang](https://scholar.google.com.hk/citations?user=0RycFIIAAAAJ&hl) for his work.
