# DML_for-personal-study **[proceeding]**
still unfinished and in progress
## pytorch implementation
This repository is the unofficial implementation of :   
[Deep Mutual Learning (Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu)](https://arxiv.org/pdf/1706.00384.pdf)   

### To do
- more student networks (now 2 networks)   
- implement other model not only Resnet but also MobileNet, InceptionV1, WRN_28_10
- experimental environment setting and comparison of paper results

### Experimental setting
1. CIFAR 100
    - epochs : 200
    - batch size : 64
    - optimizer : 
      - SGD with Nesterov momentum
      - initial learning rate = 0.1
      - momentum = 0.9
      - The learning rate dropped by 0.1 every 60 epochs (step=60, gamma=0.1)
    - augmentation
      -  horizontal flips
      -  random crops : padding=4, padding_mode='reflect'

### Loss
<img width="743" alt="Deep Mutual Learning (DML) schematic" src="https://user-images.githubusercontent.com/87313780/130558874-4d072008-a703-45d3-8a76-af216dd8195b.png">

<img width="310" alt="스크린샷 2021-08-24 오후 2 08 39" src="https://user-images.githubusercontent.com/87313780/130559283-024df8f1-8cd0-4a33-adcc-4cc0b94665ec.png">
<img width="351" alt="스크린샷 2021-08-24 오후 2 38 45" src="https://user-images.githubusercontent.com/87313780/130561940-c80e8801-ce39-4bef-91d3-6b0e207b5f63.png">   

     In pytorch : L_c1=nn.CrossEntropyLoss(z1,label)
<img width="216" alt="스크린샷 2021-08-24 오후 2 48 50" src="https://user-images.githubusercontent.com/87313780/130562986-1df38555-edf8-4f76-a097-c977ef4e4362.png">
<img width="362" alt="스크린샷 2021-08-24 오후 2 04 38" src="https://user-images.githubusercontent.com/87313780/130561004-04010172-f9bd-42b1-a28d-4a950b957038.png">
<img width="444" alt="스크린샷 2021-08-24 오후 2 03 16" src="https://user-images.githubusercontent.com/87313780/130560907-1653fe53-561d-4d67-952d-78c62dcf8d77.png">

     In pytorch : D_kd(p2||p1)=nn.KLDivLoss(F.log_softmax(z1),F.softmax(z2))
  


### Result
|Network Types|Network Types|Ind.|Ind.|DML|DML|DML-Ind.|DML-Ind.|
|:-----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|
|Net1|Net2|Net1|Net2|Net1|Net2|Net1|Net2|
|Resnet-32|Resnet-32|71.18|71.18|71.25|71.24|0.07|0.06|
|Resnet-32|WRN_28_10|71.18|79.89|71.40|79.85|0.22|-0.04|

##### Top-1 accuracy (%) on the CIFAR-100 dataset

*It seems that this implementation doesn't properly yet*
+) I haven't completely separated the graphs of the two networks, and I'm experimenting with them again.

### Usage
1. clone repository `git clone 'https://github.com/pilsHan/DML_for-personal-study.git'`
2. requirements : pytorch and torchvision (can be run with colab)
3. run main.py `python main.py --num_workers 2`

### Reference  
https://github.com/chxy95/Deep-Mutual-Learning  
https://github.com/meliketoy/wide-resnet.pytorch    
https://arxiv.org/pdf/1706.00384.pdf
