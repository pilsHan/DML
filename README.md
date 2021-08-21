# DML_for-personal-study **[proceeding]**
still unfinished and in progress
## pytorch implementation    
[Deep Mutual Learning (Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu)](https://arxiv.org/pdf/1706.00384.pdf)   

### To do
- more student networks (now 2 networks)   
- implement other model not only Resnet but also MobileNet, InceptionV1, 28_10_WRN
- experimental environment setting and comparison of paper results

###Experimental setting
- CIFAR 100
  - epochs : 200
  - batch size : 64
  - optimizer : 
    - SGD with Nesterov momentum
    - initial learning rate = 0.1
    - momentum = 0.9
    - The learning rate dropped by 0.1 every 60 epochs (step=60, gamma=0.1)
  - augmentation
    -  horizontal flips
    -  random crops : padding=4
### Result
|Network Types|Network Types|Independent|Independent|DML|DML|
|----|----|----|----|----|----|
|Net1|Net2|Net1|Net2|Net1|Net2|
|Resnet-32|Resnet-32|69.89|69.89|69.95|70.11|

It is thought that the accuracy is within the error range, and it seems that the implementation is not done properly yet.

### Usage
1. clone repository `git clone 'https://github.com/pilsHan/DML_for-personal-study.git'`
2. requirements : pytorch and torchvision (can be run with colab)
3. run main.py `python main.py --num_workers 2`

### Reference  
https://github.com/chxy95/Deep-Mutual-Learning
