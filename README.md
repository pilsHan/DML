# DML_for-personal-study **[proceeding]**
still unfinished and in progress
## pytorch implementation    
[Deep Mutual Learning (Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu)](https://arxiv.org/pdf/1706.00384.pdf)   

### To do
- more student networks (now 2 networks)   
- implement other model not only Resnet but also MobileNet, InceptionV1, 28_10_WRN
- experimental environment setting and comparison of paper results

### Result
|Network Types|Network Types|Independent|Independent|DML|DML|
|----|----|----|----|----|----|
|Net1|Net2|Net1|Net2|Net1|Net2|
|Resnet-32|Resnet-32|69.89|69.89|Net1|Net2|

### Usage
1. clone repository `git clone 'https://github.com/pilsHan/DML_for-personal-study.git'`
2. requirements : pytorch and torchvision (can be run with colab)
3. run main.py `python main.py --num_workers 2`

### Reference  
https://github.com/chxy95/Deep-Mutual-Learning
