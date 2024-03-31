# STN-YOLO
**SPATIAL TRANSFORMER NETWORK YOU ONLY LOOK ONCE (STN-YOLO) FOR IMPROVED OBJECT DETECTION**

_Yash Zambre and Joshua Peeples_

![STN-YOLO/ultralytics/pipeline.png](https://github.com/Advanced-Vision-and-Learning-Lab/STN-YOLO/blob/main/ultralytics/pipeline.png)

Note: If this code is used, cite it: Yash Zambre and Joshua Peeples. 

TBD for paper links 

In this repository, we provide the paper and code for the "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification."

## Installation Prerequisites

This code uses python, pytorch and YOLO model. 
Please use [`Pytorch's website`](https://pytorch.org/get-started/locally/) to download necessary packages.
[YOLO](https://docs.ultralytics.com/modes/) is used for the object detection model. Please follow the instructions on each website to download the modules.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

## Main Functions

The STN-YOLO runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(data, epochs, batch, device, pretrained, etc..)```

2. Prepare dataset(s) for model

 ```The dataset should be in YOLOV8 format```

3. Train model 

```model.train(data, epochs, batch, device, pretrained, etc..)```

4. Test model

```model.test(data, epochs, batch, device, pretrained, etc..)```



## Inventory

```
https://github.com/Peeples-Lab/XAI_Analysis

└── root dir
	├── demo.py   //Run this. Main demo file.
    	├── Ultralytics
		├── cfg/datasets.yaml
		├── cfg/models/models.yaml (change for the addition of STN here)
		├── data (does the data loading)
		├── models (All the models in the Ultralytics framework are present here)
		├── modules/block.py (The STN is defined here with its localization network.)
	└── Utils  //utility functions
    		├── Network_functions.py  // Contains functions to initialize, train, and test model. 
    	
	
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2023 Y. Zambre and J. Peeples. All rights reserved.


tbd
