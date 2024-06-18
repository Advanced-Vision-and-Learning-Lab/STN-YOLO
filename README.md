# STN-YOLO
**Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection**

_Yash Zambre, Joshua Peeples, Akshatha Mohan and Ekdev Rajkitikul_

![STN-YOLO/ultralytics/pipeline.png](https://github.com/Advanced-Vision-and-Learning-Lab/STN-YOLO/blob/main/ultralytics/architecture.png)

[`Zendo`](https://zenodo.org/records/10905984). https://zenodo.org/records/10905984
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10905984.svg)](https://zenodo.org/records/10905984)

[`IEEE Xplore (name)`](tbd)

[`arXiv`](tbd)

[`BibTeX`](tbd)


Note: If this code is used, cite it: Yash Zambre, Joshua Peeples, Akshatha Mohan and Ekdev Rajkitikul. 

TBD for paper links 

In this repository, we provide the code for the "Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection"


## Installation Prerequisites

This code uses python, pytorch and YOLO model. 
Please use [`Pytorch's website`](https://pytorch.org/get-started/locally/) to download necessary packages.
[YOLO](https://docs.ultralytics.com/modes/) is used for the object detection model and the framework used is Ultralytics. Please follow the instructions on each website to download the modules.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

## Main Functions

The STN-YOLO runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(data, epochs, batch, device, pretrained, etc..)```

2. Prepare dataset(s) for model

 ```The dataset should be in YOLOV8 format```
 A sample dataset thet we used for this project is given here, this dataset is an inhouse dataset grown in the Texas A&M Agrilife facility - College Station, TX 
 https://drive.google.com/file/d/13d-PxhwRYguQ0JeKA3EuY9sN3dnhAXFB/view?usp=drive_link

3. Train model 

```model.train(data, epochs, batch, device, pretrained, etc..)```

4. Test model

```model.test(data, epochs, batch, device, pretrained, etc..)```



## Inventory

```
https://github.com/Advanced-Vision-and-Learning-Lab/STN-YOLO

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

This source code is licensed under the license found in the [`LICENSE`](LICENSE.txt) 
file in the root directory of this source tree.

This product is Copyright (c) 2023 Y. Zambre and J. Peeples. All rights reserved.

## <a name="CitingSTN-YOLO"></a>Citing Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

Yash Zambre, Joshua Peeples, Akshatha Mohan and Ekdev Rajkitikul, "Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection,"  in Review.

**BibTex:**
```


```
