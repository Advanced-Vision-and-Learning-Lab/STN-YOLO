# STN-YOLO
**Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection**

_Yash Zambre, Ekdev Rajkitikul, Akshatha Mohan and Joshua Peeples_

![STN-YOLO/ultralytics/pipeline.png](https://github.com/Advanced-Vision-and-Learning-Lab/STN-YOLO/blob/main/ultralytics/architecture.png)

[`Zendo`](https://zenodo.org/records/10905984). https://zenodo.org/records/10905984
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10905984.svg)](https://zenodo.org/records/10905984)

[`arXiv`](https://arxiv.org/abs/2407.21652)

[`BibTeX`](https://github.com/Advanced-Vision-and-Learning-Lab/STN-YOLO/blob/main/README.md#citing-spatial-transformer-network-you-only-look-once-stn-yolo-for-improved-object-detection)


Note: If this code is used, cite it: Yash Zambre, Joshua Peeples, Akshatha Mohan and Ekdev Rajkitikul. 



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
[Dataset](https://drive.google.com/drive/folders/17IfXOsj0zTceSetX8syem751PVlcloXz?usp=drive_link)

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

This product is Copyright (c) 2023 Yash Zambre and Ekdev Rajkitkul and Akshatha Mohan and Joshua Peeples. All rights reserved.

## <a name="CitingSTN-YOLO"></a>Citing Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

Yash Zambre and Ekdev Rajkitkul and Akshatha Mohan and Joshua Peeples, "Spatial Transformer Network You Only Look Once (STN-YOLO) for Improved Object Detection,"  in Review.

**BibTex:**
```
@misc{zambre2024spatialtransformernetworkyolo,
      title={Spatial Transformer Network YOLO Model for Agricultural Object Detection}, 
      author={Yash Zambre and Ekdev Rajkitkul and Akshatha Mohan and Joshua Peeples},
      year={2024},
      eprint={2407.21652},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.21652}, 
}

```
