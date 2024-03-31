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

The XAI Analyis runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```

5. XAI Methods

``` xai_methods, x_batch, y_batch, s_batch = get_attributions(**Parameters)```

6. XAI Metrics

```df_metric, df_metric_ranks = get_metrics(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/Peeples-Lab/XAI_Analysis

└── root dir
	├── demo.py   //Run this. Main demo file.
	├── Demo_Parameters.py // Parameters file for demo.
	├── Prepare_Data.py  // Load data for demo file.
	├── View_Results.py // Run this after demo to view saved results.
    	├── Datasets
		├── Get_transform.py // Transforms applied on test, train, val dataset splits
		├── loader.py // MSTAR dataset loader object
		├── mstar.py // MSTAR metadata
		├── preprocess.py // Pytorch Transforms applied to numpy files
		├── Pytorch_Datasets.py // Return Index for Pytorch datasets
		├── Pytorch_Datasets_Names.py // Return names of classes in each dataset
	└── Utils  //utility functions
		├── Compute_FDR.py  // Compute Fisher Score
		├── Confusion_mats.py  // Create and plot confusion matrix.
		├── Focalnet.py // Implementation of Focal Modulation Networks
    		├── Generating_Learning_Curves.py  // Plot training and validation accuracy and error measures.
    		├── Generate_TSNE_visual.py  // Create TSNE visual for results.
    		├── Network_functions.py  // Contains functions to initialize, train, and test model. 
    		├── pytorchtools.py // Function for early stopping.
    		├── Save_Results.py  // Save results from demo script.
	└── XAI_Methods  // XAI functions
		├── get_attributes.py // Compute attributions of XAI methods
		├── get_explanations.py // Calculate XAI metrics
		├── get_spyderplot.py // Create and plot spyderplot
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2023 A. Mohan and J. Peeples. All rights reserved.

## <a name="CitingQuantitative"></a>Citing Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

A. Mohan and J. Peeples, "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification,"  in 2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS, pp. 950-953. IEEE, 2023

**BibTex:**
```
@inproceedings{mohan2023quantitative,
  title={Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification},
  author={Mohan, Akshatha and Peeples, Joshua},
  booktitle={2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS},
  pages={950-953},
  year={2023},
  organization={IEEE}
}

```
