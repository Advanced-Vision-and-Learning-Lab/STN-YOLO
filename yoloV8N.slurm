#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobXAI       #Set the job name to "JobExample4"
#SBATCH --time=48:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=8                   #Request 1 task
#SBATCH --mem=9000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=yolov8nstn_modified.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2               #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=yashzambre@tamu.edu    #Send all emails to email_address 

#First Executable Line

cd /scratch/user/yashzambre/ultralytics; 
ml GCCcore/9.3.0 GCC/9.3.0 Python/3.8.2 CUDAcore/11.0.2 CUDA/11.0.2 cuDNN/8.0.5.39-CUDA-11.0.2
module load Anaconda3/2022;
source activate myenv1;
export MKL_THREADING_LAYER=GNU;
python demo_og.py --yaml '/scratch/user/yashzambre/ultralytics/augmentations/5/data.yaml' --model 'YOLOV8N' --num_epochs 100 --folder 'Saved_Models/augmentation_5_YOLO_100' 
