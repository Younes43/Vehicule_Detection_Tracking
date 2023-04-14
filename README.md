# Vehicle Detection and Tracking using DeepSort Algorithm and YOLOv4

This GitHub repository contains code for training and testing a vehicle detection and tracking system using DeepSort algorithm and YOLOv4 object detector. The repository provides step-by-step instructions on how to prepare the data, train the models, and test the models for vehicle detection and tracking.

## Modules Diagram 
![Detection and Tracking](detection-tracking-diagram2.png)

## Data Preparation

### DeepSort Data Preparation

1. Activate the virtual environment for YOLOv4 GPU.
```bash
conda activate yolov4-gpu
```

2. Navigate to the detrac_tools directory.
```bash
cd ~/Vehicule_Detection_Tracking/Multi-Camera-Live-Object-Tracking/detrac_tools
```
3. Run the crop_dataset.py script to prepare the DeepSort dataset.
```bash
python crop_dataset.py --DETRAC_images ../../../data/Insight-MVT_Annotation_Train/ --DETRAC_annots ../../../data/DETRAC-Train-Annotations-XML-v3/ --output_train ./Detrac_deepsort_09_09/bouding_box_train/ --occlusion_threshold=0.9 --truncation_threshold=0.9 --occurrences=10
```

## DeepSort Training
1. Navigate to the cosine_metric_learning directory.
```bash
cd ~/Vehicule_Detection_Tracking/Multi-Camera-Live-Object-Tracking/cosine_metric_learning/
```
2. Run the train_market1501.py script to train the DeepSort model.
```bash
python train_market1501.py  --dataset_dir=../Multi-Camera-Live-Object-Tracking/detrac_tools/Detrac_deepsort_09_09/  --loss_mode=cosine-softmax  --log_dir=./output/Detrac_09_09/  --run_id=cosine-softmax
```
3. Open a new terminal and start TensorBoard for visualizing the training progress.
```bash
tensorboard --logdir ./output/Detrac_09_09/cosine-softmax/ --host=0.0.0.0 --port 6006
```
4. Open a web browser and go to http://localhost:6006/ to view the TensorBoard.

### DeepSort Model Evaluation
1. Open a new terminal and run the evaluation script for the DeepSort model.
```bash
CUDA_VISIBLE_DEVICES="" python train_market1501.py  --mode=eval  --dataset_dir=../Multi-Camera-Live-Object-Tracking/detrac_tools/Detrac_deepsort_09_09/  --loss_mode=cosine-softmax  --log_dir=./output/Detrac_09_09/  --run_id=cosine-softmax  --eval_log_dir=./eval_output/Detrac_09_09
```

2. Open a new terminal and start TensorBoard for visualizing the evaluation results.
```bash
tensorboard --logdir ./eval_output/Detrac_09_09/cosine-softmax/ --host=0.0.0.0 --port 6007
```
3. Open a web browser and go to http://localhost:6007/ to view the TensorBoard.

## YOLOv4 Preparation

1. Navigate to the detrac_tools directory.
```bash
cd ~/Vehicule_Detection_Tracking/Multi-Camera-Live-Object-Tracking/detrac_tools
```
2. Run the detrac_to_yolo.py script to prepare the YOLOv4 dataset.
```bash
python detrac_to_yolo.py --DETRAC_images ../../../data/Insight-MVT_Annotation_Train/ --DETRAC_annots ../../../data/DETRAC-Train-Annotations-XML-v3/ --output_train ./DETRAC_YOLO_training_09_09/ --occlusion_threshold=0.9 --truncation_threshold=0.9
```
3. Copy the produced files into Yolo directory 
```bash
cp train.txt valid.txt detrac_classes.names DETRAC.data ~/Vehicule_Detection_Tracking/darknet/data/
cp yolov4-obj.cfg yolov4.conv.137 ~/Vehicule_Detection_Tracking/darknet/cfg/
```
## YOLOv4 Training

4. Navigate to the darknet directory
```bash
cd ~/Vehicule_Detection_Tracking/darknet
```
5. Compile darknet by runing the following command

make

6. Launch the training of YOLOv4 using the command 
```bash
./darknet detector train data/DETRAC.data cfg/yolov4-obj.cfg cfg/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map
```
7. On a browser open the url http://localhost:8090/ or http://127.0.0.1:8090/ to visualize the Loss Curve


## Testing yolov4+deepsort




## References and Credits

The following GitHub repositories were used in the development of this project:

- [DETRAC Tools](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking) by LeonLok
- [DeepSort: Cosine Metric Learning](https://github.com/nwojke/cosine_metric_learning) by Nwojke
- [YOLOv4 Darknet](https://github.com/AlexeyAB/darknet) by AlexeyAB
- [Yolov4 & Deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) by Atwine


We would like to thank the authors of these repositories for their contributions to the field of computer vision and object tracking, which greatly aided the development of our project. 

Please refer to the respective repositories for more information and details on how to use their code. 


