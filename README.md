# Defence-Relevant Object Detection in Satellite Imagery Using Deep Learning

This project presents a deep learning-based object detection system for identifying defence-relevant objects in satellite imagery. The model is trained using the YOLO11s-OBB (Oriented Bounding Box) architecture on a filtered version of the DIOR-R dataset containing three important classes: **airplane, ship, and vehicle**.

The use of Oriented Bounding Boxes enables accurate localization of rotated objects commonly found in aerial and satellite images. The trained model achieves high detection performance and is integrated with a Streamlit web application for interactive inference on uploaded satellite images.

## Features

* YOLO11s-OBB based object detection
* Detection of airplanes, ships, and vehicles
* Oriented Bounding Box (OBB) support
* Satellite and remote sensing image analysis
* Streamlit-based web interface
* High detection accuracy (mAP50: 92.9%)

## Applications

* Border surveillance
* Maritime monitoring
* Airfield observation
* Strategic asset detection
* Defence reconnaissance and intelligence

## Technologies Used

* Python
* YOLO11s-OBB (Ultralytics)
* OpenCV
* Streamlit
* Google Colab
* DIOR-R Dataset
