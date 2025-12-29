# MSTCT - Colab Setup & Guide

Hey! This is a quick guide to set up and run **MSTCT (Multi-Scale Temporal Convolutional Transformer)** in **Google Colab**. MSTCT is awesome for extracting **temporal features from video sequences** and running inference with pre-trained models.

Original project: [MS-TCT GitHub](https://github.com/dairui01/MS-TCT)

---

## 🚀 Quick Start

1. Open the Colab notebook here:  
👉 [MSTCT Colab Notebook](https://colab.research.google.com/drive/1IfDJjycOXuyrrJPYwLx-NXuSYleFfc8b?usp=sharing)

2. Run all the cells from top to bottom.

The notebook will automatically:  
- Check your Python version and GPU availability  
- Install all required libraries like PyTorch, torchvision, OpenCV, and NumPy  
- Clone the MSTCT repository  
- Load pre-trained MSTCT weights  
- Process video frames and generate temporal features or predictions

---

## ⚙️ Requirements

**Basic:**  
- Python 3.8+  
- PyTorch & torchvision  
- OpenCV  
- NumPy  
- Git

**Optional (highly recommended):**  
- GPU runtime in Colab → `Runtime → Change runtime type → GPU`  

> Using a GPU will make processing videos way faster, especially long ones.

---

## 🎥 Video Input

You can use your videos in multiple ways:  
1. Upload directly to Colab  
2. Mount your Google Drive and load videos from there  
3. Use sample videos provided in the notebook

Supported formats: `.mp4`, `.avi`, `.mov`

---

## 🛠 Model Setup & Inference

Here’s basically what the notebook does under the hood:

**Clone the repo and install dependencies:**

```bash
!git clone https://github.com/dairui01/MS-TCT.git
%cd MS-TCT
!pip install -r requirements.txt
