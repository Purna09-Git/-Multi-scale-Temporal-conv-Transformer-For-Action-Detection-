# MSTCT - Colab Setup & Guide

Hey! This is a quick guide to set up and run **MSTCT (Multi-Scale Temporal Convolutional Transformer)** in **Google Colab**. MSTCT is awesome for extracting **temporal features from video sequences** and running inference with pre-trained models.

Original project: [MS-TCT GitHub](https://github.com/dairui01/MS-TCT)

---


##  Project Demo

[![Watch the video](Video%20Activity%20Analysis.png)](https://drive.google.com/file/d/1wYmdFlScK1cmAHWWXWcELuFcSiLMt3FP/view?usp=drive_link)
<img width="1565" height="839" alt="image" src="https://github.com/user-attachments/assets/2a46201e-bce8-4be0-9f9a-d165c2598e48" />

> **Note:** Click the image above to watch the full demo on Google Drive.

##  Quick Start

1. Open the Colab notebook here:  
👉 [MSTCT Colab Notebook](https://colab.research.google.com/drive/1GTmWSqlaZPdvgDsY7ryEvct_DpAPBTqU?usp=sharing)

2. Run all the cells from top to bottom.

The notebook will automatically:  
- Check your Python version and GPU availability  
- Install all required libraries like PyTorch, torchvision, OpenCV, and NumPy  
- Clone the MSTCT repository  
- Load pre-trained MSTCT weights  
- Process video frames and generate temporal features or predictions

---

##  Requirements

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

## Video Input

You can use your videos in multiple ways:  
1. Upload directly to Colab  
2. Mount your Google Drive and load videos from there  
3. Use sample videos provided in the notebook

Supported formats: `.mp4`, `.avi`, `.mov`

---

## Model Setup & Inference

Here’s basically what the notebook does under the hood:

**Clone the repo and install dependencies:**

```bash
!git clone https://github.com/dairui01/MS-TCT.git
%cd MS-TCT
!pip install -r requirements.txt
```
Load the model and run inference:

```bash
from MSTCT.model import MSTCT

model = MSTCT(pretrained=True)  # loads pre-trained weights
```

Then you can:

Process video frames

Generate temporal embeddings

Save outputs (.npy files) in /content/

Pro tip: If you have multiple videos, batch processing will save you a lot of time.

## Outputs

After running the notebook, you’ll get:

Extracted temporal features

Processed frames

Prediction logs

Saved .npy files

Everything will be in /content/—you can download manually.

## Notes

This setup is made for Colab, but you can run it locally if your Python version and dependencies match.

Always try to use a GPU, especially for long or high-res videos.

You can tweak the notebook to save features or predictions in your preferred format.


