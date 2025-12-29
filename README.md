# MSTCT - Colab Setup

This repository contains a Google Colab notebook to set up and run the MSTCT (Multi-Scale Temporal Convolutional Transformer) (https://github.com/dairui01/MS-TCT) project in a Colab environment.

MSTCT is a project focused on **multi-scale temporal feature extraction**, typically used for processing video sequences, extracting temporal embeddings, and performing inference using pre-trained MSTCT models.

---

## Contents

- `MSTCT.ipynb` (or your notebook name):  
  A Google Colab notebook that sets up the environment, installs dependencies, loads the MSTCT model repository, and runs inference on sample video data.

---

## Setup Instructions

To use the notebook:

1. Open the notebook in **Google Colab**  
   👉 https://colab.research.google.com/drive/1IfDJjycOXuyrrJPYwLx-NXuSYleFfc8b?usp=sharing

2. Run all cells **sequentially from top to bottom**.

3. The notebook will automatically:
   - Check Python version and environment compatibility  
   - Install all required dependencies (PyTorch, torchvision, OpenCV, etc.)  
   - Clone the official **MS-TCT GitHub repository**  
   - Load pre-trained MSTCT model weights (if included)  
   - Process sample video frames  
   - Extract features or run inference depending on your setup  

---

## Requirements

This project is optimized for **Google Colab**, but can also run locally in a Jupyter Notebook environment with the following:

### Software Requirements
- Python 3.8+
- PyTorch (GPU recommended)
- torchvision
- OpenCV
- NumPy
- Git

### Optional (but recommended)
- GPU runtime (CUDA)  
  In Colab:  
  **Runtime → Change runtime type → GPU**

---

## Setting up Video Input

The MSTCT notebook supports multiple ways of loading video data:

1. Upload a video directly into Colab  
2. Mount Google Drive and load videos from your folders  
3. Use any sample videos included in the notebook  

Ensure the video is in a supported format (`.mp4`, `.avi`, etc.).

---

## Model Loading & Inference

The Colab notebook will:

- Clone the official MSTCT GitHub repository:

  ```bash
  !git clone https://github.com/dairui01/MS-TCT.git
  ```

- Navigate into the project directory  
- Install required Python modules  
- Load the MSTCT model for inference  
- Process uploaded video frames  
- Generate temporal embeddings or prediction outputs  
  *(depending on your implementation)*

---

## Output

After running the notebook, outputs such as:

- Extracted temporal features  
- Processed frame sequences  
- Prediction logs  
- Saved `.npy` output files  

will be available in the Colab environment under:

```
/content/
```

Download them manually if needed.

---

## License

This project uses content and code from the original  
**MS-TCT repository (https://github.com/dairui01/MS-TCT)**.  
Please refer to their repository for licensing details.

---
