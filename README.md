
# GLACIA: Instance–Aware Positional Reasoning for Glacial Lake Segmentation via Multimodal Large Language Model

This repository contains the **testing code** for **GLACIA**, which introduces a novel framework that integrates large language models with segmentation capabilities to produce both accurate glacial lake segmentation masks and corresponding spatial reasoning outputs. 



> 🚧 **Note**: The details of training instruction and full documentation will be made available soon. Please stay tuned.


## 📂 Dataset

For the GLake-Pos dataset, it is available in:

- Google Drive: [Download Dataset](https://drive.google.com/file/d/16_OF2GFwkgLSaNpkOL8MM_nBUyyQKVuU/view)
- [Hugging Face Dataset:]

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone 
cd GLACIA
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights (Will be updated soon)

```bash
bash download.sh
```
---

## 🏋️‍♂️ Training (Full code is not completely updated yet)

Use `train_lm.py` to train a segmentation model.


## 🧪 Inference

Use `infer.py` to perform inference on images.

### Example:

```bash
python infer.py \
```

**Note:** Please updated your checkpoint inside that folder to match with what your trained model

---


---

## Acknowledgements
This codebase is heavily borrowed from [PRS-Med](https://github.com/huyquoctrinh/PRS-Med)
