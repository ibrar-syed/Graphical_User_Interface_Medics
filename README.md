# 🩺 Image Segmentation & Detection Web Application

This Streamlit-powered application offers an interactive platform for analyzing medical images using detection and segmentation tasks. Users can upload medical images (e.g., MRI, CT, TRUS), select a model architecture (currently placeholder), perform tasks, and visualize results. While this version contains dummy logic for modeling, it is structured for integration with actual deep learning models,  **In Progress,...**

---

## 🚀 Features

- Upload MRI, CT, or TRUS images in `.png`, `.jpeg`, `.tiff`, or `.nifti` format
- Select model type (GAN, YOLOv8, Diffusion - placeholders)
- Choose task: Detection, Segmentation, or Both
- View overlays for segmentation and bounding boxes
- Upload ground truth masks to compute evaluation metrics (IoU, Dice)
- Download predicted segmentation masks
- Fully interactive web-based experience via Streamlit

---

## 📁 Project Structure

```
.
├── streamlit_app.py              # Main Streamlit UI script
├── saved_predictions/           # Folder to save segmentation outputs
└── README.md
```

---

## 📦 Installation

Install required dependencies using pip:

```bash
pip install streamlit torch torchvision pillow opencv-python-headless scikit-learn
```

---

## 🧠 How to Run

Launch the Streamlit interface using the following command:

```bash
streamlit run streamlit_app.py
```

---

## 🔍 Workflow

### Step 1: Upload Files
- Upload a medical image from your local machine
- Optionally upload a ground truth mask to compute metrics

### Step 2: Select Parameters
- Choose a model type (placeholder options provided)
- Choose your task (Detection / Segmentation / Both)
- Select the image modality (MRI / CT / TRUS)

### Step 3: Run Task
- Click "▶️ Run Task" to start the process
- Outputs are rendered side-by-side (original, prediction)
- Download the predicted mask (PNG format)
- View metrics if a ground truth mask is uploaded

---

## 📊 Evaluation Metrics

If a ground truth mask is uploaded, the app calculates:

| Metric | Description |
|--------|-------------|
| **IoU** (Jaccard) | Measures overlap between predicted and ground truth mask |
| **Dice Score** | Measures similarity between predicted and ground truth regions |

---

## 🧪 Output Example

- 🔲 Bounding box visualization for detection
- 🟥 Overlayed mask output for segmentation
- 📈 Score summary in metric section
- 📥 Downloadable mask result

---

## 🧩 Extending the App

To integrate your trained models, update the following placeholder functions:

```python
def load_model(model_name): ...
def generate_mask(image, model): ...
def detect_regions(image, model): ...
```

These can be customized to include TensorFlow, PyTorch, or ONNX inference pipelines.

---

## ⚠️ Limitations

- Models used are placeholders (i.e., no learning or inference)
- Segmentation and detection results are simulated for demo purposes
- NIfTI file format requires additional packages like `nibabel`

---

## 👨‍💻 Author

Developed as a modular prototype for interactive medical image processing applications.

---

## 💡 License

This project is intended for research and educational use. For clinical deployment, consult relevant medical regulations and validation standards.
