# Graphical_User_Interface_Medics
To integate front/back end for clinicians/medical professionals,,,


#  Prostate Segmentation & Detection Tool

An interactive web application for running deep learning-based **prostate segmentation and detection** tasks on medical images using Streamlit.

---
#folders structure

├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── saved_predictions/      # Auto-created directory for saved masks
└── README.md               # This file


##  Features

- Upload medical scans (MRI, CT, TRUS)
- Choose from dummy models: GAN, YOLOv8, Diffusion
- Perform detection, segmentation, or both
- Display and download predicted masks
- Optional ground truth comparison with IoU and Dice metrics
- Save masks locally with timestamp
- Clean, responsive user interface

---

##  Tech Stack

- **Frontend & Interface**: Streamlit
- **Backend Framework**: PyTorch (placeholder models)
- **Image Processing**: OpenCV & PIL
- **Metrics**: Scikit-learn (IoU, Dice)
- **Language**: Python 3.x

---

##  Installation

1. **Clone the repository**


##Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


##install_dependencies
pip install -r requirements.txt

to run the app, and generate ip
streamlit run app.py



👤 Author
Syed Ibrar Hussain
Ph.D. in Mathematics and Computer Science
Researcher in AI-based Medical Imaging
(In progress....)
For scientific contributions, refer to my ResearchGate profile
git clone https://github.com/yourusername/prostate-segmentation-streamlit.git
cd prostate-segmentation-streamlit
