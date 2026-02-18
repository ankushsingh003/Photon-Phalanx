# ☀️ Photon Phalanx: Solar Intelligence

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

**Photon Phalanx** is an advanced AI-driven diagnostic system for solar asset management. Utilizing state-of-the-art Computer Vision (Transfer Learning with MobileNetV2), it identifies and classifies defects in solar panels to optimize energy yield and streamline maintenance.

---

## 🚀 Key Features

*   **Multi-Class Defect Detection**: Classifies imagery into 6 distinct categories:
    *   🛡️ **Physical Damage**: Cracks and structural hits.
    *   ⚡ **Electrical Damage**: Burn marks and circuit faults.
Classifies imagery into 6 distinct categories:
    *   ❄️ **Snow Coverage**: Obstruction analysis.
    *   🐦 **Bird-Drop**: Localized fouling detection.
    *   🌪️ **Dusty**: Efficiency loss assessment.
    *   ✅ **Clean**: Operational excellence verification.
*   **Edge Inference UI**: A premium, high-performance Streamlit interface for field inspections.
*   **Real-time Intelligence**: Confidence scoring and automated maintenance alerts.
*   **Live Training Stream**: Integrated log monitoring for model optimization transparency.

---

## 🛠️ Technology Stack

*   **Engine**: TensorFlow / Keras (MobileNetV2 & EfficientNetB0 backbones)
*   **Frontend**: Streamlit (with custom HSL-themed CSS)
*   **Data Ops**: NumPy, PIL, Pandas
*   **Containerization**: Docker (optimized python-slim builds)

---

## 📦 Installation & Setup

### Local Setup
1. **Clone the Phalanx:**
   ```bash
   git clone https://github.com/ankushsingh003/Photon-Phalanx.git
   cd Photon-Phalanx
   ```

2. **Environment Configuration:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Ignite the UI:**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment
Build and run the containerized engine:
```bash
docker build -t photon-phalanx .
docker run -p 8501:8501 photon-phalanx
```

---

## 🧠 Model Architecture

The core uses **Transfer Learning** on ImageNet-pre-trained weights.
- **Input Resolution**: 224x224x3 (RGB)
- **Bottleneck**: Global Average Pooling
- **Head**: 128-node Dense (ReLU) -> Dropout (0.5) -> Softmax Output
- **Optimizer**: Adam (Learning Rate: 1e-3)

---

## 🌍 Sustainable Impact
By automating defect detection, **Photon Phalanx** helps reduce the "energy payback time" of solar installations and ensures maximum ROI for sustainable energy infrastructures.

---
*Built for Sustainable Energy | Photon Phalanx Pro v1.2*
