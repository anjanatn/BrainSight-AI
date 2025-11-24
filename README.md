
 ###  **BrainSight AI**

**Intelligent Brain Abnormality Heatmap Analyzer**

BrainSight AI is an interactive Flask-based medical imaging demo that transforms any brain CT/MRI slice into **six clinically-inspired heatmaps** and provides heuristic **abnormality probability scores**.

Designed as a **recruiter-friendly portfolio project**, it highlights full-stack AI development, computer vision reasoning, and medical imaging visualization.

---

## Features

###  **1. CT/MRI Upload Interface**

Upload any PNG/JPG/TIFF brain slice directly in the browser.

###  **2. Six Abnormality Heatmaps**

The system generates the following visualizations:

| Abnormality                      | Description                                    |
| -------------------------------- | ---------------------------------------------- |
|  **Hemorrhage Heatmap**        | Highlights bright-intensity hyperdense regions |
|  **Tumor Heatmap**             | Emphasizes central high-density patterns       |
|  **Lesion Map**                | Edge-sensitive structural disturbances         |
|  **Atrophy Map**               | Diffuse low-density regions                    |
|  **Volume Proxy Map**          | 3D-inspired intensity field                    |
|  **Composite 6-panel Figure** | Professional radiology-style layout            |

###  **3. Abnormality Probability Scores**

Each map generates an approximate probability (0–100%) indicating likelihood of an abnormality.

###  **4. Clean Medical-Style Visual Output**

All results are rendered into intuitive heatmaps with “inferno” color maps & grayscale baselines.

###  **5. Plug-and-Play Model Integration**

The synthetic inference engine can be replaced with your:

✔ MONAI model
✔ UNet segmentation model
✔ Any PyTorch classifier


##  Tech Stack

**Backend**

* Python 3.11+
* Flask
* NumPy
* SciPy (optional)
* Matplotlib
* Pillow

**Frontend**

* HTML5, CSS3
* Responsive diagnostic layout

**Optional ML**

* MONAI UNet
* PyTorch inference adapter

---

##  Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/BrainSight-AI.git
cd BrainSight-AI
```

Create environment:

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

##  Usage

### Mode 1 — Upload your CT/MRI slice

* Choose any `.png / .jpg / .jpeg / .tif / .tiff` file
* Click **Predict**
* View heatmaps + probabilities

### Mode 2 — Demo Mode

Click **Use Demo Image** to automatically run inference on the included sample image.

---

##  Project Structure

```
BrainSight-AI/
│ app.py
│ requirements.txt
│ README.md
│ LICENSE
│ .gitignore
├── templates/
│     upload.html
│     result.html
├── static/
│     style.css
│     results/
├── demo/
│     op.PNG



