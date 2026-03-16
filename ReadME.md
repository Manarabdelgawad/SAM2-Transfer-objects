🎨 SAM2 Object Transfer API + Streamlit Frontend


**Extract any object from a source image and seamlessly place it on a new background** using **Segment Anything Model 2 (SAM2)**.

Built with **FastAPI** (backend) + **Streamlit** (beautiful frontend). One-click object transfer with automatic segmentation, mask refinement, scaling, and centering.

---

##  Features

- Automatic object segmentation with **SAM2.1** (b model)
- Smart mask selection 
- Edge refinement 
- Auto-scaling & perfect centering on background
- High-quality PNG output
- **Streamlit UI** included

---


## 📁 Project Structure
sam2-object-transfer/
├── main.py                 # FastAPI backend

├── streamlit.py            # Streamlit frontend

├── sam2.1_b.pt             # Model weights (download once)

├── requirements.txt

├── README.md

└── inputs/

└── outputs/

🚀 Installation

 1. Clone & Setup
 2. Create Virtual Environment and activate 
```bash
python -m venv venv
```
 3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Download SAM2 Model
 download sam2.1_b.pt from:
 ```bash
https://github.com/ultralytics/ultralytics/releases
```
### Run
 Backend + Frontend Together
 1. fastapi
```bash
uvicorn main:app --reload --port 8000
```
 2. streamlit in anthor terminal
```bash
streamlit run app.py
```
 3. Open browser → http://localhost:8501
 4. to open api documenation 
```bash
http://127.0.0.1:8000/docs
```
## API Endpoints
```bash
Method, Route, Description
GET,      /   , Welcome message
POST,   /transfer,Main endpoint (source + background)
 ```


## How It Works 

SAM2 segments the source image

Picks the  valid object 

Refines mask (dilation + soft edges)

Scales object to fit background nicely

Composites with perfect centering
    


