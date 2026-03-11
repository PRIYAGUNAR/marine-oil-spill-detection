# marine-oil-spill-detection

This project detects marine oil spills from uploaded images using a trained PyTorch U-Net model. The current deployment target is a Streamlit web app.

## Project files

- `app.py`: Streamlit web application for upload and prediction
- `api.py`: FastAPI inference API
- `best_model_1 (1).pth`: trained model checkpoint
- `requirements.txt`: Python dependencies

## Run locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```powershell
python -m streamlit run app.py --server.port 8502
```

Open the app in your browser:

```text
http://localhost:8502
```

## Deploy on Streamlit Community Cloud

1. Push the latest code to GitHub.
2. Open https://share.streamlit.io/.
3. Click `New app`.
4. Select the repository `PRIYAGUNAR/marine-oil-spill-detection`.
5. Select branch `main`.
6. Set the main file path to `app.py`.
7. Click `Deploy`.

## Model details

- Model architecture: Legacy U-Net
- Framework: PyTorch
- Input preprocessing: grayscale conversion and resize to `512 x 512`
- Output: predicted oil-spill mask and oil ratio

## Report Preparation Team

- Ammar Shibli
- B V Vedamurthi






##  Deployment and Execution Guide

###  Clone the Repository

git clone https://github.com/PRIYAGUNAR/marine-oil-spill-detection.git
cd marine-oil-spill-detection

   Before running the project install the reqiured Python packages 
   
     pip install -r requirements.txt

## Download the oil spill dataset and place it inside the "data" folder.

* Example
marine-oil-spill-detection/
│
├── data/
│   ├── train
│   ├── test
│
├── notebooks/
├── models/
└── README.md

## Execute the training or detection notebook/script:

jupyter notebook

Open the project notebook and run all cells to train and test the model


## The system classifies satellite image pixels into:

  Oil Spill & Clean Water
  
