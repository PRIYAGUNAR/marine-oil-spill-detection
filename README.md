# marine-oil-spill-detection


## Deployable Inference API

This branch now includes a deployment-ready FastAPI service for oil-spill mask prediction.

### Start locally (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Check health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Run prediction on one image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?threshold=0.5&return_mask=true" -F "file=@D:/path/to/your/image.png"
```

### Deploy using Docker

Build image:

```powershell
docker build -t marine-oil-spill-api:latest .
```

Run container:

```powershell
docker run --rm -p 8000:8000 marine-oil-spill-api:latest
```

Then test:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Notes:
- The API loads checkpoint: best_model_1 (1).pth.
- Input images are converted to grayscale and resized to 512x512 before inference.
- The response includes oil_ratio and an optional PNG mask as base64.

### Cloud Deploy (Render)

This project includes render.yaml, so deploy with Render Blueprint.

1. Push this branch to GitHub.
2. Open: https://dashboard.render.com/blueprint/new
3. Connect your repository.
4. Render will detect render.yaml and create the web service.
5. After deploy, test:

```powershell
Invoke-RestMethod https://<your-render-service>.onrender.com/health
```

### One-click scripts (Windows)

Local deploy (install, start, health-check):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy-local.ps1
```

Real image prediction test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test-predict.ps1 -ImagePath test_inputs/lena.jpg
```

Prepare and push for Render deploy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy-render.ps1
```



## Report Preparation Team

The following members are responsible for preparing the project report for this project.

- Ammar Shibli
- B V Vedamurthi
