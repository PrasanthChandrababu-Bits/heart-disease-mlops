# Assignment Submission Steps

Run all commands from the repository root.

## 1. Start the environment

```bash
source .venv/bin/activate
export PATH="/opt/homebrew/bin:$PATH"
export DOCKER_API_VERSION=1.43
```

## 2. Capture tests and CI

```bash
flake8 src scripts tests
pytest -q
```

Capture:

- passing test output;
- successful GitHub Actions run:
  <https://github.com/PrasanthChandrababu-Bits/heart-disease-mlops/actions>.

## 3. Capture EDA

Use the images in `reports/eda/`:

- `numeric_histograms.png`
- `class_balance.png`
- `correlation_heatmap.png`

## 4. Capture MLflow

```bash
python scripts/train.py
mlflow ui --backend-store-uri ./mlruns
```

Open <http://127.0.0.1:5000> and capture the experiment runs and metrics.

## 5. Capture Docker

```bash
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/patient.json
```

Capture the build and prediction response. Stop the container with `Ctrl+C`.

## 6. Capture Kubernetes

```bash
minikube start --driver=docker
minikube image load heart-disease-mlops:latest
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl rollout status deployment/heart-disease-api
kubectl rollout status deployment/prometheus
kubectl get pods,services
```

Capture the running pods and services.

## 7. Capture Kubernetes prediction

```bash
kubectl port-forward service/heart-disease-api 8000:8000
```

In another terminal, repeat the `/health` and `/predict` commands from Step 5.

## 8. Capture monitoring and logs

```bash
kubectl port-forward service/prometheus 9090:9090
```

Open <http://127.0.0.1:9090/targets> and capture the API target showing **UP**.

Capture request logs:

```bash
kubectl logs deployment/heart-disease-api
```

## 9. Finish the report

Use `docs/report.md`:

1. Add your name, student ID, course, and date.
2. Insert the screenshots from Steps 2–8.
3. Export approximately 10 pages as DOCX or PDF.

## 10. Record the video

Follow `docs/video-checklist.md` and briefly demonstrate the complete pipeline.

## 11. Clean up

```bash
minikube stop
```

## 12. Submit

Submit:

- Repository:
  <https://github.com/PrasanthChandrababu-Bits/heart-disease-mlops>
- Final DOCX/PDF report
- Short video
- Minikube access instructions from `README.md`
