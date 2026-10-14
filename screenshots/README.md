# Submission evidence checklist

Capture these images after the final GitHub and Minikube runs:

1. EDA histograms, class balance, and correlation heatmap.
2. MLflow experiment list with both candidate runs and metrics.
3. Passing local test output.
4. Successful GitHub Actions workflow and uploaded artifacts.
5. Successful Docker build and local `/predict` response.
6. `kubectl get pods,services` with API and Prometheus running.
7. Minikube-hosted `/health` and `/predict` responses.
8. Prometheus **Targets** page showing `heart-disease-api` as UP.
9. API request logs from `kubectl logs`.

Do not include credentials, access tokens, or personal account details.
