#!/usr/bin/env bash
set -euo pipefail

source .env

IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_ARTIFACT_REGISTRY_REPO}/${GCP_SERVICE_NAME}"

echo "--- Building image ---"
docker build -t "${IMAGE}" .

echo "--- Pushing image ---"
docker push "${IMAGE}"

echo "--- Deploying to Cloud Run ---"
gcloud run deploy "${GCP_SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${GCP_REGION}" \
  --project "${GCP_PROJECT_ID}" \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1

echo "--- Done ---"
gcloud run services describe "${GCP_SERVICE_NAME}" \
  --region "${GCP_REGION}" \
  --project "${GCP_PROJECT_ID}" \
  --format "value(status.url)"