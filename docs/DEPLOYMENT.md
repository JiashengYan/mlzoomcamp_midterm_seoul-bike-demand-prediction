# Optional deployment to Google Cloud Run

**Deploy the existing Docker image as one authenticated service after the owner approves the account and costs.** Local reproduction remains the default in [README](../README.md). These commands are a deployment recipe, not evidence of a completed cloud deployment; no hosted URL is currently available.

## 1. Prepare the same local artifacts

**Use the existing CSV, saved model, and Dockerfile; the cloud service performs prediction only.** No new application framework or database is needed. Commands below use PowerShell from the project root.

Complete README section 4.2 first. Check the local Docker service with section 5. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and use a billing-enabled Google Cloud project. The deploying account needs permission to enable APIs, manage the image repository, deploy Cloud Run services, and act as the runtime service account. The testing account needs Cloud Run Invoker permission.

This image includes the fixed historical CSV, so the cloud endpoint accepts the same November 2018 replay dates as the local API. It does not provide live forecasts for today's dates.

## 2. Build, upload, and deploy after approval

**Build the existing image locally and upload it to Artifact Registry, then run it on Cloud Run.** Choose the approved project and region. Repository creation is a one-time step; skip it when the named repository already exists.

~~~powershell
$projectId = "YOUR_APPROVED_PROJECT_ID"
$region = "europe-west1"
$repository = "seoul-bike"
$service = "seoul-bike"
$image = "${region}-docker.pkg.dev/${projectId}/${repository}/seoul-bike:v1"

gcloud auth login
gcloud config set project $projectId
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create $repository --repository-format=docker --location=$region
gcloud auth configure-docker "${region}-docker.pkg.dev"

docker build --platform linux/amd64 -t $image .
docker push $image
gcloud run deploy $service --image=$image --region=$region --port=8000 --cpu=1 --memory=1Gi --min=0 --max=1 --concurrency=8 --no-allow-unauthenticated
~~~

Cloud Run uses the image's existing command; port 8000 matches the API. Minimum zero instances allows scale-down, and maximum one limits service scaling. These settings do not impose a spending cap: image storage, requests, and other billable usage still apply. Authentication is required; public access needs its own owner decision.

Commands follow the official [image upload guide](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling) and [Cloud Run deployment reference](https://cloud.google.com/sdk/gcloud/reference/run/deploy).

## 3. Test and record the deployed service

**Call the resulting URL with an identity token from an account allowed to invoke the service.** Authentication follows the [developer invocation guide](https://cloud.google.com/run/docs/authenticating/developers).

~~~powershell
$serviceUrl = gcloud run services describe $service --region=$region --format="value(status.url)"
$token = gcloud auth print-identity-token
$headers = @{ Authorization = "Bearer $token" }

Invoke-RestMethod -Uri "$serviceUrl/health" -Headers $headers
$body = '{"issued_at":"2018-11-16T17:00:00+09:00"}'
Invoke-RestMethod -Uri "$serviceUrl/predict" -Method Post -Headers $headers -ContentType "application/json" -Body $body
~~~

Expect readiness plus a rental-count prediction for 18:00–19:00. After an actual deployment, record the date, revision/image, service URL, and health/prediction responses in the work log. A test image or video can show the request and response; omit identity tokens. State whether reviewer access requires authentication.

## 4. Remove the demonstration resources

**Delete the service and its dedicated image repository when the demonstration is finished.** These commands permanently remove the named resources; use them only for this project's dedicated demonstration resources after confirming their names.

~~~powershell
gcloud run services delete $service --region=$region
gcloud artifacts repositories delete $repository --location=$region
~~~

Deletion is part of the future deployment procedure and has not been executed here.
