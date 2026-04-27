#  Distributed Image Classifier on Kubernetes

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?logo=kubernetes&logoColor=white)](https://kubernetes.io/)


The autoscaling microservice deployed on Kubernetes that accepts image via API, performs classification using a PyTorch ResNet18 model, and draws metrics graphs using Grafana and Prometheus


##  Architecture & Design Choices

- **API Framework (`FastAPI`):** Chosen for its high performance, native async support, and automatic OpenAPI documentation. Heavy ML inference is offloaded to a separate threadpool to prevent event loop blocking.

- **Machine Learning (`PyTorch` & `ResNet18`):** ResNet18 is pretrained model for classification images into 1000 classes

- **Package Management (`uv`):** Used for extremely fast dependency resolution and strict lockfile generation.

- **Automation (`go-task`):** Used instead of complex Bash scripts. Features built-in smart K8s context detection (automatically loads local images into `minikube` or `kind` without needing a local registry).

- **Autoscaling (`HPA`):** Horizontal Pod Autoscaler monitors CPU utilization and dynamically scales the classification pods under load.

- **Metrics (`Prometheus and Grafana`):**  The industry-standard stack for K8s observability. Prometheus handles real-time metrics collection and storage, while Grafana provides deep visualization of system health, enabling precise tracking of request latency and HPA scaling efficiency.

- **Testing (`Locust`):** Python-based distributed performance testing tool used to simulate hundreds of concurrent users. It serves to validate microservice stability and benchmark the HPA's responsiveness to aggressive traffic spikes.

---

## Step 0: Prerequisites (Crucial)

To ensure a smooth deployment, you **must** have the following installed on your host machine:

1. **Python 3.10+** (Required to parse deployment scripts and run Locust).
2. **A local Kubernetes Cluster** (Docker Desktop, Minikube, or Kind).

> **⚠️ MINIKUBE USERS - CRITICAL STEP:**
> The Autoscaler (HPA) requires metrics to function. You **must** enable the metrics server before deploying:
> ```bash
> minikube addons enable metrics-server
> ```

**3. Install `go-task` (The Task Runner)**
This project uses `task` to automate everything. You must install it first:

```bash
# Windows (via built-in winget)
winget install Task.Task

# macOS (Homebrew)
brew install go-task/tap/go-task

# Linux (Installs globally to /usr/local/bin)
sudo sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
```

*(Note: The `task` CLI will automatically verify if you have `uv`, `docker`, `kubectl`, and `helm` installed when you run it).*

---

##  Step 1: One-Click Deployment

Deploy the entire architecture (App Build, K8s Deployment, Helm Monitoring Stack) with a single command:

```bash
task start
```

**What this does automatically:**
1. Verifies system requirements.
2. Calculates source code hash to generate a unique Docker image tag.
3. Builds the image and intelligently loads it into your specific K8s environment.
4. Installs the `kube-prometheus-stack` via Helm into the `monitoring` namespace.
5. Applies Kubernetes manifests, waits for the rollout, and prints the **Grafana admin password**.

###  Accessing the API
Once deployed, the FastAPI Swagger UI is available at:
👉 **http://localhost/docs**

> **⚠️ MINIKUBE USERS:** K8s LoadBalancers do not expose to `localhost` automatically in Minikube. You **must** run the following in a separate terminal and keep it open:
> ```bash
> minikube tunnel
> ```

---

##  Step 2: Load Testing & Autoscaling

To validate throughput, latency, and HPA behavior, use the built-in test suite. This will set up the necessary port-forwards to Grafana and start the Locust load testing tool.

```bash
task test
```

*(You can override default load parameters: `task test USERS=50 RATE=5`)*

###  How to Monitor the Test:

1.  **Start the Load (Locust):** 
    Open **http://localhost:8089**. The test will automatically start hitting the `/predict` endpoint with random image data. Observe RPS and Latency here.
2.  **Watch the Autoscaler (HPA):** 
    Open a new terminal and watch Kubernetes spawn new pods as the CPU load increases:
    ```bash
    kubectl get hpa -w
    kubectl get pods -w
    ```
3.  **View Prometheus Metrics (Grafana):** 
    Open **http://localhost:3000**. Login with username `admin` (use the password printed at the end of `task start`).
    *   *Navigate to:* Dashboards -> Kubernetes / Compute Resources / Namespace (Workloads).

---

## Local Development (Without K8s)

You can run and test the application locally without full cluster deployment:

```bash
task run       # Starts the FastAPI server locally on port 4123
task lint      # Runs Ruff to format and lint code
```

---

## Step 3: Cleanup

Manage your cluster state easily with these commands to free up resources:

```bash
# Remove Application only (keeps monitoring stack and Grafana data)
task down

# NUCLEAR Clean: Delete everything (App, Monitoring Stack, Namespace)
task teardown

# Clean local caches (Python __pycache__, Ruff, UV, Docker dangling images)
task clean
```