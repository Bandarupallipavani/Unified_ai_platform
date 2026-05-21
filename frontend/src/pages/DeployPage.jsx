import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  deployModel,
  downloadDockerfile,
  downloadFastapiServer,
  downloadRenderBlueprint,
  downloadModel,
} from "../api";
import { DEPLOY_TARGETS } from "../constants/models";

const LAST_MODEL_KEY = "unifiedai:lastModelId";

const TARGET_CONFIGS = {
  render_free: {
    title: "Render full-stack public links",
    description: "Generate a free Render blueprint for both the frontend app and backend API so users can open public onrender.com URLs right away.",
    steps: [
      "Enter the public Git repo URL and the service names you want to use on Render.",
      "Generate the render.yaml blueprint and optionally open the Deploy to Render link.",
      "Approve both free services and share the frontend URL as the public app link.",
    ],
    fields: [
      { key: "repo_url", label: "Public Git repo URL", placeholder: "https://github.com/your-org/unified-ai-platform" },
      { key: "branch", label: "Branch", placeholder: "main" },
      { key: "frontend_service_name", label: "Frontend service name", placeholder: "uap-web-demo" },
      { key: "backend_service_name", label: "Backend service name", placeholder: "uap-api-demo" },
    ],
  },
  rest: {
    title: "FastAPI REST server",
    description: "Generate a lightweight Python service for immediate local or cloud serving.",
    steps: [
      "Download the generated server file.",
      "Install FastAPI, Uvicorn, pandas, and the model runtime.",
      "Run the endpoint on port 8080 and post prediction payloads to /predict.",
    ],
    fields: [],
  },
  docker: {
    title: "Portable Docker image",
    description: "Package the endpoint into a container you can run locally or push to a registry.",
    steps: [
      "Generate the serving code and Dockerfile bundle.",
      "Build with docker build -t unified-ai-model .",
      "Run the image and expose port 8080 for inference traffic.",
    ],
    fields: [],
  },
  aws: {
    title: "AWS SageMaker",
    description: "Export a cloud deployment script for a managed AWS endpoint.",
    steps: [
      "Fill in your AWS deployment metadata.",
      "Generate the deployment bundle.",
      "Run the exported script inside an authenticated AWS environment.",
    ],
    fields: [
      { key: "region", label: "AWS region", placeholder: "us-east-1" },
      { key: "s3_bucket", label: "S3 bucket", placeholder: "my-sagemaker-bucket" },
      { key: "iam_role", label: "IAM role ARN", placeholder: "arn:aws:iam::123456789012:role/SageMakerRole" },
      { key: "instance_type", label: "Instance type", placeholder: "ml.m5.large" },
    ],
  },
  gcp: {
    title: "GCP Vertex AI",
    description: "Create the script and metadata needed for a managed Vertex AI deployment.",
    steps: [
      "Provide project, region, and storage details.",
      "Generate the deployment script.",
      "Run it inside a gcloud-authenticated shell.",
    ],
    fields: [
      { key: "project_id", label: "Project ID", placeholder: "my-gcp-project" },
      { key: "region", label: "Region", placeholder: "us-central1" },
      { key: "gcs_bucket", label: "GCS bucket", placeholder: "my-vertex-bucket" },
      { key: "machine_type", label: "Machine type", placeholder: "n1-standard-2" },
    ],
  },
  azure: {
    title: "Azure ML",
    description: "Trigger an Azure ML deployment from this app and return the live scoring endpoint when it succeeds.",
    steps: [
      "Make sure the backend machine is authenticated to Azure.",
      "Add your subscription, resource group, and workspace details.",
      "Deploy the model and capture the endpoint URL directly in the result.",
    ],
    fields: [
      { key: "subscription_id", label: "Subscription ID", placeholder: "xxxx-xxxx-xxxx-xxxx" },
      { key: "resource_group", label: "Resource group", placeholder: "unified-ai-rg" },
      { key: "workspace", label: "Workspace", placeholder: "unified-ai-workspace" },
      { key: "service_name", label: "Service name", placeholder: "unified-ai-5bf7b71a" },
      { key: "cpu_cores", label: "CPU cores", placeholder: "1" },
      { key: "memory_gb", label: "Memory (GB)", placeholder: "1" },
    ],
  },
  onnx: {
    title: "ONNX export",
    description: "Convert the model into an interoperable ONNX runtime artifact for broader serving options.",
    steps: [
      "Provide the input feature count if needed.",
      "Run the export to generate the ONNX file.",
      "Use ONNX Runtime in your destination environment.",
    ],
    fields: [{ key: "input_dim", label: "Input features", placeholder: "24" }],
  },
  download: {
    title: "Raw artifact download",
    description: "Pull down the trained model file directly for custom packaging or offline scoring.",
    steps: [
      "Download the stored model artifact.",
      "Load it in your preferred runtime and wrap your own serving layer around it.",
    ],
    fields: [],
  },
};

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function Card({ className = "", children }) {
  return (
    <div className={classNames("rounded-2xl border border-[var(--border)] bg-[var(--surface-solid)]", className)}>
      {children}
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[var(--border)] py-3 last:border-b-0 last:pb-0 first:pt-0">
      <div className="text-sm text-[var(--muted)]">{label}</div>
      <div className="text-right text-sm font-semibold text-[var(--ink)]">{value}</div>
    </div>
  );
}

function saveBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.URL.revokeObjectURL(url);
}

function getFilenameFromHeaders(headers, fallback) {
  const contentDisposition = headers?.get?.("content-disposition") || "";
  const match = contentDisposition.match(/filename="?([^"]+)"?/i);
  return match?.[1] || fallback;
}

export default function DeployPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();

  const [target, setTarget] = useState("render_free");
  const [fieldValues, setFieldValues] = useState({});
  const [deploying, setDeploying] = useState(false);
  const [deployResult, setDeployResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (modelId) localStorage.setItem(LAST_MODEL_KEY, modelId);
  }, [modelId]);

  const config = TARGET_CONFIGS[target];
  const endpointUrl = deployResult?.result?.endpoint_url || "";
  const frontendUrl = deployResult?.result?.frontend_url || "";
  const backendUrl = deployResult?.result?.backend_url || endpointUrl;
  const deployUrl = deployResult?.result?.deploy_url || "";
  const serviceName = deployResult?.result?.service_name || "";
  const workflowState = frontendUrl || backendUrl ? "Created" : deploying ? "Running" : deployResult ? "Created" : "Ready";
  const downloadActions =
    target === "render_free"
      ? [{ key: "render", label: "Download render.yaml" }]
      : target === "rest"
      ? [
          { key: "server", label: "Download server.py" },
          { key: "model", label: "Download model artifact" },
        ]
      : target === "docker"
        ? [
            { key: "server", label: "Download server.py" },
            { key: "docker", label: "Download Dockerfile" },
            { key: "model", label: "Download model artifact" },
          ]
        : target === "download"
          ? [{ key: "model", label: "Download artifact" }]
          : [];

  const handleDeploy = async () => {
    if (!modelId) return;

    setDeploying(true);
    setError("");
    setDeployResult(null);

    try {
      const response = await deployModel(modelId, target, fieldValues);
      setDeployResult(response.data);
    } catch (deployError) {
      setError(deployError.response?.data?.detail || "Deployment failed.");
    } finally {
      setDeploying(false);
    }
  };

  const handleDownload = async (type) => {
    if (!modelId) return;

    try {
      const response =
        type === "server"
          ? await downloadFastapiServer(modelId)
          : type === "render"
            ? await downloadRenderBlueprint(modelId)
          : type === "docker"
            ? await downloadDockerfile(modelId)
            : await downloadModel(modelId);

      if (type === "server") {
        saveBlob(response.data, getFilenameFromHeaders(response.headers, `server_${modelId}.py`));
      }
      if (type === "render") {
        saveBlob(response.data, getFilenameFromHeaders(response.headers, `render_${modelId}.yaml`));
      }
      if (type === "docker") {
        saveBlob(response.data, getFilenameFromHeaders(response.headers, "Dockerfile"));
      }
      if (type === "model") {
        saveBlob(response.data, getFilenameFromHeaders(response.headers, `model_${modelId}.bin`));
      }
    } catch (downloadError) {
      setError(downloadError.response?.data?.detail || "Download failed.");
    }
  };

  return (
    <div className="space-y-5">
      <Card className="p-5 lg:p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-[var(--ink)]">Deploy model</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">
              Choose a target, fill in any required details, and export or deploy the trained model.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => navigate(-1)}
              className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
            >
              Back
            </button>
            <span className="rounded-full bg-[var(--accent-soft)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
              Model {modelId ? modelId.slice(0, 8) : "demo"}
            </span>
          </div>
        </div>

        {error ? (
          <div className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        ) : null}
      </Card>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-5">
          <Card className="p-5 lg:p-6">
            <div className="text-base font-semibold text-[var(--ink)]">Choose target</div>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {DEPLOY_TARGETS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => {
                    setTarget(option.id);
                    setDeployResult(null);
                    setError("");
                  }}
                  className={classNames(
                    "rounded-xl border p-4 text-left",
                    target === option.id
                      ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                      : "border-[var(--border)] hover:border-[var(--border-strong)]"
                  )}
                >
                  <div className="text-base font-semibold text-[var(--ink)]">{option.name}</div>
                  <div className="mt-1 text-sm leading-6 text-[var(--muted)]">{option.desc}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card className="p-5 lg:p-6">
            <div className="text-base font-semibold text-[var(--ink)]">{config.title}</div>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{config.description}</p>

            <ol className="mt-4 space-y-2 pl-5 text-sm leading-6 text-[var(--ink)]">
              {config.steps.map((step) => (
                <li key={step} className="list-decimal">
                  {step}
                </li>
              ))}
            </ol>

            {target === "azure" ? (
              <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-800">
                Azure deploy runs on the backend host. That machine needs Azure access already configured, typically with `az login`, plus the Azure ML SDK installed.
              </div>
            ) : null}

            {target === "render_free" ? (
              <div className="mt-4 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm leading-6 text-sky-800">
                This option is for hosting the full app, not just a single model endpoint. Render can give you free public frontend and backend links, but the repo must contain the app code and any artifacts you want available after deploy.
              </div>
            ) : null}

            <div className="mt-5 grid gap-4 md:grid-cols-2">
              {config.fields.map((field) => (
                <label key={field.key} className="text-sm text-[var(--muted)]">
                  {field.label}
                  <input
                    type="text"
                    className="mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                    placeholder={field.placeholder}
                    value={fieldValues[field.key] || ""}
                    onChange={(event) =>
                      setFieldValues((previous) => ({ ...previous, [field.key]: event.target.value }))
                    }
                  />
                </label>
              ))}
            </div>

            {config.fields.length === 0 ? (
              <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm text-[var(--muted)]">
                No extra configuration is needed for this target.
              </div>
            ) : null}

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={handleDeploy}
                disabled={deploying}
                className="rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
              >
                {deploying ? "Running..." : `Deploy to ${config.title}`}
              </button>

              {downloadActions.map((action) => (
                <button
                  key={action.key}
                  type="button"
                  onClick={() => handleDownload(action.key)}
                  className="rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </Card>

          {deployResult ? (
            <Card className="border-emerald-200 bg-emerald-50 px-5 py-5">
              <div className="text-base font-semibold text-emerald-900">Deployment result</div>
              {frontendUrl || backendUrl ? (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  {frontendUrl ? (
                    <div className="rounded-xl border border-emerald-200 bg-white px-4 py-4">
                      <div className="text-sm text-emerald-700">Frontend public URL</div>
                      <a href={frontendUrl} target="_blank" rel="noreferrer" className="mt-2 block break-all font-mono text-sm text-emerald-900 underline">
                        {frontendUrl}
                      </a>
                    </div>
                  ) : null}
                  {backendUrl ? (
                    <div className="rounded-xl border border-emerald-200 bg-white px-4 py-4">
                      <div className="text-sm text-emerald-700">Backend public URL</div>
                      <a href={backendUrl} target="_blank" rel="noreferrer" className="mt-2 block break-all font-mono text-sm text-emerald-900 underline">
                        {backendUrl}
                      </a>
                    </div>
                  ) : null}
                </div>
              ) : null}
              {deployUrl ? (
                <div className="mt-4 rounded-xl border border-emerald-200 bg-white px-4 py-4">
                  <div className="text-sm text-emerald-700">Deploy to Render</div>
                  <a href={deployUrl} target="_blank" rel="noreferrer" className="mt-2 inline-flex rounded-full bg-emerald-700 px-4 py-2 text-sm font-semibold text-white">
                    Open Render deploy flow
                  </a>
                </div>
              ) : null}
              {endpointUrl && !backendUrl ? (
                <div className="mt-4 rounded-xl border border-emerald-200 bg-white px-4 py-4">
                  <div className="text-sm text-emerald-700">Endpoint URL</div>
                  <div className="mt-2 break-all font-mono text-sm text-emerald-900">{endpointUrl}</div>
                  {serviceName ? <div className="mt-2 text-sm text-emerald-800">Service: {serviceName}</div> : null}
                </div>
              ) : null}
              <pre className="mt-4 overflow-auto whitespace-pre-wrap break-words text-sm text-emerald-900">
                {JSON.stringify(deployResult, null, 2)}
              </pre>
            </Card>
          ) : null}
        </div>

        <div className="space-y-5">
          <Card className="p-5">
            <div className="text-base font-semibold text-[var(--ink)]">Summary</div>
            <div className="mt-4">
              <SummaryRow label="Target" value={target.toUpperCase()} />
              <SummaryRow label="Artifact" value={modelId ? `${modelId.slice(0, 8)}...` : "--"} />
              <SummaryRow label="Status" value={workflowState} />
              {frontendUrl ? <SummaryRow label="Frontend URL" value={frontendUrl} /> : null}
              {backendUrl ? <SummaryRow label="Backend URL" value={backendUrl} /> : null}
              {serviceName ? <SummaryRow label="Service" value={serviceName} /> : null}
            </div>
          </Card>

          <Card className="p-5">
            <div className="text-base font-semibold text-[var(--ink)]">
              {target === "render_free" ? "Public link preview" : "Request preview"}
            </div>
            <div className="mt-4 rounded-xl bg-[var(--surface-dark)] p-4 text-xs leading-7 text-emerald-300">
              <pre className="whitespace-pre-wrap font-mono">{target === "render_free"
                ? `Frontend app: ${frontendUrl || "https://your-frontend-name.onrender.com"}
Backend API: ${backendUrl || "https://your-backend-name.onrender.com"}
Deploy flow: ${deployUrl || "https://render.com/deploy"}`
                : `curl -X POST ${endpointUrl || "http://localhost:8080/predict"}
  -H "Content-Type: application/json"
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]], "columns": []}'`}</pre>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
