import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Template
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.run import Run
from mlflow.tracking.client import MlflowClient

from app.constants import (
    K8S_NS,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    MS_CONFIGMAP_TEMPLATE,
    MS_DEPLOYMENT_PATH,
    MS_CONFIGMAP_NAME,
    MS_DEPLOYMENT_NAME, MS_DEPLOYMENT_TEMPLATE_PATH,
    POLLER_DELAY,
    MS_DEPLOYMENT_TEMPLATE)

logger = logging.getLogger(__name__)


def _poll_once(
    client: MlflowClient, prev_run_id: Optional[str] = None
) -> Optional[str]:
    lst = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE])
    if not lst:
        logger.info(f"No model name={MLFLOW_MODEL_NAME} in stage={MLFLOW_MODEL_STAGE}")
        logger.info(f"Deleting existing model deployments.")
        subprocess.run(
            f"kubectl -n {K8S_NS} delete --ignore-not-found cm {MS_CONFIGMAP_NAME}",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"kubectl -n {K8S_NS} delete --ignore-not-found deployment {MS_DEPLOYMENT_NAME}",
            shell=True,
            check=True,
        )
        return None
    assert len(lst) == 1, (len(lst), lst)
    m: ModelVersion = lst[0]
    if m.run_id == prev_run_id:
        logger.info(f"Already deployed model: run_id={prev_run_id} version={m.version}")
    else:
        logger.info(f"Found a model to deploy! run_id={m.run_id} version={m.version}")
        logger.info(f"Model details:\n{m.to_proto()}")
        r: Run = client.get_run(m.run_id)
        logger.info(f"Model run details:\n{yaml.dump(r.to_dictionary())}")

        logger.info(f"Deploying model run_id={m.run_id} version={m.version}")
        k = f"kubectl -n {K8S_NS}"
        configmap_path = _render_configmap(MS_CONFIGMAP_TEMPLATE, run_id=m.run_id)
        # logger.debug(f"Rendered configmap:\n{configmap_path.read_text()}")
        subprocess.run(f"{k} apply -f {configmap_path}", shell=True, check=True)

        # No zero-downtime :(
        subprocess.run(f"{k} delete deployment {MS_DEPLOYMENT_NAME}", shell=True, check=True)
        subprocess.run(
            f"envsubst '$$GIT_BRANCH' < {MS_DEPLOYMENT_PATH} | {k} apply -f -",
            shell=True,
            check=True,
        )
        logger.info(f"")
        logger.info(f"[+] Successfully patched model deployment to run_id={m.run_id} version={m.version}")
        logger.info(f"")

    return m.run_id


def _render_configmap(configmap_template: Template, run_id: str) -> Path:
    configmap = configmap_template.render(run_id=run_id)
    temp = Path(tempfile.mktemp())
    temp.write_text(configmap)
    return temp


def poll_infinitely():
    client = MlflowClient()
    logger.info(f"Starting poller with delay {POLLER_DELAY}")
    run_id = None
    while True:
        try:
            run_id = _poll_once(client, run_id)
        except BaseException as e:
            logger.exception(f"ERROR: {e}")
        time.sleep(POLLER_DELAY)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    poll_infinitely()
