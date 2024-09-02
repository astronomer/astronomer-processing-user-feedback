"""Anyscale utility functions."""

import time
import warnings
from anyscale.job import status as anyscale_get_job_status  # type: ignore
from anyscale.job.models import JobState, JobStatus  # type: ignore
from anyscale.service import (  # type: ignore
    status as anyscale_get_service_status,
)
from anyscale.service.models import ServiceStatus, ServiceState  # type: ignore


#################################
# Helper Functions
#################################
def wait_for_job_success(job_id: str, poll_interval: int = 10) -> JobStatus:
    while True:
        # Wait for the service to be deployed
        job_status = anyscale_get_job_status(job_id=job_id)
        time.sleep(poll_interval)
        if job_status.state == JobState.SUCCEEDED:
            break
        elif job_status.state == JobState.FAILED:
            raise RuntimeError(f"Job {job_id} failed.")
        elif job_status.state == JobState.UNKNOWN:
            warnings.warn(f"Job {job_id} is in an unknown state.")
    return job_status


def wait_for_service_success(
    service_name: str, poll_interval: int = 10
) -> ServiceStatus:
    while True:
        # Wait for the service to be deployed
        service_status = anyscale_get_service_status(name=service_name)
        time.sleep(poll_interval)
        if service_status.state == ServiceState.RUNNING:
            break
        elif service_status.state in [
            ServiceState.UNHEALTHY,
            ServiceState.TERMINATED,
            ServiceState.SYSTEM_FAILURE,
        ]:
            raise RuntimeError(f"Service {service_name} failed.")
        elif service_status.state in [ServiceState.UNKNOWN]:
            warnings.warn(f"Service {service_name} is in an unknown state.")
    return service_status


def wait_for_canary_version(
    service_name: str, poll_interval: int = 10
) -> ServiceStatus:
    while True:
        # Wait for the service to be deployed
        service_status = anyscale_get_service_status(service_name)
        time.sleep(poll_interval)
        if service_status.canary_version.state == ServiceState.RUNNING:
            break
        elif service_status.canary_version.state in [
            ServiceState.UNHEALTHY,
            ServiceState.TERMINATED,
            ServiceState.SYSTEM_FAILURE,
        ]:
            raise RuntimeError(f"Service {service_name} failed.")
        elif service_status.canary_version.state in [ServiceState.UNKNOWN]:
            warnings.warn(f"Service {service_name} is in an unknown state.")
    return service_status
