"""Tool argument models and Kubernetes name validation.

These models are the validation boundary between an untrusted caller (eventually an
LLM) and the cluster. Two properties matter:

* ``extra="forbid"`` -- an argument the tool does not define is an error, not something
  quietly dropped. A model that invents ``{"namespace": "x", "force": true}`` fails loudly.
* Every name is pattern-checked against the Kubernetes naming rules before it reaches
  the API. The patterns admit only lowercase alphanumerics, ``-`` and ``.``, so shell
  metacharacters, path traversal and field-selector injection cannot survive validation
  even if a downstream call were ever built by string concatenation.

The same models double as the JSON schema advertised to the model later, so the
description a tool-caller sees and the rules actually enforced cannot drift apart.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic import ValidationError as PydanticValidationError

from app.errors import ToolArgumentError

# RFC 1123 label: namespaces, container names. Max 63 characters.
DNS_LABEL_PATTERN = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
# RFC 1123 subdomain: pod and deployment names, which may contain dots. Max 253.
DNS_SUBDOMAIN_PATTERN = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"

NamespaceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=63, pattern=DNS_LABEL_PATTERN
    ),
]
ResourceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=253, pattern=DNS_SUBDOMAIN_PATTERN
    ),
]
ContainerName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=63, pattern=DNS_LABEL_PATTERN
    ),
]

# Upper bound independent of configuration, so a caller cannot ask for a million lines.
# The configured `max_log_lines` clamps further at execution time.
MAX_TAIL_LINES = 2000


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListPodsInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace to list pods in.")


class GetPodInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace containing the pod.")
    pod_name: ResourceName = Field(description="Exact name of the pod.")


class DescribePodInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace containing the pod.")
    pod_name: ResourceName = Field(description="Exact name of the pod.")


class DiagnosePodInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace containing the pod.")
    pod_name: ResourceName = Field(description="Exact name of the pod to gather evidence on.")


class GetPodLogsInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace containing the pod.")
    pod_name: ResourceName = Field(description="Exact name of the pod.")
    container: ContainerName | None = Field(
        default=None,
        description="Container name. Required only when the pod has multiple containers.",
    )
    tail_lines: int = Field(
        default=100,
        ge=1,
        le=MAX_TAIL_LINES,
        description="Number of lines to return from the end of the log.",
    )
    previous: bool = Field(
        default=False,
        description=(
            "Read the log of the previous terminated container. Required to diagnose a "
            "CrashLoopBackOff, where the current container has no output yet."
        ),
    )


class ListDeploymentsInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace to list deployments in.")


class RestartDeploymentInput(ToolInput):
    namespace: NamespaceName = Field(description="Namespace containing the deployment.")
    deployment_name: ResourceName = Field(description="Exact name of the deployment.")


def parse_arguments[TInput: ToolInput](model: type[TInput], **kwargs: object) -> TInput:
    """Validate arguments, converting Pydantic failures into the app error taxonomy."""
    try:
        return model(**kwargs)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            "Invalid tool arguments.",
            tool_input=model.__name__,
            errors=[
                {
                    "field": ".".join(str(part) for part in error["loc"]),
                    "message": error["msg"],
                }
                for error in exc.errors()
            ],
        ) from exc
