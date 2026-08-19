"""Deterministic tool layer (not implemented yet).

Every fact the agent states about the cluster originates here. Tools contain no LLM
calls; they are ordinary, unit-testable Python functions over the Kubernetes API.

Planned contract for this package:

``base.py``
    ``Tool``: a name, description, Pydantic argument model, a ``mutating`` flag, and a
    ``run(args) -> ToolResult``. The argument model doubles as the JSON schema handed to
    the LLM and as the validation boundary, so a hallucinated argument fails closed with
    ``ToolArgumentError`` before reaching the cluster.

``registry.py``
    Name -> tool lookup, and the source of the tool schemas advertised to the model.
    When ``settings.read_only_mode`` is set, mutating tools are never registered.

``k8s/client.py``
    Kubeconfig loading (context, timeouts) and translation of ``ApiException`` into the
    application error taxonomy.

``k8s/read.py``
    ``list_pods``, ``get_pod``, ``get_pod_logs``, ``describe_pod``, ``list_events``,
    ``list_deployments``. Safe to run without confirmation.

``k8s/mutate.py``
    ``restart_deployment``, ``scale_deployment``, ``delete_pod``. Always gated behind an
    explicit confirmation token.
"""
