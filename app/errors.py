"""Error taxonomy and HTTP exception handlers.

Every failure -- from a bad request body to a Kubernetes timeout -- leaves the API in
the same envelope shape so clients (and later, the agent itself) can branch on a stable
``code`` instead of parsing prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.context import get_correlation_id
from app.observability.logging import get_logger

_logger = get_logger(__name__)

# Spelled out rather than taken from `status`, where the old name is deprecated and the
# new one is not present in every supported Starlette release.
HTTP_422_UNPROCESSABLE_CONTENT = 422


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    TOOL_ARGUMENT_INVALID = "tool_argument_invalid"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    RESOURCE_NOT_FOUND = "resource_not_found"
    NAMESPACE_NOT_ALLOWED = "namespace_not_allowed"
    CLUSTER_UNAVAILABLE = "cluster_unavailable"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_INVALID = "confirmation_invalid"
    MUTATION_DISABLED = "mutation_disabled"
    LLM_UNAVAILABLE = "llm_unavailable"
    AGENT_LOOP_LIMIT = "agent_loop_limit"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    correlation_id: str | None = None


class AppError(Exception):
    """Base class for every expected failure in the application."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorDetail(code=self.code, message=self.message, details=self.details),
            correlation_id=get_correlation_id(),
        )


class InvalidRequestError(AppError):
    code = ErrorCode.INVALID_REQUEST
    status_code = status.HTTP_400_BAD_REQUEST


class ToolArgumentError(AppError):
    code = ErrorCode.TOOL_ARGUMENT_INVALID
    status_code = status.HTTP_400_BAD_REQUEST


class ToolNotFoundError(AppError):
    code = ErrorCode.TOOL_NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND


class ToolExecutionError(AppError):
    code = ErrorCode.TOOL_EXECUTION_FAILED
    status_code = status.HTTP_502_BAD_GATEWAY


class ResourceNotFoundError(AppError):
    code = ErrorCode.RESOURCE_NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND


class NamespaceNotAllowedError(AppError):
    code = ErrorCode.NAMESPACE_NOT_ALLOWED
    status_code = status.HTTP_403_FORBIDDEN


class ClusterUnavailableError(AppError):
    code = ErrorCode.CLUSTER_UNAVAILABLE
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class ConfirmationRequiredError(AppError):
    code = ErrorCode.CONFIRMATION_REQUIRED
    status_code = status.HTTP_409_CONFLICT


class ConfirmationInvalidError(AppError):
    code = ErrorCode.CONFIRMATION_INVALID
    status_code = status.HTTP_409_CONFLICT


class MutationDisabledError(AppError):
    code = ErrorCode.MUTATION_DISABLED
    status_code = status.HTTP_403_FORBIDDEN


class LLMUnavailableError(AppError):
    code = ErrorCode.LLM_UNAVAILABLE
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class AgentLoopLimitError(AppError):
    code = ErrorCode.AGENT_LOOP_LIMIT
    status_code = HTTP_422_UNPROCESSABLE_CONTENT


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        _logger.warning(
            "request.failed",
            error_code=exc.code.value,
            error_message=exc.message,
            status_code=exc.status_code,
            **exc.details,
        )
        return _json(exc.status_code, exc.to_response())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        response = ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INVALID_REQUEST,
                message="Request payload failed validation.",
                details={"errors": _serialisable_errors(exc)},
            ),
            correlation_id=get_correlation_id(),
        )
        _logger.info("request.invalid", error_count=len(exc.errors()))
        return _json(HTTP_422_UNPROCESSABLE_CONTENT, response)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = (
            ErrorCode.RESOURCE_NOT_FOUND
            if exc.status_code == status.HTTP_404_NOT_FOUND
            else ErrorCode.INVALID_REQUEST
        )
        response = ErrorResponse(
            error=ErrorDetail(code=code, message=str(exc.detail)),
            correlation_id=get_correlation_id(),
        )
        return _json(exc.status_code, response)

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the caller; the correlation ID ties the opaque
        # response back to the full traceback in the logs.
        _logger.exception("request.unhandled_error", error_type=type(exc).__name__)
        response = ErrorResponse(
            error=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="An unexpected error occurred.",
            ),
            correlation_id=get_correlation_id(),
        )
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, response)


def _json(status_code: int, response: ErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))


def _serialisable_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {
            "location": list(error.get("loc", ())),
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]
