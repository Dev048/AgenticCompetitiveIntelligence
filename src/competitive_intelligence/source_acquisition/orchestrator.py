"""Acquisition orchestration and fallback management for Source Acquisition subsystem.

Provides AcquisitionOrchestrator, ProviderRegistry, DefaultFallbackPolicy, and ProviderNotRegisteredError.
"""

from competitive_intelligence.source_acquisition.interfaces import (
    AcquisitionClient,
    FallbackPolicy,
)
from competitive_intelligence.source_acquisition.models import (
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    Source,
)


class ProviderNotRegisteredError(Exception):
    """Raised when an acquisition method defined in a tool chain has no registered provider client."""

    def __init__(self, method: AcquisitionMethod) -> None:
        self.method = method
        super().__init__(
            f"No acquisition provider registered for acquisition method: '{method.value}'"
        )


class DefaultFallbackPolicy(FallbackPolicy):
    """Default acquisition fallback policy.

    Classifies failures as recoverable vs non-recoverable:
    - Success: Never falls back (attempt already succeeded).
    - Non-recoverable failures (NOT_FOUND, AUTH_REQUIRED): Halts the chain immediately,
      as subsequent attempts with fallback providers will encounter the same fundamental issue.
    - Recoverable failures (RATE_LIMITED, BLOCKED, TIMEOUT, NETWORK_ERROR, PARSER_ERROR, SERVER_ERROR, FAILURE):
      Eligible for fallback.

    Note on PARSER_ERROR:
        Parser/extraction errors are treated as recoverable on the assumption that an alternate
        provider in the tool chain may use a different acquisition or extraction mechanism
        (e.g., direct HTTP text parsing vs. headless browser DOM extraction).
    """

    NON_RECOVERABLE_STATUSES: frozenset[AcquisitionStatus] = frozenset(
        {
            AcquisitionStatus.NOT_FOUND,
            AcquisitionStatus.AUTH_REQUIRED,
        }
    )

    def is_recoverable(self, result: AcquisitionResult) -> bool:
        """Determine whether an acquisition failure is eligible for fallback."""
        if result.is_success:
            return False
        return result.status not in self.NON_RECOVERABLE_STATUSES

    def should_fallback(self, result: AcquisitionResult) -> bool:
        """Determine whether the orchestrator should attempt fallback to the next provider."""
        return self.is_recoverable(result)


class ProviderRegistry:
    """Registry mapping AcquisitionMethod enums to their corresponding AcquisitionClient implementations."""

    def __init__(
        self,
        providers: dict[AcquisitionMethod, AcquisitionClient] | None = None,
    ) -> None:
        self._providers: dict[AcquisitionMethod, AcquisitionClient] = {}
        if providers:
            for method, client in providers.items():
                self.register(method, client)

    def register(self, method: AcquisitionMethod, client: AcquisitionClient) -> None:
        """Register an acquisition client for a specific acquisition method."""
        self._providers[method] = client

    def get(self, method: AcquisitionMethod) -> AcquisitionClient | None:
        """Get the registered acquisition client for a method, if available."""
        return self._providers.get(method)

    def has_provider(self, method: AcquisitionMethod) -> bool:
        """Check if a provider is registered for a method."""
        return method in self._providers

    def __contains__(self, method: AcquisitionMethod) -> bool:
        return self.has_provider(method)

    def __getitem__(self, method: AcquisitionMethod) -> AcquisitionClient:
        client = self.get(method)
        if client is None:
            raise ProviderNotRegisteredError(method)
        return client


class AcquisitionOrchestrator:
    """Orchestrates source data acquisition across configured primary and fallback providers.

    Responsibilities:
    - Receives a Source domain model.
    - Resolves the execution tool chain (primary -> fallbacks).
    - Verifies that required providers are registered, raising ProviderNotRegisteredError on configuration problems.
    - Executes acquisition via registered AcquisitionClient instances.
    - Uses FallbackPolicy to decide whether to return immediately or try fallback providers.
    - Returns the final successful AcquisitionResult or the final failure AcquisitionResult.
    """

    def __init__(
        self,
        providers: dict[AcquisitionMethod, AcquisitionClient] | ProviderRegistry | None = None,
        fallback_policy: FallbackPolicy | None = None,
    ) -> None:
        if isinstance(providers, ProviderRegistry):
            self.registry = providers
        elif isinstance(providers, dict):
            self.registry = ProviderRegistry(providers)
        else:
            self.registry = ProviderRegistry()

        self.fallback_policy = fallback_policy or DefaultFallbackPolicy()

    def register_provider(
        self, method: AcquisitionMethod, client: AcquisitionClient
    ) -> None:
        """Register an acquisition client for a specific acquisition method."""
        self.registry.register(method, client)

    def get_provider(self, method: AcquisitionMethod) -> AcquisitionClient | None:
        """Retrieve a registered provider for an acquisition method."""
        return self.registry.get(method)

    async def acquire(self, source: Source) -> AcquisitionResult:
        """Execute acquisition toolchain for the given source according to its configured policy.

        Args:
            source: Source domain model to acquire.

        Returns:
            AcquisitionResult from the successful provider or the terminal failure result.

        Raises:
            ProviderNotRegisteredError: If any provider in the required tool chain is not registered.
        """
        tool_chain = source.acquisition.tool_chain
        if tool_chain is not None:
            methods = [tool_chain.primary] + list(tool_chain.fallbacks)
        else:
            methods = [source.acquisition.acquisition_method]

        last_result: AcquisitionResult | None = None

        for method in methods:
            client = self.registry.get(method)
            if client is None:
                raise ProviderNotRegisteredError(method)

            try:
                result = await client.acquire(source)
            except Exception as exc:
                result = AcquisitionResult(
                    status=AcquisitionStatus.FAILURE,
                    method=method,
                    error_message=f"Provider raised unexpected exception: {exc}",
                    metadata={"exception_type": type(exc).__name__},
                )

            last_result = result

            if result.is_success:
                return result

            if not self.fallback_policy.should_fallback(result):
                # Non-recoverable failure stops the toolchain immediately
                return result

        # All providers in the toolchain were tried and failed recoverably
        if last_result is not None:
            return last_result

        return AcquisitionResult(
            status=AcquisitionStatus.FAILURE,
            error_message="No acquisition methods configured for source",
        )
