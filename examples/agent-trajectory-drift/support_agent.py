"""A support agent, in two versions, that answer identically by different routes.

v1 looks the customer up and answers.
v2 looks the customer up, then also checks their billing status -- and returns
the exact same sentence to the user.
"""

from insideLLMs.probes.agent_probe import AgentProbe, ToolDefinition

LOOKUP = ToolDefinition(
    name="lookup_customer",
    description="Fetch a customer record by id",
    parameters={"customer_id": {"type": "string"}},
)
BILLING = ToolDefinition(
    name="check_billing",
    description="Fetch billing status for a customer",
    parameters={"customer_id": {"type": "string"}},
)

ANSWER = "Your account is active. Anything else I can help with?"


class SupportAgent(AgentProbe):
    """Answers a support question, optionally consulting billing as well."""

    def __init__(self, version: int = 1, **kwargs):
        tools = [LOOKUP] if version == 1 else [LOOKUP, BILLING]
        super().__init__(name="support-agent", tools=tools, **kwargs)
        self.version = version

    def run_agent(self, model, prompt, tools, recorder, **kwargs):
        recorder.record_generate_start(prompt)

        recorder.record_tool_call("lookup_customer", {"customer_id": "c-1001"})
        recorder.record_tool_result("lookup_customer", {"plan": "pro"})

        if self.version == 2:
            # v2 adds a second call. The user-visible answer does not change.
            recorder.record_tool_call("check_billing", {"customer_id": "c-1001"})
            recorder.record_tool_result("check_billing", {"status": "current"})

        recorder.record_generate_end(ANSWER)
        return ANSWER
