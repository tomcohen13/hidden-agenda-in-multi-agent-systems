"""Agent module"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, NotRequired, TypedDict


class PublicMessage(TypedDict):
    id: str
    player_id: int
    round_number: int
    content: str


class Nomination(BaseModel):
    """A proposed policy maker and the public justification for their nomination."""

    model_config = ConfigDict(strict=True, extra="forbid")
    policy_maker_id: int
    justification: str = Field(min_length=1)


class Vote(BaseModel):
    """An approval ballot; this does not automatically publish a message."""

    model_config = ConfigDict(strict=True, extra="forbid")
    approve: bool
    justification: str


RESPONSE_SCHEMAS = {"propose_policy_maker": Nomination, "vote": Vote}


class AgentContext(TypedDict):
    phase: Literal["propose_policy_maker", "vote"]
    prompt_kwargs: NotRequired[dict[str, Any]]


class PhaseResponseFormat(AgentMiddleware):
    """Enforce the current phase's schema without changing agent memory."""

    def wrap_model_call(self, request, handler):
        schema = RESPONSE_SCHEMAS[request.runtime.context["phase"]]
        return handler(request.override(response_format=ToolStrategy(schema, handle_errors=False)))

    async def awrap_model_call(self, request, handler):
        schema = RESPONSE_SCHEMAS[request.runtime.context["phase"]]
        return await handler(request.override(response_format=ToolStrategy(schema, handle_errors=False)))


async def catch_up_messages(
    agent: CompiledStateGraph, config: dict, messages: list[PublicMessage]
) -> list[HumanMessage]:
    """Return public board entries that aren't in this agent's checkpoint history."""
    snapshot = await agent.aget_state(config)
    seen = {
        message.additional_kwargs["public_message_id"]
        for message in snapshot.values.get("messages", [])
        if "public_message_id" in message.additional_kwargs
    }
    pending = []
    for message in messages:
        if message["id"] in seen:
            continue
        pending.append(HumanMessage(
            content=(f"Public board — round {message['round_number']}, "
                     f"player {message['player_id']}:\n{message['content']}"),
            id=f"public:{message['id']}",
            additional_kwargs={"public_message_id": message["id"]},
        ))
        seen.add(message["id"])
    return pending


def create_game_agent(
    name, player_prompt: str, *, model, phase_prompts: dict[str, str]
) -> CompiledStateGraph:
    """Build one player's runtime with isolated memory and phase-specific output."""

    @dynamic_prompt
    def phase_prompt(request) -> str:
        """Combine the player's base instructions with the current phase template."""
        context = request.runtime.context
        phase_instructions = phase_prompts[context["phase"]].format(
            **context.get("prompt_kwargs", {})
        )
        return "\n\n".join([player_prompt, phase_instructions])

    return create_agent(
        model=model,
        name=str(name),
        system_prompt=player_prompt,
        middleware=[phase_prompt, PhaseResponseFormat()],
        context_schema=AgentContext,
        response_format=ToolStrategy(Nomination | Vote, handle_errors=False),
        checkpointer=InMemorySaver(),
    )
