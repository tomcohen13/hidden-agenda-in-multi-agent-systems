"""Agent module"""

import json
from pathlib import Path

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, NotRequired, TypedDict


class CaptureModelInput(AsyncCallbackHandler):
    """Save rendered model messages for each call to a local JSON file."""

    async def on_chat_model_start(
        self, serialized, messages, *, run_id, **kwargs
    ):
        directory = Path("debug/model_inputs")
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": str(run_id),
            "messages": [messages_to_dict(batch) for batch in messages],
        }
        (directory / f"{run_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


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


class DrawAndPass(BaseModel):
    """A private selection of two policy cards to pass to the policy-maker."""

    model_config = ConfigDict(strict=True, extra="forbid")
    passed_cards: list[int] = Field(min_length=2, max_length=2)
    private_justification: str

class Enact(BaseModel):
    """A private selection of one policy color to enact."""

    model_config = ConfigDict(strict=True, extra="forbid")
    policy: Literal[0, 1]
    private_justification: str

class Response(BaseModel):
    """A public response to the group"""
    
    model_config = ConfigDict(strict=True, extra="forbid")
    content: str


RESPONSE_SCHEMAS = {
    "propose_policy_maker": Nomination,
    "vote": Vote,
    "draw_and_pass": DrawAndPass,
    "enact": Enact,
    "discussion": Response,
}


class GameAgentState(AgentState):
    # inherits messages from AgentState
    player_id: NotRequired[int]
    phase: NotRequired[str]
    observation: NotRequired[dict[str, Any]]

class AgentContext(TypedDict):
    phase: Literal[
        "propose_policy_maker",
        "vote",
        "draw_and_pass",
        "enact",
        "discussion",
    ]
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
        speaker = "game controller" if message["player_id"] == -1 else f"player {message['player_id']}"
        pending.append(HumanMessage(
            content=(f"Public board — round {message['round_number']}, "
                     f"{speaker}:\n{message['content']}"),
            id=f"public:{message['id']}",
            additional_kwargs={"public_message_id": message["id"]},
        ))
        seen.add(message["id"])
    return pending


def create_game_agent(
    name, player_prompt: str, *, model, phase_prompts: dict[str, str],
    checkpointer: BaseCheckpointSaver | None = None,
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
        state_schema=GameAgentState,
        middleware=[phase_prompt, PhaseResponseFormat()],
        context_schema=AgentContext,
        response_format=ToolStrategy(Nomination | Vote | DrawAndPass | Enact | Response, handle_errors=False),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
