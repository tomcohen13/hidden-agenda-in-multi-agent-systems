"""Catch-up tests using real checkpoints without model calls."""

import asyncio
from copy import deepcopy

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import Field

from social_deduction.agent import (
    Nomination,
    Vote,
    catch_up_messages,
    create_game_agent,
)


@pytest.fixture
def agent():
    builder = StateGraph(MessagesState)
    builder.add_node("receive", lambda state: {})
    builder.add_edge(START, "receive")
    builder.add_edge("receive", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "game-a:0"}}


@pytest.fixture
def board():
    return [
        {"id": "first", "player_id": 1, "round_number": 1, "content": "Hello."},
        {"id": "second", "player_id": 2, "round_number": 1, "content": "Hello."},
        {"id": "third", "player_id": 0, "round_number": 2, "content": "My proposal."},
    ]


def test_fresh_history_receives_board_in_order(agent, config, board):
    pending = asyncio.run(catch_up_messages(agent, config, board))

    assert all(isinstance(message, HumanMessage) for message in pending)
    assert [message.id for message in pending] == [
        "public:first",
        "public:second",
        "public:third",
    ]
    assert [message.additional_kwargs["public_message_id"] for message in pending] == [
        "first",
        "second",
        "third",
    ]
    assert pending[0].content == "Public board — round 1, player 1:\nHello."
    assert pending[2].content == "Public board — round 2, player 0:\nMy proposal."


def test_empty_board_returns_no_messages(agent, config):
    assert asyncio.run(catch_up_messages(agent, config, [])) == []


@pytest.mark.parametrize("delivered_count", [1, 2, 3])
def test_only_unseen_messages_are_returned(agent, config, board, delivered_count):
    async def scenario():
        delivered = await catch_up_messages(agent, config, board[:delivered_count])
        await agent.aupdate_state(config, {"messages": delivered}, as_node="receive")
        return await catch_up_messages(agent, config, board)

    pending = asyncio.run(scenario())

    assert [message.additional_kwargs["public_message_id"] for message in pending] == [
        message["id"] for message in board[delivered_count:]
    ]


def test_duplicate_ids_are_skipped_but_identical_text_is_preserved(
    agent, config, board
):
    pending = asyncio.run(
        catch_up_messages(agent, config, [board[0], board[0], board[1], board[0]])
    )

    assert [message.additional_kwargs["public_message_id"] for message in pending] == [
        "first",
        "second",
    ]


def test_private_history_does_not_count_as_board_delivery(agent, config, board):
    async def scenario():
        await agent.aupdate_state(
            config,
            {
                "messages": [
                    HumanMessage(content="Private task."),
                    AIMessage(content="Hello."),
                ]
            },
            as_node="receive",
        )
        return await catch_up_messages(agent, config, board)

    assert len(asyncio.run(scenario())) == len(board)


@pytest.mark.parametrize("other_thread", ["game-a:1", "game-b:0"])
def test_delivery_is_isolated_by_player_and_game(agent, config, board, other_thread):
    async def scenario():
        delivered = await catch_up_messages(agent, config, board)
        await agent.aupdate_state(config, {"messages": delivered}, as_node="receive")
        assert await catch_up_messages(agent, config, board) == []
        return await catch_up_messages(
            agent, {"configurable": {"thread_id": other_thread}}, board
        )

    assert len(asyncio.run(scenario())) == len(board)


def test_catch_up_does_not_mutate_board_or_mark_messages_delivered(
    agent, config, board
):
    original_board = deepcopy(board)

    async def scenario():
        before = await agent.aget_state(config)
        first = await catch_up_messages(agent, config, board)
        second = await catch_up_messages(agent, config, board)
        after = await agent.aget_state(config)
        assert first == second
        assert before.values == after.values

    asyncio.run(scenario())

    assert board == original_board


class RecordingModel(FakeMessagesListChatModel):
    """Record model inputs and return scripted structured responses."""

    requests: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("use_async", [False, True])
def test_dynamic_prompt_switches_phase_and_preserves_role_and_history(
    config, use_async
):
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Nomination",
                        "id": "nomination",
                        "args": {"policy_maker_id": 2, "justification": "My nominee."},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Vote",
                        "id": "vote",
                        "args": {"approve": True, "justification": "I agree."},
                    }
                ],
            ),
        ]
    )
    agent = create_game_agent(
        0,
        player_prompt="You are a red player.",
        model=model,
        phase_prompts={
            "propose_policy_maker": "Nominate from {}.",
            "vote": "Cast your vote.",
        },
    )

    async def scenario():
        inputs = {"messages": [HumanMessage(content="Public discussion.")]}
        context = {"phase": "propose_policy_maker", "prompt_args": ([1, 2],)}
        if use_async:
            first = await agent.ainvoke(inputs, config=config, context=context)
            second = await agent.ainvoke(
                {"messages": []}, config=config, context={"phase": "vote"}
            )
        else:
            first = agent.invoke(inputs, config=config, context=context)
            second = agent.invoke(
                {"messages": []}, config=config, context={"phase": "vote"}
            )
        assert isinstance(first["structured_response"], Nomination)
        assert isinstance(second["structured_response"], Vote)
        snapshot = await agent.aget_state(config)
        assert not any(
            isinstance(message, SystemMessage)
            for message in snapshot.values["messages"]
        )

    asyncio.run(scenario())

    assert (
        model.requests[0][0].content == "You are a red player.\n\nNominate from [1, 2]."
    )
    assert model.requests[1][0].content == "You are a red player.\n\nCast your vote."
    assert all(isinstance(request[0], SystemMessage) for request in model.requests)
    assert all(
        any(message.content == "Public discussion." for message in request)
        for request in model.requests
    )
