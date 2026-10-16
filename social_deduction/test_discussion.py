"""Exercise discussion prompts and visibility with scripted agent runtimes."""

import asyncio
import json
from copy import deepcopy

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from social_deduction.agent import create_game_agent
from social_deduction.game import Game, GameConfig


class DiscussionModel(FakeMessagesListChatModel):
    requests: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        assert [tool.name for tool in tools] == ["Response"]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("rounds, seconds", [(2, 60), (0, 60), (2, 0)])
def test_discussion_stages_share_only_authorized_messages(monkeypatch, rounds, seconds):
    monkeypatch.setattr("social_deduction.game.random.shuffle", lambda values: None)
    game = Game(GameConfig(model="scripted", num_players=5, p_red=0.4))
    models = {}
    for player in range(5):
        models[player] = DiscussionModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "Response", "id": f"response-{player}-{turn}",
                "args": {"content": f"player-{player}-turn-{turn}"},
            }])
            for turn in range(3)
        ])
        game.agents[player] = create_game_agent(
            player, f"You are player {player}.", model=models[player], phase_prompts=game.prompts
        )

    drawn = [{"card_id": f"card-{i}", "policy": color}
             for i, color in enumerate(["red", "blue", "blue"])]
    opening = {"id": "enacted", "player_id": 1, "round_number": 1,
               "content": "I enacted policy: red."}
    state = {
        "game_id": "discussion-test", "round_number": 1,
        "players": {0: "red", 1: "blue", 2: "red", 3: "blue", 4: "blue"},
        "round_manager": 0, "proposed_policy_maker": 1,
        "drawn_cards": drawn, "passed_cards": drawn[:2],
        "enacted_policies": {"red": 1, "blue": 0}, "messages": [opening],
        "discussion_rounds_limit": rounds, "discussion_time_limit": seconds,
    }
    before = deepcopy(state)

    update = asyncio.run(game.discussion(state))

    effective_rounds = rounds if seconds else 0
    messages = update["messages"]
    expected_players = [1, 0, 2, 3, 4] + [2, 3, 4, 1, 0] * effective_rounds
    assert [message["player_id"] for message in messages] == expected_players
    assert len({message["id"] for message in messages}) == len(messages)
    assert all(message["round_number"] == 1 for message in messages)
    assert set(update) == {"messages"}
    assert state == before
    json.dumps(update)  # Public IDs and messages remain JSON serializable.

    # Each observer's first response sees both openings, but no parallel reactions.
    first_opening_ids = {"enacted", messages[0]["id"], messages[1]["id"]}
    for player in (2, 3, 4):
        seen = {message.additional_kwargs["public_message_id"]
                for message in models[player].requests[0]
                if "public_message_id" in message.additional_kwargs}
        assert seen == first_opening_ids

    # The manager sees the policy maker's opening before giving their account.
    manager_seen = {message.additional_kwargs["public_message_id"]
                    for message in models[0].requests[0]
                    if "public_message_id" in message.additional_kwargs}
    assert manager_seen == {"enacted", messages[0]["id"]}

    for player, model in models.items():
        for turn, request in enumerate(model.requests):
            system_prompt = request[0].content
            stage = ("opening" if player in (0, 1) else "reaction") if turn == 0 else "round-robin"
            assert game.discussion_stage_instructions[stage] in system_prompt
            assert "Round manager: player 0" in system_prompt
            assert "Policy maker: player 1" in system_prompt
            if player == 0:
                assert "The cards you drew: red, blue, blue." in system_prompt
                assert "The cards you passed to the policy maker: red, blue." in system_prompt
                assert "The cards you received from the round manager:" not in system_prompt
            elif player == 1:
                assert "The cards you received from the round manager: red, blue." in system_prompt
                assert "The cards you drew:" not in system_prompt
                assert "The cards you passed to the policy maker:" not in system_prompt
            else:
                assert "### Your private observations from this round" not in system_prompt
                assert "The cards you drew:" not in system_prompt
                assert "The cards you received from the round manager:" not in system_prompt
                assert "The cards you passed to the policy maker:" not in system_prompt

    if effective_rounds:
        # A sequential speaker sees the immediately preceding player's new response.
        previous_response_id = messages[5]["id"]  # Player 2's first round-robin turn.
        assert any(message.additional_kwargs.get("public_message_id") == previous_response_id
                   for message in models[3].requests[1])
