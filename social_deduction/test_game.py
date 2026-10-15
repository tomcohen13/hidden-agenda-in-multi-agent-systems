"""Tests for the social deduction game's building blocks."""

import asyncio
import json
from collections import Counter

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from social_deduction.agent import create_game_agent
from social_deduction.game import Deck, Game, GameConfig


class TestDeck:
    @pytest.mark.parametrize("blue, red", [(6, 11), (2, 3), (0, 3), (3, 0)])
    def test_initial_composition_and_unique_card_ids(self, blue, red):
        deck = Deck(blue=blue, red=red)

        assert len(deck.cards) == blue + red
        assert Counter(card["policy"] for card in deck.cards) == Counter(
            {"blue": blue, "red": red}
        )
        assert len({card["card_id"] for card in deck.cards}) == blue + red
        assert not deck.is_empty()

    def test_draw_three_removes_the_top_three_cards(self):
        deck = Deck(blue=6, red=11)
        before = deck.cards.copy()

        drawn = deck.draw_three()

        assert drawn == list(reversed(before[-3:]))
        assert deck.cards == before[:-3]

    def test_drawing_the_last_three_cards_empties_the_deck(self):
        deck = Deck(blue=1, red=2)

        drawn = deck.draw_three()

        assert len(drawn) == 3
        assert deck.is_empty()

    def test_drawing_from_an_exhausted_deck_refills_it(self):
        deck = Deck(blue=1, red=2)
        deck.draw_three()

        drawn = deck.draw_three()

        assert len(drawn) == 3
        assert Counter(card["policy"] for card in drawn) == Counter({"blue": 1, "red": 2})
        assert deck.is_empty()

    def test_refill_mid_draw_preserves_the_remaining_card(self):
        deck = Deck(blue=2, red=2)
        deck.draw_three()
        remaining_card = deck.cards[0]

        drawn = deck.draw_three()

        assert len(drawn) == 3
        assert drawn[0] is remaining_card
        assert len(deck.cards) == 2
        assert Counter(card["policy"] for card in [*drawn[1:], *deck.cards]) == Counter(
            {"blue": 2, "red": 2}
        )


@pytest.mark.parametrize(
    "elected, candidate, red_policies, expected",
    [
        (True, 1, 0, "continue"),
        (True, 1, 2, "continue"),
        (True, 1, 3, "red_leader_elected"),
        (True, 1, 4, "red_leader_elected"),
        (True, 2, 3, "continue"),
        (False, 1, 3, "candidate_rejected"),
    ],
)
def test_election_result_requires_elected_leader_and_three_red_policies(
    elected, candidate, red_policies, expected
):
    game = Game.__new__(Game)
    state = {
        "elected": elected,
        "proposed_policy_maker": candidate,
        "red_leader": 1,
        "enacted_policies": {"red": red_policies, "blue": 0},
    }

    assert game.eval_election_results(state) == expected


def test_game_loads_prompts_and_compiles():
    game = Game(GameConfig())

    game.initialize()

    assert game.graph is not None
    assert {"player", "propose_policy_maker", "vote"} <= game.prompts.keys()
    assert all(game.prompts[name].strip() for name in ("player", "propose_policy_maker", "vote"))
    diagram = game.graph.get_graph().draw_mermaid()
    assert "candidate_rejected" in diagram


class ScriptedGameModel(FakeMessagesListChatModel):
    """Support structured-output tools without contacting a model provider."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize(
    "approvals, elected, failure_count",
    [
        ([True, True, True, True, False, False], True, 0),
        ([True, True, True, False, False, False], False, 3),
    ],
)
def test_game_runs_through_voting_with_real_prompts(
    monkeypatch, approvals, elected, failure_count
):
    def scripted_agent(name, player_prompt, *, model, phase_prompts):
        responses = []
        if name == 0:
            responses.append(AIMessage(content="", tool_calls=[{
                "name": "Nomination",
                "id": "nomination",
                "args": {"policy_maker_id": 1, "justification": "I propose player 1."},
            }]))
        responses.append(AIMessage(content="", tool_calls=[{
            "name": "Vote",
            "id": f"vote-{name}",
            "args": {"approve": approvals[name], "justification": "My public justification."},
        }]))
        assert "{red_players}" not in player_prompt
        assert "{red_leader}" not in player_prompt
        return create_game_agent(
            name, player_prompt, model=ScriptedGameModel(responses=responses),
            phase_prompts=phase_prompts,
        )

    monkeypatch.setattr("social_deduction.game.create_game_agent", scripted_agent)
    monkeypatch.setattr("social_deduction.game.random.choice", lambda choices: choices[0])
    game = Game(GameConfig(num_players=6, p_red=0.34))
    game.initialize()
    opening = {"id": "opening", "player_id": 5, "round_number": 0, "content": "Ready."}
    initial_state = {
        "game_id": "test-game",
        "players": {},
        "red_leader": None,
        "round_number": 0,
        "round_manager": None,
        "proposed_policy_maker": None,
        "last_policy_maker": None,
        "votes": [],
        "elected": None,
        "consecutive_failed_elections": 2,
        "drawn_cards": [],
        "passed_cards": [],
        "enacted_policies": {"red": 0, "blue": 0},
        "messages": [opening],
    }

    async def scenario():
        result = await game.graph.ainvoke(initial_state, interrupt_after=["vote"])
        nomination_id = result["messages"][1]["id"]
        for player_id, agent in game.agents.items():
            snapshot = await agent.aget_state({
                "configurable": {"thread_id": f"test-game:{player_id}"}
            })
            delivered_ids = [
                message.additional_kwargs["public_message_id"]
                for message in snapshot.values["messages"]
                if "public_message_id" in message.additional_kwargs
            ]
            # Everyone sees the same board, without any current-election ballots.
            assert delivered_ids == ["opening", nomination_id]
        return result

    result = asyncio.run(scenario())

    assert len(game.agents) == 6
    assert Counter(result["players"].values()) == {"red": 2, "blue": 4}
    assert result["players"][result["red_leader"]] == "red"
    assert result["round_number"] == 1
    assert result["round_manager"] == 0
    assert result["proposed_policy_maker"] == 1
    assert result["elected"] is elected
    assert result["consecutive_failed_elections"] == failure_count
    assert [vote["player_id"] for vote in result["votes"]] == list(range(6))
    assert [vote["approve"] for vote in result["votes"]] == approvals
    assert len(result["messages"]) == 8  # Opening, nomination, then six ballots.
    assert result["messages"][0] == opening
    assert [json.loads(message["content"]) for message in result["messages"][2:]] == result["votes"]
    assert len({message["id"] for message in result["messages"]}) == 8
    assert initial_state["messages"] == [opening]
