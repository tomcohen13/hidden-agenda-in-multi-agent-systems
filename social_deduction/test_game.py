"""Tests for the social deduction game's building blocks."""

import asyncio
import json
import logging
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from social_deduction.agent import create_game_agent
from social_deduction.game import Deck, Game, GameConfig, TerminationReason


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
    game = Game(GameConfig(model="scripted"))
    state = {
        "elected": elected,
        "proposed_policy_maker": candidate,
        "red_leader": 1,
        "enacted_policies": {"red": red_policies, "blue": 0},
        "consecutive_failed_elections": 1,
    }

    assert game.eval_election_results(state) == expected


def test_game_loads_prompts_and_compiles():
    game = Game(GameConfig(model="scripted"))

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
    "approvals, elected, failure_count, stop_after",
    [
        ([True, True, True, True, False, False], True, 0, "vote"),
        ([True, True, True, False, False, False], False, 3, "vote"),
        ([True, True, True, True, False, False], True, 0, "draw_and_pass"),
    ],
)
def test_game_runs_through_voting_with_real_prompts(
    monkeypatch, caplog, approvals, elected, failure_count, stop_after
):
    caplog.set_level(logging.INFO, logger="social_deduction.game")
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
        if name == 0:
            responses.append(AIMessage(content="", tool_calls=[{
                "name": "DrawAndPass", "id": "draw-and-pass",
                "args": {"passed_cards": ["red", "red"], "private_justification": "Private selection."},
            }]))
        assert "{red_players}" not in player_prompt
        assert "{red_leader}" not in player_prompt
        return create_game_agent(
            name, player_prompt, model=ScriptedGameModel(responses=responses),
            phase_prompts=phase_prompts,
        )

    monkeypatch.setattr("social_deduction.game.create_game_agent", scripted_agent)
    monkeypatch.setattr("social_deduction.game.random.choice", lambda choices: choices[0])
    game = Game(GameConfig(model="scripted", num_players=6, p_red=0.34))
    if stop_after == "draw_and_pass":
        game.deck.cards = [{"card_id": f"red-{i}", "policy": "red"} for i in range(3)]
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
        result = await game.graph.ainvoke(initial_state, interrupt_after=[stop_after])
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
            expected_ids = ["opening", nomination_id]
            if stop_after == "draw_and_pass" and player_id == 0:
                # The manager catches up on the published ballots before drawing.
                expected_ids += [message["id"] for message in result["messages"][2:]]
            assert delivered_ids == expected_ids
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
    assert "Round 1 · manager is player 0" in caplog.text
    assert "Nomination · player 0 proposes player 1" in caplog.text
    assert f"Election {'PASSED' if elected else 'REJECTED'}" in caplog.text
    assert "My public justification." not in caplog.text
    if stop_after == "draw_and_pass":
        assert result["drawn_cards"] == [
            {"card_id": f"red-{i}", "policy": "red"} for i in (2, 1, 0)
        ]
        assert result["passed_cards"] == result["drawn_cards"][:2]
        assert game.deck.is_empty()
        assert "Private selection." not in caplog.text


@pytest.fixture
def draw_game():
    def build(hand, selection):
        game = Game(GameConfig(model="scripted"))
        drawn = [{"card_id": f"card-{i}", "policy": policy} for i, policy in enumerate(hand)]
        game.deck.cards = [{"card_id": "remaining", "policy": "blue"}, *reversed(drawn)]
        manager = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values={})),
            ainvoke=AsyncMock(return_value={"structured_response": {
                "passed_cards": selection, "private_justification": "Private selection explanation.",
            }}),
        )
        policy_maker = SimpleNamespace(ainvoke=AsyncMock())
        game.agents = {0: manager, 1: policy_maker}
        state = {
            "game_id": "draw-test", "round_number": 1,
            "round_manager": 0, "proposed_policy_maker": 1,
            "elected": True, "drawn_cards": [], "passed_cards": [], "messages": [],
        }
        return game, state, drawn, manager, policy_maker

    return build


@pytest.mark.parametrize(
    "hand, selection, expected_indices",
    [
        (["red", "blue", "red"], ["red", "blue"], [0, 1]),
        (["red", "blue", "red"], ["blue", "red"], [1, 0]),
        (["red", "blue", "red"], ["red", "red"], [0, 2]),
        (["red", "red", "red"], ["red", "red"], [0, 1]),
        (["blue", "blue", "red"], ["blue", "blue"], [0, 1]),
    ],
)
def test_draw_and_pass_preserves_card_ids_and_private_visibility(
    draw_game, caplog, hand, selection, expected_indices
):
    caplog.set_level(logging.INFO, logger="social_deduction.game")
    game, state, drawn, manager, policy_maker = draw_game(hand, selection)

    result = asyncio.run(game.draw_and_pass(state))

    assert result["drawn_cards"] == drawn
    assert result["passed_cards"] == [drawn[i] for i in expected_indices]
    assert len({card["card_id"] for card in result["passed_cards"]}) == 2
    assert game.deck.cards == [{"card_id": "remaining", "policy": "blue"}]
    assert "messages" not in result
    assert state["drawn_cards"] == state["passed_cards"] == state["messages"] == []
    kwargs = manager.ainvoke.await_args.kwargs
    assert kwargs["config"]["configurable"]["thread_id"] == "draw-test:0"
    assert kwargs["context"] == {"phase": "draw_and_pass", "prompt_kwargs": {"drawn_cards": hand}}
    policy_maker.ainvoke.assert_not_awaited()
    assert "Private selection explanation." not in caplog.text
    assert "card-" not in caplog.text


@pytest.mark.parametrize("selection", [["blue", "blue"], ["blue", "red"]])
def test_draw_and_pass_rejects_unavailable_color_counts(draw_game, selection):
    game, state, _, _, _ = draw_game(["red", "red", "red"], selection)

    with pytest.raises(ValueError, match="available in the drawn hand"):
        asyncio.run(game.draw_and_pass(state))


@pytest.mark.parametrize("selection", [[], ["red"], ["red"] * 3, ["red", "green"]])
def test_draw_and_pass_rejects_malformed_selection(draw_game, selection):
    game, state, _, _, _ = draw_game(["red", "blue", "red"], selection)

    with pytest.raises(ValidationError):
        asyncio.run(game.draw_and_pass(state))


@pytest.fixture
def enact_game():
    def build(hand, policy):
        game = Game(GameConfig(model="scripted"))
        policy_maker = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values={})),
            ainvoke=AsyncMock(return_value={"structured_response": {
                "policy": policy, "private_justification": "Private enactment explanation.",
            }}),
        )
        manager = SimpleNamespace(ainvoke=AsyncMock())
        game.agents = {0: manager, 1: policy_maker}
        state = {
            "game_id": "enact-test", "round_number": 2,
            "round_manager": 0, "proposed_policy_maker": 1,
            "passed_cards": [{"card_id": f"card-{i}", "policy": color} for i, color in enumerate(hand)],
            "drawn_cards": [{"card_id": "private-manager-card", "policy": "red"}],
            "enacted_policies": {"red": 2, "blue": 1}, "messages": [],
        }
        return game, state, manager, policy_maker

    return build


@pytest.mark.parametrize(
    "hand, policy",
    [(["red", "blue"], "red"), (["red", "blue"], "blue"),
     (["red", "red"], "red"), (["blue", "blue"], "blue")],
)
def test_enact_updates_only_selected_total_and_publishes_only_outcome(
    enact_game, caplog, hand, policy
):
    caplog.set_level(logging.INFO, logger="social_deduction.game")
    game, state, manager, policy_maker = enact_game(hand, policy)
    before = deepcopy(state)

    result = asyncio.run(game.enact(state))

    expected = {"red": 2, "blue": 1}
    expected[policy] += 1
    assert result["enacted_policies"] == expected
    assert state == before
    assert len(result["messages"]) == 1
    message = result["messages"][0]
    assert message["id"]
    assert message["player_id"] == 1
    assert message["round_number"] == 2
    assert message["content"] == f"I enacted policy: {policy}."
    assert "Private enactment explanation." not in caplog.text
    assert f"player 1 chose {policy}" in caplog.text
    kwargs = policy_maker.ainvoke.await_args.kwargs
    assert kwargs["config"]["configurable"]["thread_id"] == "enact-test:1"
    assert kwargs["context"] == {"phase": "enact", "prompt_kwargs": {"passed_cards": hand}}
    manager.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("hand, policy", [(["red", "red"], "blue"), (["blue", "blue"], "red")])
def test_enact_rejects_policy_not_in_hand(enact_game, hand, policy):
    game, state, _, _ = enact_game(hand, policy)
    before = deepcopy(state)

    with pytest.raises(ValueError, match="available in the passed hand"):
        asyncio.run(game.enact(state))

    assert state == before


@pytest.mark.parametrize("hand", [[], ["red"], ["red", "red", "blue"]])
def test_enact_requires_two_passed_cards(enact_game, hand):
    game, state, _, policy_maker = enact_game(hand, "red")

    with pytest.raises(ValueError, match="exactly two passed cards"):
        asyncio.run(game.enact(state))

    policy_maker.ainvoke.assert_not_awaited()


def test_enact_rejects_unknown_policy_color(enact_game):
    game, state, _, _ = enact_game(["red", "blue"], "green")

    with pytest.raises(ValidationError):
        asyncio.run(game.enact(state))


@pytest.mark.parametrize(
    "red, blue, expected",
    [(0, 0, "continue"), (4, 4, "continue"), (5, 2, "game_over"),
     (2, 5, "game_over"), (6, 2, "game_over")],
)
def test_policy_victory_threshold_routes_to_reporting(red, blue, expected):
    game = Game(GameConfig(model="scripted"))

    assert game.is_game_over({"enacted_policies": {"red": red, "blue": blue}}) == expected


@pytest.mark.parametrize(
    "red, blue, elected, candidate, winner, reason",
    [
        (5, 2, None, 2, "red", TerminationReason.RED_POLICY_THRESHOLD),
        (2, 5, None, 2, "blue", TerminationReason.BLUE_POLICY_THRESHOLD),
        (3, 1, True, 1, "red", TerminationReason.RED_LEADER_ELECTED),
        (4, 1, True, 1, "red", TerminationReason.RED_LEADER_ELECTED),
    ],
)
def test_report_game_over_records_and_logs_outcome(
    caplog, red, blue, elected, candidate, winner, reason
):
    caplog.set_level(logging.INFO, logger="social_deduction.game")
    game = Game(GameConfig(model="scripted"))
    state = {
        "enacted_policies": {"red": red, "blue": blue},
        "elected": elected, "proposed_policy_maker": candidate, "red_leader": 1,
        "winner": None, "termination_reason": None,
    }
    before = deepcopy(state)

    result = game.report_game_over(state)

    assert result == {"winner": winner, "termination_reason": reason}
    assert json.loads(json.dumps(result))["termination_reason"] == reason.value
    assert f"Game over · {winner} wins · {reason.value}" in caplog.text
    assert state == before


@pytest.mark.parametrize("red, elected, candidate", [(2, True, 1), (3, False, 1), (3, True, 2)])
def test_report_game_over_rejects_nonterminal_state(red, elected, candidate):
    game = Game(GameConfig(model="scripted"))
    state = {
        "enacted_policies": {"red": red, "blue": 0},
        "elected": elected, "proposed_policy_maker": candidate, "red_leader": 1,
    }

    with pytest.raises(ValueError, match="without a victory condition"):
        game.report_game_over(state)


@pytest.mark.parametrize("reason", list(TerminationReason))
def test_terminal_routes_report_once_and_reach_end(monkeypatch, reason):
    game = Game(GameConfig(model="scripted"))
    leader_victory = reason == TerminationReason.RED_LEADER_ELECTED
    winning_color = "blue" if reason == TerminationReason.BLUE_POLICY_THRESHOLD else "red"
    counts = {"red": 0, "blue": 0}
    counts[winning_color] = 3 if leader_victory else 4
    state = {
        "round_number": 1, "elected": True,
        "proposed_policy_maker": 1 if leader_victory else 2, "red_leader": 1,
        "enacted_policies": counts, "winner": None, "termination_reason": None,
    }
    # Stub completed gameplay actions; exercise the real routers and reporting node.
    for node in ("initialize_players", "start_round", "propose_policy_maker", "collect_votes"):
        monkeypatch.setattr(game, node, lambda state: {})

    def draw(state):
        assert not leader_victory, "Leader victory must bypass policy selection."
        return {}

    def enact(state):
        assert not leader_victory, "Leader victory must bypass enactment."
        updated = state["enacted_policies"].copy()
        updated[winning_color] += 1
        return {"enacted_policies": updated}

    reports = []
    original_report = game.report_game_over

    def report(state):
        result = original_report(state)
        reports.append(result)
        return result

    def unexpected_discussion(state):
        pytest.fail("A terminal game must not proceed to discussion.")

    monkeypatch.setattr(game, "draw_and_pass", draw)
    monkeypatch.setattr(game, "enact", enact)
    monkeypatch.setattr(game, "report_game_over", report)
    monkeypatch.setattr(game, "discussion", unexpected_discussion)
    game.initialize()

    result = asyncio.run(game.graph.ainvoke(state))

    assert len(reports) == 1
    assert result["winner"] == winning_color
    assert result["termination_reason"] == reason
