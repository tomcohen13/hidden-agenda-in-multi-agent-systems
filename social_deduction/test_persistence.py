"""Verify game and agent checkpoints survive reopening the SQLite database."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from social_deduction.agent import DrawAndPass, create_game_agent
from social_deduction.game import Game, GameConfig


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("invalid_hand", [False, True])
def test_game_and_agent_state_persist_after_run_or_invalid_action(
    tmp_path, monkeypatch, invalid_hand
):
    selected = ["blue", "blue"] if invalid_hand else ["red", "red"]

    def response(name, args):
        return AIMessage(content="", tool_calls=[{"name": name, "id": name, "args": args}])

    def scripted_agent(name, player_prompt, *, model, phase_prompts, checkpointer):
        responses = []
        if name == 0:
            responses.append(response("Nomination", {"policy_maker_id": 1, "justification": "Test nominee."}))
        responses.append(response("Vote", {"approve": True, "justification": "Test approval."}))
        if name == 0:
            responses.append(response("DrawAndPass", {
                "passed_cards": selected, "private_justification": "Explanation of the attempted selection.",
            }))
        return create_game_agent(
            name, player_prompt, model=ScriptedModel(responses=responses),
            phase_prompts=phase_prompts, checkpointer=checkpointer,
        )

    monkeypatch.setattr("social_deduction.game.create_game_agent", scripted_agent)
    monkeypatch.setattr("social_deduction.game.random.choice", lambda choices: choices[0])
    database = str(tmp_path / "checkpoints.sqlite")
    cfg = GameConfig(model="scripted", num_players=6, p_red=0.34)
    game_config = {"configurable": {"thread_id": "persist-test:game"}}
    agent_config = {"configurable": {"thread_id": "persist-test:0"}}

    async def scenario():
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            game = Game(cfg, checkpointer=saver)
            game.deck.cards = [{"card_id": f"red-{i}", "policy": "red"} for i in range(3)]
            game.initialize()
            initial_state = {
                "game_id": "persist-test", "round_number": 0, "players": {},
                "round_manager": None, "proposed_policy_maker": None, "last_policy_maker": None,
                "votes": [], "elected": None, "consecutive_failed_elections": 0,
                "drawn_cards": [], "passed_cards": [], "enacted_policies": {"red": 0, "blue": 0},
                "messages": [], "winner": None, "termination_reason": None,
            }
            if invalid_hand:
                with pytest.raises(ValueError, match="available in the drawn hand"):
                    await game.graph.ainvoke(initial_state, config=game_config)
            else:
                await game.graph.ainvoke(initial_state, config=game_config, interrupt_after=["draw_and_pass"])

        # Fresh graphs and a fresh connection; nothing is recovered from the old runtimes.
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            reopened = Game(cfg, checkpointer=saver)
            reopened.initialize()
            outer = await reopened.graph.aget_state(game_config)
            assert outer.values["game_id"] == "persist-test"
            assert outer.values["elected"] is True
            assert len(outer.values["votes"]) == 6
            assert len(outer.values["drawn_cards"]) == (0 if invalid_hand else 3)

            agent = create_game_agent(
                0, "Inspection only.", model=ScriptedModel(responses=[AIMessage(content="unused")]),
                phase_prompts=reopened.prompts, checkpointer=saver,
            )
            snapshot = await agent.aget_state(agent_config)
            assert snapshot.values["player_id"] == 0
            assert snapshot.values["phase"] == "draw_and_pass"
            assert snapshot.values["observation"] == {"drawn_cards": ["red", "red", "red"]}
            assert "prompt_kwargs" not in snapshot.values
            decision = snapshot.values["structured_response"]
            assert isinstance(decision, DrawAndPass)
            assert decision.passed_cards == selected
            assert decision.private_justification == "Explanation of the attempted selection."
            assert (await agent.aget_state({"configurable": {"thread_id": "other-game:0"}})).values == {}

    asyncio.run(scenario())
