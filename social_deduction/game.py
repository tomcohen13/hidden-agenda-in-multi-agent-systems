"""Base models for experiment"""
import asyncio
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from operator import add
import time
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from social_deduction.agent import (
    Enact, Nomination, PublicMessage, Response, Vote, DrawAndPass,
    CaptureModelInput, catch_up_messages, create_game_agent
)
from config import SOCIAL_DEDUCTION_PATH

PROMPTS_DIR = SOCIAL_DEDUCTION_PATH / "prompts"
POLICY_WIN_THRESHOLD = 5
RED_LEADER_ELECTION_THRESHOLD = 3


class TerminationReason(StrEnum):
    RED_POLICY_THRESHOLD = "red_policy_threshold"
    BLUE_POLICY_THRESHOLD = "blue_policy_threshold"
    RED_LEADER_ELECTED = "red_leader_elected"


POLICY_TERMINATION_REASONS = {
    "red": TerminationReason.RED_POLICY_THRESHOLD,
    "blue": TerminationReason.BLUE_POLICY_THRESHOLD,
}

class PolicyCard(TypedDict):
    card_id: str
    policy: Literal["red", "blue"]

class Deck:
    """
    A standard deck contains 6 blue and 11 red cards,
    introducing an inherent bias toward the red (minority) group.

    A Deck exposes the following functionalities:
        draw: pops and returns the top card 

    """
    def __init__(self, blue: int, red: int):
        self.cards = []
        self.blue = blue
        self.red = red

        self._initialize_deck()

    def _initialize_deck(self):
        for i in range(self.blue):
            self.cards.append(PolicyCard(card_id=f"blue_{i}", policy="blue"))
        for j in range(self.red):
            self.cards.append(PolicyCard(card_id=f"red_{j}", policy="red"))
        random.shuffle(self.cards)

    def is_empty(self):
        return len(self.cards) == 0

    def draw_one(self) -> PolicyCard:
        if self.is_empty():
            self._initialize_deck()
        return self.cards.pop()

    def draw_three(self) -> list[PolicyCard]:
        """Pop and return the top card in the deck."""
        drawn = []
        for _ in range(3):
            drawn.append(self.draw_one())
        return drawn


class GameState(TypedDict):

    game_id: str
    players: dict # {player_id: team_assignment}
    red_leader: int
    round_number: int
    round_manager: int
    proposed_policy_maker: int
    last_policy_maker: int  # can't be policy-maker twice in a row
    votes: list[dict]
    elected: bool | None
    consecutive_failed_elections: int
    drawn_cards: list[PolicyCard]
    passed_cards: list[PolicyCard]
    enacted_policies: dict
    messages: Annotated[list[PublicMessage], add]
    winner: Literal["blue", "red"] | None
    termination_reason: TerminationReason | None


@dataclass
class GameConfig:
    model: str
    blue_cards: int = 6
    red_cards: int = 11
    num_players: int = 8
    p_red: float = 0.4

    discussion_rounds_limit: int = 1
    discussion_time_limit: int = 60  # in seconds


class Game:

    def __init__(
        self,
        config: GameConfig,
        checkpointer: BaseCheckpointSaver | None = None,
        logger: logging.Logger | None = None,
        prompts_dir=None,
    ):
        self.config = config
        self.prompts_dir = PROMPTS_DIR if prompts_dir is None else prompts_dir
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.agents: dict[int, CompiledStateGraph] = {}
        self.graph = None
        self.checkpointer = checkpointer
        self.deck = Deck(config.blue_cards, config.red_cards)
        self.prompts: dict = self._load_prompts()
        self.role_instructions = json.loads(
            (self.prompts_dir / "role_instructions.json").read_text(encoding="utf-8")
        )
        self.discussion_stage_instructions = json.loads(
            (self.prompts_dir / "discussion_stages.json").read_text(encoding="utf-8")
        )
        self.discussion_private_context_templates = json.loads(
            (self.prompts_dir / "discussion_private_context.json").read_text(encoding="utf-8")
        )

    def _load_prompts(self) -> None:
        prompts = {}
        paths = self.prompts_dir.glob("*.md")
        for path in paths:
            prompts[path.stem] = path.read_text()
        return prompts
        

    def initialize(self):
        """Compile the game blueprint for inspection and visualization."""
        builder = StateGraph(GameState)
        builder.add_node("initialize_players", self.initialize_players)
        builder.add_node("start_round", self.start_round)  # rotate round manager, increment round number
        builder.add_node("propose_policy_maker", self.propose_policy_maker) # round manager to propose enacter
        builder.add_node("vote", self.collect_votes)
        builder.add_node("election_deadlock", self.election_deadlock)
        builder.add_node("draw_and_pass", self.draw_and_pass) # round manager draws three policies, passes 2 to enacter 
        builder.add_node("enact", self.enact) 
        builder.add_node("discussion", self.discussion)
        builder.add_node("report_game_over", self.report_game_over)

        builder.add_edge(START, "initialize_players")
        builder.add_edge("initialize_players", "start_round")
        builder.add_edge("start_round", "propose_policy_maker")
        builder.add_edge("propose_policy_maker", "vote")
        builder.add_conditional_edges(
            "vote",
            self.eval_election_results,
            {
                TerminationReason.RED_LEADER_ELECTED: "report_game_over",
                "candidate_rejected": "propose_policy_maker",
                "election_deadlock": "election_deadlock",
                "continue": "draw_and_pass",
            },
        )
        builder.add_edge("draw_and_pass", "enact")
        builder.add_conditional_edges(
            "election_deadlock", self.is_game_over,
            {"game_over": "report_game_over", "continue": "start_round"},
        )
        builder.add_conditional_edges(
            "enact",
            self.is_game_over,
            {"game_over": "report_game_over", "continue": "discussion"},
        )
        builder.add_edge("discussion", "start_round")
        builder.add_edge("report_game_over", END)
        self.graph = builder.compile(checkpointer=self.checkpointer)
        self.logger.debug("Game graph compiled")


    def initialize_players(self, state: GameState) -> dict[str, Any]:
        """Assign teams and the red leader, then create role-specific agents."""

        cfg = self.config
        num_red_players = int(cfg.num_players * cfg.p_red)
        self.logger.info("Setup · creating %s player runtimes", cfg.num_players)

        assert num_red_players >= 2, "Must have at least 2 red players in the game."
        red_leader = random.randint(0, num_red_players - 1)

        # load prompt
        player_template = self.prompts["player"]
        phase_prompts = {
            name: template for name, template in self.prompts.items() if name != "player"
        }

        # initialize agents
        player_assignments = ["red"] * num_red_players + ["blue"] * (cfg.num_players - num_red_players)
        random.shuffle(player_assignments)
        red_leader = player_assignments.index("red") # first occurrence

        for i, assignment in enumerate(player_assignments):
            role = f"{assignment}_{'leader' if i == red_leader else 'regular'}"
            role_instructions = self.role_instructions[role].copy()
            if role == "red_regular":
                other_red_players = [
                    k for k, v in enumerate(player_assignments)
                    if v == "red" and k != i
                ]
                role_instructions = {
                    key: value.format(red_players=other_red_players, red_leader=red_leader)
                    for key, value in role_instructions.items()
                }
            role_instructions = json.dumps(role_instructions)
            player_prompt = player_template.format(
                player_id=i,
                role_instructions=role_instructions,
                num_players=cfg.num_players,
                p_red=cfg.p_red,
                num_red_players=num_red_players,
                num_blue_players=cfg.num_players - num_red_players,
                blue_cards=cfg.blue_cards,
                red_cards=cfg.red_cards,
            )
            agent = create_game_agent(
                name=i,
                player_prompt=player_prompt,
                model=cfg.model,
                phase_prompts=phase_prompts,
                checkpointer=self.checkpointer,
            )
            self.agents[i] = agent
        self.logger.info(
            "Ready · %s blue / %s red players · deck %s blue / %s red",
            cfg.num_players - num_red_players, num_red_players, cfg.blue_cards, cfg.red_cards,
        )
        return {
            "players": dict(enumerate(player_assignments)),
            "red_leader": red_leader,
        }

    def start_round(self, state: GameState) -> dict[str, Any]:
        """Increment the round number and rotate the round manager."""
        if state.get("round_manager") is None:
            assert state.get("round_number") == 0, "game cannot be ongoing if a round manager isn't selected"
            # first round, pick manager at random
            next_round_manager = random.choice(list(state["players"].keys()))
        else:
            next_round_manager = (state["round_manager"] + 1) % len(state["players"])
        self.logger.info(
            "Round %s · manager is player %s", state["round_number"] + 1, next_round_manager
        )
        return {
            "round_number": state["round_number"] + 1,
            "round_manager": next_round_manager,
            "last_policy_maker": (
                state.get("proposed_policy_maker") if state.get("elected")
                else state.get("last_policy_maker")
            ),
            "proposed_policy_maker": None,
            "elected": None,
        }

    async def propose_policy_maker(self, state: GameState) -> dict[str, Any]:
        """Ask the round manager to nominate an eligible policy enacter."""

        round_manager_id = state["round_manager"]
        all_players = set(state["players"].keys()) 
        valid_players = all_players - {round_manager_id, state["last_policy_maker"]}
        if not valid_players:
            raise ValueError("No eligible policy maker is available.")
        self.logger.info(
            "Nomination · waiting for player %s · %s eligible candidates",
            round_manager_id, len(valid_players),
        )
        round_manager_runner = self.agents[round_manager_id]
        config = {
            "configurable": {"thread_id": f"{state['game_id']}:{round_manager_id}"}
        }
        new_messages = await catch_up_messages(round_manager_runner, config, state["messages"])
        context = {
            "phase": "propose_policy_maker",
            "prompt_kwargs": {"valid_players": sorted(valid_players)},
        }
        res = await round_manager_runner.ainvoke(
            {
                "messages": new_messages,
                "player_id": round_manager_id,
                "phase": context["phase"],
                "observation": context["prompt_kwargs"],
            },
            config=config,
            context=context,
        )
        nomination = Nomination.model_validate(res["structured_response"])
        if nomination.policy_maker_id not in valid_players:
            raise ValueError(f"Ineligible policy maker: {nomination.policy_maker_id}")
        self.logger.info(
            "Nomination · player %s proposes player %s",
            round_manager_id, nomination.policy_maker_id,
        )
        return {
            "proposed_policy_maker": nomination.policy_maker_id,
            "elected": None,
            "messages": [{
                "id": str(uuid4()),
                "player_id": round_manager_id,
                "round_number": state["round_number"],
                "content": (f"I nominate player {nomination.policy_maker_id} as policy maker.\n"
                            f"{nomination.justification}"),
            }],
        }


    async def collect_votes(self, state: GameState) -> dict[str, Any]:
        """Collect votes from a shared snapshot and resolve the nomination."""

        self.logger.info(
            "Voting · %s players deciding on player %s",
            len(state["players"]), state["proposed_policy_maker"],
        )
        async def _vote(player_id: int, state: GameState) -> dict:

            agent_runner = self.agents[player_id]
            config = {"configurable": {"thread_id": f"{state['game_id']}:{player_id}"}}
            
            new_messages = await catch_up_messages(agent_runner, config, state["messages"])
            context={
                "phase": "vote",
                "prompt_kwargs": {"proposed_policy_maker": state["proposed_policy_maker"]},
            }
            res = await agent_runner.ainvoke(
                input={
                    "messages": new_messages,
                    "player_id": player_id,
                    "phase": context["phase"],
                    "observation": context["prompt_kwargs"],
                },
                config=config,
                context=context,
            )
            vote = Vote.model_validate(res["structured_response"])
            return {"player_id": player_id, **vote.model_dump()}

        # collect votes from everyone
        votes: list[dict] = await asyncio.gather(
            *(_vote(player_id, state) for player_id in state["players"])
        )

        # TODO: retry if any failed, make sure there's num_players votes...
        
        tally = sum(v["approve"] for v in votes)
        elected = tally > len(votes) / 2
        self.logger.info(
            "Election %s · %s approve / %s reject",
            "PASSED" if elected else "REJECTED", tally, len(votes) - tally,
        )
        
        # TODO: add a system message about election result

        consecutive_failed_elections = 0 if elected else state["consecutive_failed_elections"] + 1

        return {
            "elected": elected,
            "consecutive_failed_elections": consecutive_failed_elections,
            "votes": votes,
            "messages": [
                PublicMessage(
                    id=str(uuid4()),
                    player_id=v["player_id"],
                    content=json.dumps(v),
                    round_number=state["round_number"],
                )
                for v in votes
            ] + [
                PublicMessage(
                    id=str(uuid4()),
                    player_id=-1,
                    round_number=state["round_number"],
                    content=(
                        f"Election {'PASSED' if elected else 'REJECTED'} · "
                        f"{tally} approved / {len(votes) - tally} rejected"
                    ),
                )
            ]
        }


    async def draw_and_pass(self, state: GameState) -> dict[str, Any]:
        """Let the round manager draw three policies and privately pass two."""

        # draw three cards
        drawn_cards = self.deck.draw_three()
        drawn_cards_strings = [f"{i}: {card['policy']}" for i, card in enumerate(drawn_cards)]
        
        round_manager_id = state["round_manager"]
        round_manager_runner = self.agents[round_manager_id]
        config = {
            "configurable": {"thread_id": f"{state['game_id']}:{round_manager_id}"},
        }
        self.logger.info("Policy selection · waiting for player %s to pass two cards", round_manager_id)

        new_messages = await catch_up_messages(round_manager_runner, config, state["messages"])
        context = {
            "phase": "draw_and_pass",
            "prompt_kwargs": {"drawn_cards": "\n".join(drawn_cards_strings)},
        }
        res = await round_manager_runner.ainvoke(
            {
                "messages": new_messages,
                "player_id": round_manager_id,
                "phase": context["phase"],
                "observation": context["prompt_kwargs"],
            },
            config=config,
            context=context,
        )
        decision = DrawAndPass.model_validate(res["structured_response"])
        indices = decision.passed_cards
        if len(set(indices)) != 2 or any(i not in {0,1,2} for i in indices):
            raise ValueError(
                f"Select two distinct card indices; got {indices}."
            )
        passed_cards: list[PolicyCard] = [drawn_cards[i] for i in indices]

        # log decision and reasoning
        self.logger.debug(
            "Private policy selection · player %s drew %s and passed %s · justification: %s",
            round_manager_id, drawn_cards, passed_cards, decision.private_justification,
        )
        self.logger.info(
            "Policy selection · player %s passed two cards to player %s",
            round_manager_id, state["proposed_policy_maker"],
        )
        return {
            "drawn_cards": drawn_cards,
            "passed_cards": passed_cards,
        }


    async def enact(self, state: GameState) -> dict[str, Any]:
        """Let the enacter choose a passed policy and publish its outcome."""

        passed_cards_strings = [card['policy'] for card in state["passed_cards"]]
        policy_maker_id = state["proposed_policy_maker"]
        policy_maker_runner = self.agents[policy_maker_id]
        config = {
            "configurable": {"thread_id": f"{state['game_id']}:{policy_maker_id}"},
        }

        self.logger.info("Enacting · waiting for player %s to enact one card", policy_maker_id)
        new_messages = await catch_up_messages(policy_maker_runner, config, state["messages"])
        
        context = {
            "phase": "enact",
            "prompt_kwargs": {
                "passed_cards": "\n".join(
                    f"{i}: {policy}" for i, policy in enumerate(passed_cards_strings)
                )
            },
        }
        res = await policy_maker_runner.ainvoke(
            input={
                "messages": new_messages,
                "player_id": policy_maker_id,
                "phase": context["phase"],
                "observation": context["prompt_kwargs"],
            },
            config=config,
            context=context,
        )
        decision = Enact.model_validate(res["structured_response"])
        policy_color = passed_cards_strings[decision.policy]

        enacted_policies = state["enacted_policies"].copy()
        enacted_policies[policy_color] += 1
        self.logger.debug(
            "Private enactment · player %s · justification: %s",
            policy_maker_id, decision.private_justification,
        )
        self.logger.info(
            "Policy enacted · player %s chose %s · totals %s blue / %s red",
            policy_maker_id, decision.policy, enacted_policies["blue"], enacted_policies["red"],
        )
        
        return {
            "enacted_policies": enacted_policies,
            "messages": [
                PublicMessage(
                    id=str(uuid4()),
                    player_id=policy_maker_id,
                    content=f"I enacted policy: {policy_color}.",
                    round_number=state["round_number"],
                )
            ]
        }

    async def discussion(self, state: GameState) -> dict[str, Any]:
        """
        Player discussion following policy enactment.

        1. [phase 1: defense]
            1.1. policy-maker says what they got
            1.2. president responds
            1.3. publish both to the group
        2. [phase 2: initial responses]
            2.1. elicit immediate reactions from all other players in parallel
        3. [phase 3: sequential]
            3.1. round-robin, optional responses from players
        4. [phase 4: closing remarks]
            4.1 president and policy maker post final remarks
        """

        async def _respond(
            player_id: int,
            state: GameState,
            phase: str,
            prompt_kwargs: dict | None,
            discussion_messages: list[PublicMessage],
        ) -> dict:
            """Helper function to elicit reaction from agent given message history, phase"""
            agent_runner = self.agents[player_id]
            config = {"configurable": {"thread_id": f"{state['game_id']}:{player_id}"}}
            
            new_messages = await catch_up_messages(agent_runner, config, state["messages"] + discussion_messages)
            discussion_prompt_kwargs = {
                **prompt_kwargs,
                "stage_instructions": self.discussion_stage_instructions[prompt_kwargs["stage"]],
                "round_number": state["round_number"],
                "round_manager": state["round_manager"],
                "policy_maker": state["proposed_policy_maker"],
                "enacted_policies": state["enacted_policies"],
            }
            res = await agent_runner.ainvoke(
                input={
                    "messages": new_messages,
                    "player_id": player_id,
                    "phase": phase,
                    "observation": discussion_prompt_kwargs,
                },
                config=config,
                context={
                    "phase": phase,
                    "prompt_kwargs": discussion_prompt_kwargs
                }
            )
            response = Response.model_validate(res["structured_response"])
            msg = PublicMessage(
                id=str(uuid4()),
                player_id=player_id,
                round_number=state["round_number"],
                content=response.content
            )
            self.logger.info(
                "Discussion · round %s · %s · player %s: %s",
                state["round_number"], prompt_kwargs["stage"], player_id, response.content,
            )
            return msg

        # phase 1
        start_time = time.time()
        discussion_messages = []
        policy_maker_id = state["proposed_policy_maker"]
        round_manager_id = state["round_manager"]
        
        drawn_cards = [card["policy"] for card in state["drawn_cards"]]
        passed_cards = [card["policy"] for card in state["passed_cards"]]
        private_context = {
            policy_maker_id: self.discussion_private_context_templates["policy_maker"].format(
                passed_cards=", ".join(passed_cards),
            ),
            round_manager_id: self.discussion_private_context_templates["round_manager"].format(
                drawn_cards=", ".join(drawn_cards), passed_cards=", ".join(passed_cards),
            ),
        }
        
        policy_maker_msg = await _respond(
            player_id=policy_maker_id,
            state=state,
            phase="discussion",
            prompt_kwargs={
                "stage": "opening",
                "private_context": private_context[policy_maker_id]
            },
            discussion_messages=discussion_messages,
        )
        discussion_messages.append(policy_maker_msg)
        
        round_manager_msg = await _respond(
            player_id=round_manager_id,
            state=state,
            phase="discussion",
            prompt_kwargs={
                "stage": "opening",
                "private_context": private_context[round_manager_id]
            },
            discussion_messages=discussion_messages,
        )
        discussion_messages.append(round_manager_msg)
        
        # phase 2
        all_other_players = [
            p for p in state["players"] if p not in [policy_maker_id, round_manager_id]
        ]

        # collect initial reactions from everyone
        initial_reactions: list[dict] = await asyncio.gather(
            *(
                _respond(
                    player_id,
                    state, phase="discussion",
                    prompt_kwargs={"stage": "reaction", "private_context": ""},
                    discussion_messages=discussion_messages,
                )
                for player_id in all_other_players
            )
        )
        discussion_messages.extend(initial_reactions)

        # phase 3: round-robin + closing remarks
        for _ in range(self.config.discussion_rounds_limit):
            if (time.time() - start_time) >= self.config.discussion_time_limit:
                break
            round_robin_order = all_other_players[:]
            random.shuffle(round_robin_order)
            round_robin_order += [policy_maker_id, round_manager_id] # closing remarks
            for player_id in round_robin_order:
                msg = await _respond(
                    player_id,
                    state=state,
                    phase="discussion",
                    prompt_kwargs={
                        "stage": "round-robin",
                        "private_context": private_context.get(player_id, "")
                    },
                    discussion_messages=discussion_messages
                )
                discussion_messages.append(msg)
        
        return {"messages": discussion_messages}


    def election_deadlock(self, state: GameState) -> dict[str, Any]:
        """Three failed elections, top card enacted automatically."""
        policies = state["enacted_policies"].copy()
        top_policy = self.deck.draw_one()["policy"]
        policies[top_policy] += 1
        self.logger.info(
            f"Election deadlock · {top_policy} gains a point · red total %s · blue total %s",
            policies["red"], policies["blue"]
        )
        return {
            "enacted_policies": policies,
            "consecutive_failed_elections": 0,
            "drawn_cards": [],
            "passed_cards": [],
            "messages": [
                PublicMessage(
                    id=str(uuid4()), player_id=-1, round_number=state["round_number"],
                    content=(
                        "Three consecutive elections failed. Top-card policy enacted: "
                        f"{top_policy} team got a point. The rejection counter resets."
                    ),
                )
            ],
        }

    def report_game_over(self, state: GameState) -> dict[str, Any]:
        """Record and log the winning team and the condition that ended the game."""
        for team, reason in POLICY_TERMINATION_REASONS.items():
            if state["enacted_policies"][team] >= POLICY_WIN_THRESHOLD:
                winner = team
                break
        else:
            if not (
                state["enacted_policies"]["red"] >= RED_LEADER_ELECTION_THRESHOLD and
                state["proposed_policy_maker"] == state["red_leader"] and
                state["elected"] is True
            ):
                raise ValueError("Cannot report game over without a victory condition.")
            winner = "red"
            reason = TerminationReason.RED_LEADER_ELECTED

        self.logger.info(
            "Game over · %s wins · %s · totals %s blue / %s red",
            winner, reason, state["enacted_policies"]["blue"], state["enacted_policies"]["red"],
        )
        return {"winner": winner, "termination_reason": reason}


    ## Router functions ##
    def eval_election_results(
        self, state: GameState
    ) -> Literal[TerminationReason.RED_LEADER_ELECTED, "candidate_rejected", "election_deadlock", "continue"]:
        """Route to a new nomination, policy selection, or red-leader victory."""

        if state["elected"] is None:
            raise ValueError("Cannot evaluate an unresolved election.")
        if state["elected"] == False:
            if state["consecutive_failed_elections"] >= 3:
                return "election_deadlock"
            self.logger.info(
                "Next · new nomination required · %s consecutive failed elections",
                state["consecutive_failed_elections"],
            )
            return "candidate_rejected"
        elif (
            state["proposed_policy_maker"] == state["red_leader"] and
            state["enacted_policies"]["red"] >= RED_LEADER_ELECTION_THRESHOLD
        ):
            return TerminationReason.RED_LEADER_ELECTED
        return "continue"

    def is_game_over(self, state: GameState) -> Literal["game_over", "continue"]:
        """Check whether the enacted policy satisfies a win condition."""
        if any(state["enacted_policies"][team] >= POLICY_WIN_THRESHOLD for team in POLICY_TERMINATION_REASONS):
            return "game_over"
        return "continue"


def is_sublist(a, b):
    """Check that every item in a occurs in b with sufficient multiplicity."""
    return not (Counter(a) - Counter(b))
