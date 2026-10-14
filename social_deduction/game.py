"""Base models for experiment"""
import json
import random
from dataclasses import dataclass
from operator import add
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from social_deduction.agent import Nomination, PublicMessage, catch_up_messages, create_game_agent
from config import SOCIAL_DEDUCTION_PATH

PROMPTS_DIR = SOCIAL_DEDUCTION_PATH / "prompts"

class PolicyCard(TypedDict):
    card_id: int
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

    def draw_three(self) -> list[PolicyCard, PolicyCard]:
        """Pop and return the top card in the deck."""
        drawn = []
        while len(drawn) < 3:
            if self.is_empty():
                self._initialize_deck()
            drawn.append(self.cards.pop())
        return drawn


class GameState(TypedDict):

    game_id: str
    players: list
    assignments: dict
    red_leader: int
    round_number: int
    round_manager: int
    proposed_policy_maker: int
    votes: list[str]
    last_policy_maker: int  # can't be policy-maker twice in a row
    consecutive_failed_elections: int
    drawn_cards: list[PolicyCard]
    passed_cards: list[PolicyCard]
    enacted_policies: dict
    messages: Annotated[list[PublicMessage], add]


@dataclass
class GameConfig:
    blue_cards: int = 6
    red_cards: int = 11
    num_players: int = 10
    p_red: float = 0.3
    model: str = "openai:gpt-5-mini"


class Game:

    def __init__(
        self,
        config: GameConfig,
        logger = None,
    ):
        self.config = config
        self.agents: dict[int, CompiledStateGraph] = {}
        self.graph = None
        self.deck = Deck(config.blue_cards, config.red_cards)
        self.prompts: dict = self._load_prompts()
        self.role_instructions = json.loads(
            (PROMPTS_DIR / "role_instructions.json").read_text(encoding="utf-8")
        )

    def _load_prompts(self) -> None:
        prompts = {}
        paths = PROMPTS_DIR.glob(".md")
        for path in paths:
            prompts[path.stem] = path.read_text()
        return prompts
        

    def initialize(self):
        """Compile the game blueprint for inspection and visualization."""
        builder = StateGraph(GameState)
        builder.add_node("initialize_players", self.initialize_players)
        builder.add_node("start_round", self.start_round)  # rotate round manager, increment round number
        builder.add_node("propose_policy_maker", self.propose_policy_maker) # round manager to propose enacter
        builder.add_node("vote", self.vote)
        builder.add_node("draw_and_pass", self.draw_and_pass) # round manager draws three policies, passes 2 to enacter 
        builder.add_node("enact", self.enact) 
        builder.add_node("discussion", self.discussion)

        builder.add_edge(START, "initialize_players")
        builder.add_edge("initialize_players", "start_round")
        builder.add_edge("start_round", "propose_policy_maker")
        builder.add_edge("propose_policy_maker", "vote")
        builder.add_conditional_edges(
            "vote",
            self.eval_election_results,
            {
                "red_leader_elected": END,
                "candidate_rejected": "propose_policy_maker",
                "continue": "draw_and_pass",
            },
        )
        builder.add_edge("draw_and_pass", "enact")
        builder.add_conditional_edges(
            "enact",
            self.is_game_over,
            {"game_over": END, "continue": "discussion"},
        )
        builder.add_edge("discussion", "start_round")
        self.graph = builder.compile()


    def initialize_players(self, state: GameState) -> dict[str, Any]:
        """Assign teams and the red leader, then create role-specific agents."""

        cfg = self.config
        num_red_players = int(cfg.num_players * cfg.p_red)

        assert num_red_players >= 2, "Must have at least 2 red players in the game."
        red_leader = random.randint(0, num_red_players - 1)

        # load prompts
        prompt = self.prompts["player"]

        # initialize agents
        player_assignments = ["red"] * num_red_players + ["blue"] * (cfg.num_players - num_red_players)
        random.shuffle(player_assignments)
        red_leader = player_assignments.index("red") # first occurrence

        for i, assignment in enumerate(player_assignments):
            role = f"{assignment}_{'leader' if i == red_leader else 'regular'}"
            role_instructions = self.role_instructions[role]
            if role == "red_regular":
                other_red_players = [
                    k for k, v in enumerate(player_assignments)
                    if v == "red" and k != i
                ]
                role_instructions = role_instructions.format(peers=other_red_players, leader=red_leader)
            agent_prompt = prompt.format(role_instructions=role_instructions)
            agent = create_game_agent(
                name=i, prompt=agent_prompt, model=cfg.model, prompts=self.prompts
            )
            self.agents[i] = agent
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
        return {
            "round_number": state["round_number"] + 1,
            "round_manager": next_round_manager
        }

    async def propose_policy_maker(self, state: GameState) -> dict[str, Any]:
        """Ask the round manager to nominate an eligible policy enacter."""

        round_manager_id = state["round_manager"]
        all_players = set(state["players"].keys()) 
        valid_players = all_players - {round_manager_id, state["last_policy_maker"]}
        if not valid_players:
            raise ValueError("No eligible policy maker is available.")
        round_manager_runner = self.agents[round_manager_id]
        config = {
            "configurable": {"thread_id": f"{state['game_id']}:{round_manager_id}"}
        }
        new_messages = await catch_up_messages(round_manager_runner, config, state["messages"])
        res = await round_manager_runner.ainvoke(
            {"messages": new_messages},
            config=config,
            context={"phase": "propose_policy_maker", "prompt_args": (sorted(valid_players),)},
        )
        nomination = Nomination.model_validate(res["structured_response"])
        if nomination.policy_maker_id not in valid_players:
            raise ValueError(f"Ineligible policy maker: {nomination.policy_maker_id}")
        return {
            "proposed_policy_maker": nomination.policy_maker_id,
            "messages": [{
                "id": str(uuid4()),
                "player_id": round_manager_id,
                "round_number": state["round_number"],
                "content": (f"I nominate player {nomination.policy_maker_id} as policy maker.\n"
                            f"{nomination.justification}"),
            }],
        }


    async def vote(self, state: GameState) -> dict[str, Any]:
        """Collect votes from a shared snapshot and resolve the nomination."""
        raise NotImplementedError

    async def draw_and_pass(self, state: GameState) -> dict[str, Any]:
        """Let the round manager draw three policies and privately pass two."""
        raise NotImplementedError

    async def enact(self, state: GameState) -> dict[str, Any]:
        """Let the enacter choose a passed policy and publish its outcome."""
        raise NotImplementedError

    async def discussion(self, state: GameState) -> dict[str, Any]:
        """Run an ordered discussion with role-appropriate message visibility."""
        raise NotImplementedError

    def eval_election_results(
        self, state: GameState
    ) -> Literal["red_leader_elected", "candidate_rejected", "continue"]:
        """Route to a new nomination, policy selection, or red-leader victory."""
        raise NotImplementedError

    def is_game_over(self, state: GameState) -> Literal["game_over", "continue"]:
        """Check whether the enacted policy satisfies a win condition."""
        raise NotImplementedError
