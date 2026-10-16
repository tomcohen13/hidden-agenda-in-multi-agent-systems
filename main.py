"""Run the social deduction game."""

import argparse
import asyncio
import json
import logging
import sys
from time import perf_counter
from uuid import uuid4

from config import load_env
from social_deduction.game import Game, GameConfig, GameState


class ConsoleFormatter(logging.Formatter):
    """Compact terminal output, with colors only when stdout is a terminal."""

    COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def __init__(self, use_color: bool):
        super().__init__("%(message)s", datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:7}"
        if self.use_color:
            timestamp = f"\033[2m{timestamp}\033[0m"
            level = f"{self.COLORS.get(record.levelno, '')}{level}\033[0m"
        return f"{timestamp}  {level} │ {super().format(record)}"


def configure_logging() -> logging.Logger:
    """Show game progress on stdout and keep library chatter at warning level."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter(use_color=sys.stdout.isatty()))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logger = logging.getLogger("hidden_agenda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args(argv=None):
    """Parse the settings for a single election smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openrouter:openai/gpt-6-luna")
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--p-red", type=float, default=0.4)
    parser.add_argument("--blue-cards", type=int, default=6)
    parser.add_argument("--red-cards", type=int, default=11)
    args = parser.parse_args(argv)

    if args.num_players < 1 or not 0 < args.p_red < 0.5:
        parser.error("Require a positive player count and 0 < --p-red < 0.5.")
    if int(args.num_players * args.p_red) < 2:
        parser.error("The player count and red fraction must produce at least two red players.")
    if min(args.blue_cards, args.red_cards) < 0 or args.blue_cards + args.red_cards < 3:
        parser.error("Card counts must be nonnegative and total at least three.")
    return args


async def run_game(
    cfg: GameConfig,
    logger: logging.Logger,
    interrupt_after: list[str] | None = None,
    interrupt_before: list[str] | None = None,
) -> GameState:
    """Initialize one game and run with optional graph breakpoints."""
    started = perf_counter()
    game = Game(config=cfg, logger=logger)
    game.initialize()

    initial_state = GameState(
        game_id=str(uuid4()),
        players={},
        red_leader=None,
        round_number=0,
        round_manager=None,
        proposed_policy_maker=None,
        last_policy_maker=None,
        votes=[],
        elected=None,
        consecutive_failed_elections=0,
        drawn_cards=[],
        passed_cards=[],
        messages=[],
        enacted_policies={"red": 0, "blue": 0},
        winner=None,
        termination_reason=None,
        discussion_rounds_limit=1,
        discussion_time_limit=60,
    )

    logger.info("Game %s · %s", initial_state["game_id"], cfg.model)
    result = await game.graph.ainvoke(
        initial_state, interrupt_after=interrupt_after, interrupt_before=interrupt_before
    )
    logger.info("Run finished · %.1fs elapsed", perf_counter() - started)
    return result


def main(argv=None):
    args = parse_args(argv)
    logger = configure_logging()
    load_env()
    cfg = GameConfig(
        model=args.model,
        num_players=args.num_players,
        p_red=args.p_red,
        blue_cards=args.blue_cards,
        red_cards=args.red_cards,
    )
    result = asyncio.run(run_game(cfg, logger))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
