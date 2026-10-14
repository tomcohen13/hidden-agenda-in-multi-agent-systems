"""Tests for the social deduction game's building blocks."""

from collections import Counter

import pytest

from social_deduction.game import Deck


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
