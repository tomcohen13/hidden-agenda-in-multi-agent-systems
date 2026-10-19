## Current phase: select policies to pass

You are the round manager. Your current private hand is:

{drawn_cards}

Select exactly two card indices from this hand to pass to the policy maker, who will
enact one of them. Each list entry is one card and can be selected only once.

Return the required response schema:
- passed_cards: two distinct integer indices from your current hand, each either 0, 1, or 2.
- private_justification: briefly explain your selection.

valid passed_cards values: [0, 1], [1, 2], [0, 2]
invalid passed_cards values: [0, 0], [1, 1], [2, 2]

Your hand and justification are private. The policy maker receives only the selected cards.
