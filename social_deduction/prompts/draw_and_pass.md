## Current phase: select policies to pass

You are the round manager. The nomination has been approved, and you have
privately drawn these three policy cards:

{drawn_cards}

Choose exactly two cards to pass to the elected policy maker. The remaining card
is discarded. The policy maker will choose one of the two passed cards to enact.

Return your selection using the required response schema:
- private_justification: a concise private explanation of your selection.
- passed_cards: a list of exactly two colors, each either "red" or "blue".
  You cannot pass more cards of either color than appear in your drawn hand.

Return colors rather than indices. Both selected cards may have the
same color if your hand contains at least two cards of that color.

Your drawn hand and this explanation are completely private.
The policy maker will receive only the two passed cards.
