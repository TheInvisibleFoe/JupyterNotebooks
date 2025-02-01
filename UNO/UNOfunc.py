
def deal_cards(AC,num,nop):
    Pd = []
    deck = list(AC)
    random.shuffle(deck)
    for i in range(0, nop):
        Pd.append(deck[i:i+num*nop:nop])
    deck = deck[num*nop:]
    return Pd, deck


def discard(discard_pile, card):
    discard_pile = list(discard_pile)
    discard_pile.append(card)
    return discard_pile

def first_card(deck):
    topcard = deck[0]
    if topcard[:-1] in ss or topcard in SC:
        deck.append(topcard)
        deck.pop(0)
        first_card(deck)
    else :
        return topcard,deck
