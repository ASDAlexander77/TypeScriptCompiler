let a = {
    a: 1
};

let deck = {
    suits: ["hearts", "spades", "clubs", "diamonds"],
    cards: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    createCardPicker: function () {
        // NOTE: the line below is now an arrow function, allowing us to capture 'this' right here
        return () => {
            return { suit: this.suits[1], card: 2 };
        };
    },
};

function main() {
    assert(a.a == 1);

    let cardPicker = deck.createCardPicker();
    let pickedCard = cardPicker();

    assert(pickedCard.card == 2);
    assert(pickedCard.suit == "spades");

    print("done.");
}
