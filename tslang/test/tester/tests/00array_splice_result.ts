// splice gives back the elements it removes, as JavaScript does: it gave back an index (the new
// length), so `a.splice(1, 2).length` did not compile
class Box {
    constructor(public name: string) {}
}

function main() {
    const a = [1, 2, 3, 4, 5];
    const removed = a.splice(1, 2);
    assert(removed.length == 2 && removed[0] == 2 && removed[1] == 3, "the removed numbers");
    assert(a.length == 3 && a[1] == 4, "what is left");

    const none = a.splice(1, 0, 9, 8);
    assert(none.length == 0 && a.length == 5 && a[1] == 9 && a[2] == 8, "an insert removes nothing");

    const tail = a.splice(-2);
    assert(tail.length == 2 && tail[0] == 4 && tail[1] == 5 && a.length == 3, "from the end, to the end");

    const past = a.splice(1, 100);
    assert(past.length == 2 && a.length == 1, "a delete count past the end");

    a.splice(0, 1);
    assert(a.length == 0, "the result unused");

    // the removed objects are the result's now: the source no longer holds them
    const boxes = [new Box("p"), new Box("q"), new Box("r")];
    const taken = boxes.splice(0, 2, new Box("s"));
    assert(taken.length == 2 && taken[0].name == "p" && taken[1].name == "q", "the removed objects");
    assert(boxes.length == 2 && boxes[0].name == "s" && boxes[1].name == "r", "the objects left");

    const strings = ["x", "y", "z"];
    const extra = ["m", "n"];
    const gone = strings.splice(1, 1, ...extra);
    assert(gone.length == 1 && gone[0] == "y", "with spread items");
    assert(strings.length == 4 && strings[1] == "m" && strings[2] == "n" && strings[3] == "z", "spread items in place");

    for (let i = 0; i < 1000; i++) {
        const round = [new Box("a"), new Box("b"), new Box("c")];
        const out = round.splice(1, 1);
        assert(out[0].name == "b" && round.length == 2, "in a loop");
        round.splice(0, 1);
    }

    print("done.");
}
