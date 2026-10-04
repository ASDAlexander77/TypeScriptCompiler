// A destructuring rest element is a new array, not a view into the source (#477).
class Box {
    constructor(public v: number) {}
}

function main() {
    const src: number[] = [1, 2, 3];
    const [first, ...rest] = src;
    assert(first == 1, "first");
    rest[0] = 20;
    assert(src[1] == 2, "a write to the rest leaks into the source");
    rest.push(4);
    assert(rest.length == 3 && rest[2] == 4, "push onto the rest");
    assert(src.length == 3 && src[2] == 3, "the source is unchanged");

    const [a, b, ...none] = [1];
    assert(none.length == 0, "the rest of a shorter array is empty");

    const boxes: Box[] = [new Box(1), new Box(2), new Box(3)];
    const [head, ...tail] = boxes;
    tail.push(new Box(4));
    assert(head.v == 1 && tail.length == 3 && tail[0].v == 2 && tail[2].v == 4, "class elements");

    const words: string[] = ["a", "b", "c"];
    const [w0, ...ws] = words;
    ws.push("d");
    assert(w0 == "a" && ws.length == 3 && ws[2] == "d" && words.length == 3, "string elements");

    const [p0, ...p1] = src;
    const [q0, ...q1] = p1;
    q1.push(9);
    assert(p0 == 1 && q0 == 2 && q1.length == 2 && q1[1] == 9 && p1.length == 2 && p1[1] == 3, "nested and repeated rest");

    // a rest from index 0 copies the whole source, and a push onto an empty rest grows it
    const letters: string[] = ["a", "b"];
    const [...all] = letters;
    all.push("c");
    assert(all.length == 3 && letters.length == 2 && all[0] == "a", "a rest from index 0");
    const [l0, l1, l2, ...noLetters] = letters;
    noLetters.push("q");
    noLetters.push("r");
    assert(noLetters.length == 2 && noLetters[1] == "r", "a push onto an empty rest");

    print("done.");
}
