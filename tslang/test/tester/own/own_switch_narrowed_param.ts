// A `switch` on the tag of a union parameter narrows it in each case into a local declared without a
// value: a copy of the caller's record that only reads its fields borrows the parameter, as an `if`
// narrowing it does - the caller's record keeps what it holds. Values are made at run time, in a
// loop, so a double release or a leak shows.
class Node {
    constructor(public v: number) {}
}

type Loading = { state: "loading" };
type Failed = { state: "failed"; code: number; reason: string };
type Success = { state: "success"; title: string; node: Node; info: { summary: string; size: number } };
type Net = Loading | Failed | Success;

function describe(s: Net): string {
    switch (s.state) {
        case "loading":
            return "...";
        case "failed":
            return `code ${s.code} ${s.reason}`;
        case "success":
            return s.title + ":" + s.node.v + ":" + s.info.summary;
    }
}

function weight(s: Net): number {
    switch (s.state) {
        case "success":
            return s.title.length + s.node.v + s.info.size;
        default:
            return 0;
    }
}

function make(i: number): Net {
    if (i % 3 == 0) {
        return { state: "loading" };
    }

    if (i % 3 == 1) {
        return { state: "failed", code: i, reason: "r" + i };
    }

    return { state: "success", title: "t" + i, node: new Node(i), info: { summary: "s" + i, size: i } };
}

function main() {
    let total = 0;
    for (let i = 0; i < 3000; i++) {
        const s = make(i);
        const text = describe(s);
        total += text.length + weight(s);
        assert(describe(s) == text, "the record is still the caller's");
    }

    assert(total > 0, "total");
    print("done.");
}
