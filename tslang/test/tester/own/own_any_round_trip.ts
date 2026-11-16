// A value boxed into `any` only to be cast straight back and read (spec 92, user ruling 2026-11-16):
// the box borrows it - no copy, no error - and its release frees the box only. A parameter, a record
// read through a union, a local array changed through what is cast back, and a fresh value this
// function owns. Values are made at run time, in a loop, so a double release or a leak shows.
class Node {
    constructor(public v: number, public name: string) {}
}

type Failed = { state: string; code: number; reason: string };
type Ok = { state: string; title: string };
type Net = Failed | Ok;

function reason(s: Net) {
    return s.state == "failed" ? (<Failed><any>s).reason : (<Ok><any>s).title;
}

function nodeName(n: Node) {
    return (<Node><any>n).name.length;
}

function grow(i: number) {
    let a = [i, i + 1];
    const box = a as any;
    const unboxed = box as number[];
    unboxed.push(i + 2);
    return a.length;
}

function makeOk(i: number): Net {
    return { state: "ok", title: "t" + i };
}

function main() {
    let total = 0;
    for (let i: number = 0; i < 3000; i++) {
        const f: Net = { state: "failed", code: i, reason: "r" + i };
        total += reason(f).length;
        assert(reason(f) == "r" + i, "the record is still the caller's");

        const n = new Node(i, "n" + i);
        total += nodeName(n);
        assert(n.name == "n" + i, "the node is still the caller's");

        assert(grow(i) == 3, "pushed through what is cast back");

        const o = makeOk(i);
        assert((<Ok><any>o).title == "t" + i, "a fresh value, boxed and read");
        assert((<Ok><any>o).title.length > 1, "boxed again");
    }

    assert(total > 0, "total");
    print("done.");
}
