// The body of a `for await` runs one iteration at a time, in order (#498): each body was a task
// of its own, all awaited together after the loop, so iterations interleaved and lost each
// other's writes.
let log = "";
let keep: number[] = [];
async function* numbers() {
    for (let i = 1; i <= 8; i++) {
        yield i;
    }
}
async function note(s: string) {
    log += s;
}
async function main() {
    for await (const x of numbers()) {
        log += "<" + x;
        keep.push(x);
        log += ">";
    }
    assert(log == "<1><2><3><4><5><6><7><8>", "each body whole, in order");
    assert(keep.length == 8, "every push kept");

    log = "";
    for await (const x of numbers()) {
        await note("[" + x);
        await note("]");
    }
    assert(log == "[1][2][3][4][5][6][7][8]", "awaits in the body, in order");
    print("done.");
}
