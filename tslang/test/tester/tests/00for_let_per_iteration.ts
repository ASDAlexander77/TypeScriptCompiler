// `for (let i ...)`: each iteration has an `i` of its own, so a closure made in it keeps that iteration's value
// (ECMAScript CreatePerIterationEnvironment). Every closure saw the one `i` the loop had moved past (3 3 3, not
// 0 1 2). A body that assigns the variable keeps the one binding. Also at the top level, in a block.
function f() {
    const fs: (() => number)[] = [];
    for (let i: number = 0; i < 3; i++) { fs.push(() => i); }
    return fs[0]() * 100 + fs[1]() * 10 + fs[2]();
}

function skipping() {
    const fs: (() => number)[] = [];
    for (let i: number = 0; i < 4; i++) { if (i == 1) continue; fs.push(() => i * 2); }
    let s = "";
    for (const h of fs) s += h();
    return s;
}

function assigned() {
    let last = 0;
    for (let i = 0; i < 5; i++) { i++; last = i; }
    return last;
}

function two() {
    const fs: (() => boolean)[] = [];
    for (let i = 0, b = true; i < 2; i++, b = !b) { fs.push(() => b); }
    return fs[0]() && !fs[1]();
}

const top: (() => number)[] = [];
for (let i: number = 0; i < 3; i++) { top.push(() => i); }

let blockResult = 0;
{
    let x = 7;
    let a = () => x;
    blockResult = a() + (() => x)();
}

function main() {
    assert(f() == 12);
    assert(skipping() == "046");
    assert(assigned() == 5);
    assert(two());
    assert(top[0]() + top[1]() * 10 + top[2]() * 100 == 210);
    assert(blockResult == 14);
    print("done.");
}
