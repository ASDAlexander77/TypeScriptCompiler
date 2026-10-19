// yield* gives back nothing yet: its value cannot be used.
function* inner() {
    yield 1;
}

function* gen() {
    const r = yield* inner();
}

function main() {
    for (const v of gen()) print(v);
}
