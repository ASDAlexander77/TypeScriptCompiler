// A property that does not resolve is named in the error, wherever it is read (#442): during the
// run that infers main's return type the access gave no value and said nothing, so an `if` said
// only "the condition has no value".
function main() {
    const o = { x: 1 };
    while (o.foo) print(1);
}
