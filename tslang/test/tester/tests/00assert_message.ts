// An assert message known only at run time used to fail to compile ("operation's operand is
// unlinked"): the message was erased as if it were a constant.
function check(ok: boolean, what: string) {
    assert(ok, what + ": failed");
}

function main() {
    check(true, "check");

    const n = 3;
    assert(n == 3, `n is ${n}`);

    let none: string | null = null;
    assert(true, none);

    assert(n > 0, n);

    print("done.");
}
