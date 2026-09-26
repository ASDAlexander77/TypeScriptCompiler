// A function renamed to a name another function is renamed away from: which one the call binds
// would depend on the order the renames run, so it is an error - in either textual order. This
// is the order that was accepted: linkname_chain_b is renamed first, and `a` then took its name.
@linkname("linkname_chain_c")
function linkname_chain_b(): i32 {
    return 2;
}

@linkname("linkname_chain_b")
function a(): i32 {
    return 1;
}

function main() {
    print(a(), linkname_chain_b());
}
