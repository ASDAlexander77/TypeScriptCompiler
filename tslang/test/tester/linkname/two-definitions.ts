@linkname("linkname_dup")
function a(): i32 {
    return 1;
}

@linkname("linkname_dup")
function b(): i32 {
    return 2;
}

function main() {
    print(a(), b());
}
