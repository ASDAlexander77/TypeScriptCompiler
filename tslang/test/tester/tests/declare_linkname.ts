// @linkname is @dllname: the TS name is cStrLen, the symbol it binds is the C runtime's strlen
@linkname("strlen")
declare function cStrLen(s: string): index;

function main() {
    assert(cStrLen("hello") == 5);
    print("done.");
}
