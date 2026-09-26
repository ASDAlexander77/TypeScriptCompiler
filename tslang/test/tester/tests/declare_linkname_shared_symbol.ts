// Several declarations may bind one C symbol. Each used to reach LLVM as a function of its own,
// and LLVM's setName made the taken name unique ("strlen.1"), so the call bound nothing.

declare function strlen(s: string): index;

@linkname("strlen")
declare function cStrLen(s: string): index;

@dllname("strlen")
declare function cStrLen2(s: string): index;

// both spellings naming the same symbol is not a conflict
@dllname("strlen")
@linkname("strlen")
declare function cStrLen3(s: string): index;

namespace C {
    @linkname("strlen")
    export declare function len(s: string): index;
}

// a declaration bound to a function defined in this module
@linkname("linkname_twice")
function twice(x: i32): i32 {
    return x * 2;
}

declare function linkname_twice(x: i32): i32;

// naming the symbol a function already has changes nothing
@linkname("same")
function same(): i32 {
    return 7;
}

function main() {
    assert(strlen("a") == 1);
    assert(cStrLen("ab") == 2);
    assert(cStrLen2("abc") == 3);
    assert(cStrLen3("abcd") == 4);
    assert(C.len("abcde") == 5);
    assert(twice(3) == 6);
    assert(linkname_twice(4) == 8);
    assert(same() == 7);
    print("done.");
}
