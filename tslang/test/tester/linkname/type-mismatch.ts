// string and Opaque both lower to ptr, but the calls are still TS-typed when the rename runs
@linkname("strlen")
declare function a(s: string): index;

@linkname("strlen")
declare function b(s: Opaque): index;

function main() {
    let o: Opaque;
    print(a("x"), b(o));
}
