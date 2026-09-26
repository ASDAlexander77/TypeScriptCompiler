@dllname("strlen")
@linkname("wcslen")
declare function f(s: string): index;

function main() {
    print(f("a"));
}
