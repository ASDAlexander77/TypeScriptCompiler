let linkname_counter = 1;

@linkname("linkname_counter")
declare function f(): i32;

function main() {
    print(linkname_counter, f());
}
