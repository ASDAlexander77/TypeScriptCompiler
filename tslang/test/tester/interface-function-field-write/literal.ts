// An object literal filling an interface's function-typed field is not written through the
// interface: the write stops the program with a message.
interface Holder { f: () => int; }
function set(h: Holder) { h.f = () => 2; }
function main() {
    print("printed before");
    set({ f() { return 1; } });
    print("not reached");
}
