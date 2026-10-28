// A class filling an interface's function-typed field with a method has no field to store to: a
// write through the interface stops the program with a message.
interface Holder { f: () => int; }
class WithMethod { f(): int { return 1; } }
function set(h: Holder) { h.f = () => 2; }
function main() {
    print("printed before");
    set(new WithMethod());
    print("not reached");
}
