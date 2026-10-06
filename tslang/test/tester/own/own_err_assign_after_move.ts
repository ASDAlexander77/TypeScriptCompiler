// -mm=own rejects: `a` is read after its value moved into `b`. It is given a new value on one path
// only, and on the other it still holds the moved one. (Given a new value on every path, it owns
// that one: own_move_reassign.)
class C { x: number = 5; }
function main() {
    let a = new C();
    let b = a;
    if (b.x > 3) {
        a = new C();
    }

    print(a.x, b.x);
    print("done.");
}
