// A member of another declaration of an enum is a name in a later one's initializers (`enum E { a } enum E { c = a }`),
// as a member of the same declaration is: "can't resolve name: a".
enum E { a = 2, b = a }
enum E { c = a + 1, d = c * 2 }

function main() {
    assert(E.a == 2 && E.b == 2 && E.c == 3 && E.d == 6);
    print("done.");
}
