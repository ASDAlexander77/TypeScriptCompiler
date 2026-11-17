// A class field holding a class holds the class's value, `typeof C` (`static self = X;`, `cls = B;`): it failed
// with "failed to legalize operation 'ts.ClassRef'". A tuple type spreading a tuple has its elements in place
// (`[...A, ...A, number]`), which crashed in getRestType; an empty array of `void` (`[[]] as [void[]]`) crashed
// making its LLVM array of void.
class B { static n = 7; static make() { return "b"; } }
class X { static self = X; static k = 3; cls = B; }

type A = [s: string];
type C = [...A, ...A, number];

function voids() { const t = [[]] as [void[]]; return t[0].length; }

function main() {
    assert(X.self.k == 3);
    const x = new X();
    assert(x.cls.n == 7 && x.cls.make() == "b");

    const t: C = ["a", "b", 3];
    assert(t[0] == "a" && t[1] == "b" && t[2] == 3 && t.length == 3);
    assert(voids() == 0);
    print("done.");
}
