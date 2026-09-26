// `import { a as b }`: every kind of declaration under another name. Each used to be
// "can't resolve name".
import { add as plus, ANSWER as K, Point as P, Color as C, Named as N, Pair as TwoNumbers, Box as B, identity as id } from './names_module'

assert(plus(2, 3) == 5);
assert(K == 42);

const p = new P(1, 2);
assert(p.sum() == 3);

assert(C.Blue == 3);

const n: N = { name: "n" };
assert(n.name == "n");

const t: TwoNumbers = [4, 5];
assert(t[1] == 5);

const b = new B<number>(6);
assert(b.v == 6);
assert(id<number>(7) == 7);

print("done.");
