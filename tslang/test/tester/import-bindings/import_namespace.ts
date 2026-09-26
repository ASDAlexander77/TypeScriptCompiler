// `import * as M`: the module's declarations through M, in expressions and in types. M used to be
// "can't resolve name".
import * as M from './names_module'

assert(M.add(2, 3) == 5);
assert(M.ANSWER == 42);

const p = new M.Point(1, 2);
assert(p.sum() == 3);

assert(M.Color.Blue == 3);

const n: M.Named = { name: "n" };
assert(n.name == "n");

const t: M.Pair = [4, 5];
assert(t[1] == 5);

const b = new M.Box<number>(6);
assert(b.v == 6);
assert(M.identity<number>(7) == 7);

print("done.");
