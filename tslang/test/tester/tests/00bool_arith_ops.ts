// regression test: boolean operands in arithmetic/comparison binary ops must coerce to
// number the same way JS/TS does (true -> 1, false -> 0), not operate on the raw i1
// representation.
//
// bugs found and fixed:
// - `+`/`-` had no case forcing numeric coercion when both operands were already the
//   same type (boolean), unlike `/`/`%`/`**` which always force getNumberType(). So
//   `true + true` computed 1+1 as a wrapping i1 add (result: false) instead of 2, and
//   `true - false` similarly wrapped instead of giving 1.
// - ordering comparisons (`> >= < <=`) picked their icmp predicate via
//   Type::isUnsignedInteger(), which is false for the custom BooleanType (a signless
//   i1), so they fell to the *signed* predicate -- where i1 `true` (bit pattern 1)
//   reads as -1, making `true > false` compare as `-1 > 0` (false) instead of true.

function main() {
    assert(true + true == 2);
    assert(true + false == 1);
    assert(false + false == 0);

    assert(true - false == 1);
    assert(false - true == -1);
    assert(true - true == 0);

    assert(true * 2 == 2);
    assert(false * 2 == 0);

    assert(true > false);
    assert(!(false > true));
    assert(false < true);
    assert(!(true < false));
    assert(true >= true);
    assert(true <= true);
    assert(true >= false);
    assert(!(false >= true));

    assert(true == true);
    assert(true != false);
    assert(true === true);
    assert(true !== false);

    print("done.");
}
