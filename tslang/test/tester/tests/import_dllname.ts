import './export_dllname'

function main() {
    assert(M.add(2, 3) == 5);
    assert(M.addTwice(2, 3) == 10);

    assert(M.counter == 40);
    assert(M.bump() == 42);

    assert(M.Calc.twice(4) == 8);

    const calc = new M.Calc(3);
    assert(calc.scale(5) == 15);

    assert(calc.double == 6);
    calc.double = 10;
    assert(calc.factor == 5);

    print("done.");
}
