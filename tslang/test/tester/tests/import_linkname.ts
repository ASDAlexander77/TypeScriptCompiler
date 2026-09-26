import './export_linkname'

function main() {
    assert(L.add(2, 3) == 5);
    assert(L.addTwice(2, 3) == 10);

    assert(L.counter == 40);
    assert(L.bump() == 42);

    assert(L.Calc.twice(4) == 8);

    const calc = new L.Calc(3);
    assert(calc.scale(5) == 15);

    assert(calc.double == 6);
    calc.double = 10;
    assert(calc.factor == 5);

    print("done.");
}
