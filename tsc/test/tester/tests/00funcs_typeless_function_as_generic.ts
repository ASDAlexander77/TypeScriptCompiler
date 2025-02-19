function getMax(a, b) 
{
    return a > b ? a : b;
}

function init(f: (x:int, y:int) => int) {
    assert(f(10, 20) == 20);
}

function main() {

    init(getMax); init(getMax);

    assert(getMax(1, 2) == 2);
    assert(getMax(1, 2) == 2);
    assert(getMax(2.1, 3.2) == 3.2);
    assert(getMax("200", "400") == "400");

    print("done.");
}