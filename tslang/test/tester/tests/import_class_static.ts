import './export_class_static'

class NamedCounter extends M.Counter {
    static incrementTwice(n: number): void {
        super.increment(n);
        super.increment(n);
    }
}

function main() {
    M.Counter.increment(5);
    assert(M.Counter.count == 5);
    assert(NamedCounter.count == 5);

    NamedCounter.incrementTwice(10);
    assert(M.Counter.count == 25);
    assert(NamedCounter.count == 25);

    assert(M.Counter.describe() == "counter:25");
    assert(NamedCounter.describe() == "counter:25");

    // mutate the inherited static field through the derived class name and
    // confirm the base sees it too - proves shared storage, not a copy
    NamedCounter.label = "derived";
    assert(M.Counter.label == "derived");
    assert(M.Counter.describe() == "derived:25");

    print("done.");
}
