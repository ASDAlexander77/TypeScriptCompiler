class Test {
    run() {
        const add = (x: TypeOf<1>, y: TypeOf<1>) => x + y;

        let sum = [1, 2, 3, 4, 5].reduce((s, v) => add(s, v), 0);
        print(sum);
        assert(sum == 15);
    }
}

function main() {
    const add = (x: TypeOf<1>, y: TypeOf<1>) => x + y;

    let sum = [1, 2, 3, 4, 5].reduce((s, v) => add(s, v), 0);
    print(sum);
    assert(sum == 15);

    const t = new Test();
    t.run();

    print("done.");
}
