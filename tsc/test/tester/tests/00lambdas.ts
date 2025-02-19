class Test {
    run() {
        const add = (x, y) => x + y;

        let sum = [1, 2, 3, 4, 5].reduce((s, v) => add(s, v), 0);
        print(sum);
        assert(sum == 15);
    }
}

static class Array<T>
{
    public reduce2<V>(this: T[], func: (v: V, t: T) => V, initial: V) {
        let result = initial;
        print("initial = ", initial, "result=", result);
        for (const v of this) result = func(result, v);
        return result;
    }    
}

function main() {
    const add = (x: TypeOf<1>, y: TypeOf<1>) => x + y;
    const add2 = (x, y) => x + y;

    let sum = [1, 2, 3, 4, 5].reduce((s, v) => add(s, v), 0);
    print(sum);
    assert(sum == 15);

    const t = new Test();
    t.run();

    let sum2 = [1, 2, 3, 4, 5].reduce2((s, v) => add(s, v), 0);
    print(sum2);
    assert(sum2 == 15);

    let sum3 = [1, 2, 3, 4, 5].reduce2((s, v) => add2(s, v), 0);
    print(sum3);
    assert(sum3 == 15);

    print("done.");
}
