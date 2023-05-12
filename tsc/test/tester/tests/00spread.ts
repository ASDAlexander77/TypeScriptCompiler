function sum(x = 1, y = 2, z = 3) {
    print(`Values ${x}, ${y}, ${z}`);
    return x + y + z;
}

function main() {
    const numbers = [10, 20, 30];
    const r = sum(...numbers);
    print(r);
    assert(r === 60);

    const numbers2 = [10, 20, 30.0];
    const r2 = sum(...numbers2);
    print(r2);
    assert(r2 === 60);

    const filtered_nums = [1, 2, 3, 4, 5, 6].filter(x => x % 2 == 0);
    const r3 = sum(...filtered_nums);
    print(r3);
    assert(r3 === 12);

    const filtered_nums = [1, 2, 3, 4, 5].filter(x => x % 2 == 0);
    const r4 = sum(...filtered_nums);
    print(r4);
    assert(r4 === 9);

    print("done.");
}