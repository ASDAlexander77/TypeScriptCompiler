function sum(x = 0, y = 0, z = 0) {
    print(`Values ${x}, ${y}, ${z}`);
    return x + y + z;
}

function main() {
    const numbers = [10, 20, 30];
    const r = sum(...numbers);
    print(r);
    assert(r === 60);
    print("done.");
}