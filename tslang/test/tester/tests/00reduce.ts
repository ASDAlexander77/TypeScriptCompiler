function main() {
    let sum = [1, 2, 3].reduce((s, v) => s + v, 0)

    print(sum);

    assert(sum == 6, "red")

    print("done.");
}
