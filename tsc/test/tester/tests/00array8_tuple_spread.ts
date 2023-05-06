function main() {

    let count = 0;

    [...[1, 2, "Hello"]].forEach(
        x => {
            if (typeof x == "string") { print(x); count++; }
            if (typeof x == "i32") { print(x); count++; }
        }
    );

    assert(count == 3);

    print("done.");
}