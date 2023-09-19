function getProperty<T, K extends keyof T>(obj: T, key: K) {
    //return obj[key]; // Inferred type is T[K]
    return 1.0;
}

function setProperty<T, K extends keyof T>(obj: T, key: K, value: T[K]) {
    //obj[key] = value;
}

function main() {

    let x = { foo: 10, bar: "hello!" };
    let foo = getProperty(x, "foo"); // number
    let bar = getProperty(x, "bar"); // string

    // TODO: because in setProperty we send K as 'string' type (should be LiteralType - "foo"
    // we can't identify value T[K]

    //setProperty(x, "foo", 1); // number
    //setProperty(x, "bar", "2"); // string
    print("done.");
}