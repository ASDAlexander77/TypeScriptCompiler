function getProperty<T, K extends keyof T>(obj: T, key: K) {
  return obj[key]; // Inferred type is T[K]
}

function setProperty<T, K extends keyof T>(obj: T, key: K, value: T[K]) {
  obj[key] = value;
}

function main() {

let x = { foo: 10, bar: "hello!" };
let foo = getProperty(x, "foo"); // number
let bar = getProperty(x, "bar"); // string

    print("done.");
}