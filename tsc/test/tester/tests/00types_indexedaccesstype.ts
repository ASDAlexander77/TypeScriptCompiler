type Person = { age: number; name: string; alive: boolean };
type Age = Person["age"];

function main() {

    let a: Age = 10.0;

    print("done.");
}
