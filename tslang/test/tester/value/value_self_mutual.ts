// ...or one that holds itself through another @value type
@value
interface A {
    readonly b: B | null;
}

@value
interface B {
    readonly a: A;
}

function main() {
    print(1);
}
