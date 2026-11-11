// A @final interface is not extended: a value of the extending one would be a value of it.
@final
interface Shape {
    area(): number;
}

interface Solid extends Shape {
    volume(): number;
}

function main() {
    print("done.");
}
