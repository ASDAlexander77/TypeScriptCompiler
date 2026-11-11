// An object literal is no implementer of a @final interface: it would bring a vtable of its own.
@final
interface Shape {
    area(): number;
}

function main() {
    const shape: Shape = { area() { return 1; } };
    print(shape.area());
    print("done.");
}
