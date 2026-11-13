// A @cpp class extending one another module exported (export_virtual_slots.ts): its @virtual methods
// start after the imported class's slots, whose number __decls carries, and its @destructor
// overrides the imported one.
import './export_virtual_slots'

@cpp
class Circle extends Shape {
    @virtual radius(): s32;
    @virtual area(): s32;
    @destructor destroy(): void;
}

function radiusOf(circle: Circle) {
    return circle.radius() + circle.area();
}

function main() {
    print("done.");
}
