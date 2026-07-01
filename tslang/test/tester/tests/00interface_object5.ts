interface Point {
    x: number;
    y: number;

    fromOrigin(p: Point): number;
}

const o = {
    x:1, 
    y:2, 
    fromOrigin(p: Point) { return 3.0; }
};

const iface: Point = o;

print(iface.x);
print(iface.y);
print(iface.fromOrigin(iface));

assert(iface.x == 1);
assert(iface.y == 2);
assert(iface.fromOrigin(iface) == 3.0);

print("done.")