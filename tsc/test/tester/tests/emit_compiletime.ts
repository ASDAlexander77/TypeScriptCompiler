type Point = { x: number; y: number; };

@dllimport
declare function point (x: number, y: number): Point;

function main() {
    const p = point (1.0, 2.0);
    print(`x=${p.x}, y=${p.y}`);

    print("done.");
}