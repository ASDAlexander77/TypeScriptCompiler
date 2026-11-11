// @strict-null false
// The ray tracer of rtbench.ts written the way rust/src/bin/static_dispatch.rs is: no interfaces
// and no closures, so nothing is called through a vtable or a function value. A thing is one
// @final class with a `shape` tag and a `switch` (Rust's `enum Thing` and `match`), a surface is
// an enum, Vector, Color, Ray, Light and Hit are @value classes, the ray tracer's methods are free
// functions and the lights are added up in a `for` loop. A miss is the distance Infinity where the
// Rust port has `None`. The same file builds under every memory model, and the scene, the
// arithmetic and the checksum (38837369) are rtbench.ts's.
// print() shows the sum with 6 significant digits, so the low digits are printed separately.
declare function sqrt(v: number): number;
declare function floor(v: number): number;
declare function pow(v: number, s: number): number;

@value
class Vector {
    constructor(public readonly x: number,
        public readonly y: number,
        public readonly z: number) {
    }
    static times(k: number, v: Vector) { return new Vector(k * v.x, k * v.y, k * v.z); }
    static minus(v1: Vector, v2: Vector) { return new Vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
    static plus(v1: Vector, v2: Vector) { return new Vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
    static dot(v1: Vector, v2: Vector) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
    static mag(v: Vector) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    static norm(v: Vector) {
        const mag = Vector.mag(v);
        const div = (mag === 0) ? Infinity : 1.0 / mag;
        return Vector.times(div, v);
    }
    static cross(v1: Vector, v2: Vector) {
        return new Vector(v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x);
    }
}

@value
class Color {
    constructor(public readonly r: number,
        public readonly g: number,
        public readonly b: number) {
    }
    static scale(k: number, v: Color) { return new Color(k * v.r, k * v.g, k * v.b); }
    static plus(v1: Color, v2: Color) { return new Color(v1.r + v2.r, v1.g + v2.g, v1.b + v2.b); }
    static times(v1: Color, v2: Color) { return new Color(v1.r * v2.r, v1.g * v2.g, v1.b * v2.b); }
}

const white = new Color(1.0, 1.0, 1.0);
const grey = new Color(0.5, 0.5, 0.5);
const black = new Color(0.0, 0.0, 0.0);
const background = black;
const defaultColor = black;

@value
class Ray {
    constructor(public readonly start: Vector, public readonly dir: Vector) {
    }
}

@value
class Light {
    constructor(public readonly pos: Vector, public readonly color: Color) {
    }
}

@final
class Camera {
    public forward: Vector;
    public right: Vector;
    public up: Vector;

    constructor(public pos: Vector, lookAt: Vector) {
        const down = new Vector(0.0, -1.0, 0.0);
        this.forward = Vector.norm(Vector.minus(lookAt, this.pos));
        this.right = Vector.times(1.5, Vector.norm(Vector.cross(this.forward, down)));
        this.up = Vector.times(1.5, Vector.norm(Vector.cross(this.forward, this.right)));
    }
}

enum Surface {
    Shiny,
    Checkerboard
}

function isWhiteSquare(pos: Vector) {
    return (floor(pos.z) + floor(pos.x)) % 2 !== 0;
}

function diffuse(surface: Surface, pos: Vector): Color {
    switch (surface) {
        case Surface.Shiny: return white;
        default: return isWhiteSquare(pos) ? white : black;
    }
}

function specular(surface: Surface): Color {
    switch (surface) {
        case Surface.Shiny: return grey;
        default: return white;
    }
}

function reflect(surface: Surface, pos: Vector): number {
    switch (surface) {
        case Surface.Shiny: return 0.7;
        default: return isWhiteSquare(pos) ? 0.1 : 0.7;
    }
}

function roughness(surface: Surface): number {
    switch (surface) {
        case Surface.Shiny: return 250.0;
        default: return 150.0;
    }
}

enum Shape {
    Sphere,
    Plane
}

// Rust's `enum Thing { Sphere { center, radius2, .. }, Plane { norm, offset, .. } }`: `v` is the
// sphere's center or the plane's normal, `k` the sphere's squared radius or the plane's offset.
@final
class Thing {
    constructor(public readonly shape: Shape, public readonly v: Vector, public readonly k: number,
        public readonly surface: Surface) {
    }

    static sphere(center: Vector, radius: number, surface: Surface) {
        return new Thing(Shape.Sphere, center, radius * radius, surface);
    }

    static plane(norm: Vector, offset: number, surface: Surface) {
        return new Thing(Shape.Plane, norm, offset, surface);
    }

    normal(pos: Vector): Vector {
        switch (this.shape) {
            case Shape.Sphere: return Vector.norm(Vector.minus(pos, this.v));
            default: return this.v;
        }
    }

    // The distance along the ray to the thing, or Infinity when the ray misses it.
    intersect(ray: Ray): number {
        switch (this.shape) {
            case Shape.Sphere: {
                const eo = Vector.minus(this.v, ray.start);
                const v = Vector.dot(eo, ray.dir);
                let dist = 0.0;
                if (v >= 0) {
                    const disc = this.k - (Vector.dot(eo, eo) - v * v);
                    if (disc >= 0) {
                        dist = v - sqrt(disc);
                    }
                }
                return dist === 0 ? Infinity : dist;
            }
            default: {
                const denom = Vector.dot(this.v, ray.dir);
                if (denom > 0) {
                    return Infinity;
                }
                return (Vector.dot(this.v, ray.start) + this.k) / (-denom);
            }
        }
    }
}

@final
class Scene {
    constructor(public things: Thing[], public lights: Light[], public camera: Camera) {
    }
}

// What a ray hit first: the thing, as its index in the scene's things, and the distance to it. A
// ray that hit nothing has dist === Infinity. (The Rust port holds a `&Thing`; under -mm=own a
// @value class cannot hold a borrowed element of an array, and an index needs no lifetime.)
@value
class Hit {
    constructor(public readonly index: number, public readonly dist: number) {
    }
}

const maxDepth = 5;

function intersections(ray: Ray, scene: Scene): Hit {
    const things = scene.things;
    let closest = Infinity;
    let index = 0;
    for (let i = 0; i < things.length; i++) {
        const dist = things[i].intersect(ray);
        if (dist < closest) {
            index = i;
            closest = dist;
        }
    }

    return new Hit(index, closest);
}

function traceRay(ray: Ray, scene: Scene, depth: number): Color {
    const hit = intersections(ray, scene);
    if (hit.dist === Infinity) {
        return background;
    }

    return shade(hit, ray, scene, depth);
}

function shade(hit: Hit, ray: Ray, scene: Scene, depth: number): Color {
    const thing = scene.things[hit.index];
    const d = ray.dir;
    const pos = Vector.plus(Vector.times(hit.dist, d), ray.start);
    const normal = thing.normal(pos);
    const reflectDir = Vector.minus(d, Vector.times(2, Vector.times(Vector.dot(normal, d), normal)));
    const naturalColor = Color.plus(background, getNaturalColor(thing, pos, normal, reflectDir, scene));
    const reflectedColor = (depth >= maxDepth) ? grey : getReflectionColor(thing, pos, reflectDir, scene, depth);
    return Color.plus(naturalColor, reflectedColor);
}

function getReflectionColor(thing: Thing, pos: Vector, rd: Vector, scene: Scene, depth: number): Color {
    return Color.scale(reflect(thing.surface, pos), traceRay(new Ray(pos, rd), scene, depth + 1));
}

function getNaturalColor(thing: Thing, pos: Vector, norm: Vector, rd: Vector, scene: Scene): Color {
    const lights = scene.lights;
    const surface = thing.surface;
    let col = defaultColor;
    for (let i = 0; i < lights.length; i++) {
        const light = lights[i];
        const ldis = Vector.minus(light.pos, pos);
        const livec = Vector.norm(ldis);
        // in shadow: something is nearer along the ray to the light than the light itself
        if (intersections(new Ray(pos, livec), scene).dist <= Vector.mag(ldis)) {
            continue;
        }

        const illum = Vector.dot(livec, norm);
        const lcolor = (illum > 0) ? Color.scale(illum, light.color) : defaultColor;
        const spec = Vector.dot(livec, Vector.norm(rd));
        const scolor = (spec > 0) ? Color.scale(pow(spec, roughness(surface)), light.color) : defaultColor;
        col = Color.plus(col, Color.plus(Color.times(diffuse(surface, pos), lcolor),
            Color.times(specular(surface), scolor)));
    }

    return col;
}

function getPoint(x: number, y: number, camera: Camera, screenWidth: number, screenHeight: number): Vector {
    const recenterX = (x - (screenWidth / 2.0)) / 2.0 / screenWidth;
    const recenterY = - (y - (screenHeight / 2.0)) / 2.0 / screenHeight;
    return Vector.norm(Vector.plus(camera.forward,
        Vector.plus(Vector.times(recenterX, camera.right), Vector.times(recenterY, camera.up))));
}

function legalize(d: number) {
    return d > 1 ? 1 : d;
}

function render(scene: Scene, screenWidth: number, screenHeight: number): number {
    const camera = scene.camera;
    let sum: number = 0;
    for (let y = 0; y < screenHeight; y++) {
        for (let x = 0; x < screenWidth; x++) {
            const color = traceRay(new Ray(camera.pos, getPoint(x, y, camera, screenWidth, screenHeight)), scene, 0);
            sum += floor(legalize(color.r) * 255) + floor(legalize(color.g) * 255) + floor(legalize(color.b) * 255);
        }
    }

    return sum;
}

function defaultScene(): Scene {
    return new Scene(
        [Thing.plane(new Vector(0.0, 1.0, 0.0), 0.0, Surface.Checkerboard),
        Thing.sphere(new Vector(0.0, 1.0, -0.25), 1.0, Surface.Shiny),
        Thing.sphere(new Vector(-6, 3.0, -5.0), 1.0, Surface.Shiny),
        Thing.sphere(new Vector(-1.0, 0.5, 1.5), 0.5, Surface.Shiny)],
        [new Light(new Vector(-2.0, 2.5, 0.0), new Color(0.49, 0.07, 0.07)),
        new Light(new Vector(1.5, 2.5, 1.5), new Color(0.07, 0.07, 0.49)),
        new Light(new Vector(1.5, 2.5, -1.5), new Color(0.07, 0.49, 0.071)),
        new Light(new Vector(0.0, 3.5, 0.0), new Color(0.21, 0.21, 0.35))],
        new Camera(new Vector(3.0, 2.0, 4.0), new Vector(-1.0, 0.5, 0.0)));
}

function main() {
    print("start...");
    const sum = render(defaultScene(), 512, 512);
    print("checksum:", sum, "low digits:", sum - 38837000);
    print("done.");
}
