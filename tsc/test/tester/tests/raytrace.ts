// @strict-null false
declare function sqrt(v: number): number;
declare function floor(v: number): number;
declare function pow(v: number, s: number): number;
declare function fabs(v: number): number;

class Math {
    static sqrt(v: number): number {
        return sqrt(v);
    }

    static floor(v: number): number {
        return floor(v);
    }

    static pow(v: number, s: number): number {
        return pow(v, s);
    }

    static abs(v: number): number {
        return fabs(v);
    }

    static sign(v: number): number {
	if (v > 0) return 1;
	if (v < 0) return -1;
	return 0;
    }
}

class Vector {
    constructor(public x: number,
        public y: number,
        public z: number) {
    }
    static times(k: number, v: Vector) { return new Vector(k * v.x, k * v.y, k * v.z); }
    static minus(v1: Vector, v2: Vector) { return new Vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
    static plus(v1: Vector, v2: Vector) { return new Vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
    static dot(v1: Vector, v2: Vector) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
    static mag(v: Vector) { return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    static norm(v: Vector) {
        let mag = Vector.mag(v);
        let div = (mag === 0) ? Infinity : 1.0 / mag;
        return Vector.times(div, v);
    }
    static cross(v1: Vector, v2: Vector) {
        return new Vector(v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x);
    }
}

class Color {
    constructor(public r: number,
        public g: number,
        public b: number) {
    }
    static scale(k: number, v: Color) { return new Color(k * v.r, k * v.g, k * v.b); }
    static plus(v1: Color, v2: Color) { return new Color(v1.r + v2.r, v1.g + v2.g, v1.b + v2.b); }
    static times(v1: Color, v2: Color) { return new Color(v1.r * v2.r, v1.g * v2.g, v1.b * v2.b); }
    static white = new Color(1.0, 1.0, 1.0);
    static grey = new Color(0.5, 0.5, 0.5);
    static black = new Color(0.0, 0.0, 0.0);
    static background = Color.black;
    static defaultColor = Color.black;
    static toDrawingColor(c: Color) {
        let legalize = (d: number) => d > 1 ? 1 : d;
        return {
            r: Math.floor(legalize(c.r) * 255),
            g: Math.floor(legalize(c.g) * 255),
            b: Math.floor(legalize(c.b) * 255)
        }
    }
}

class Camera {
    public forward: Vector;
    public right: Vector;
    public up: Vector;

    constructor(public pos: Vector, lookAt: Vector) {
        let down = new Vector(0.0, -1.0, 0.0);
        this.forward = Vector.norm(Vector.minus(lookAt, this.pos));
        this.right = Vector.times(1.5, Vector.norm(Vector.cross(this.forward, down)));
        this.up = Vector.times(1.5, Vector.norm(Vector.cross(this.forward, this.right)));
    }
}

interface Ray {
    start: Vector;
    dir: Vector;
}

interface Intersection {
    thing: Thing;
    ray: Ray;
    dist: number;
}

interface Surface {
    diffuse: (pos: Vector) => Color;
    specular: (pos: Vector) => Color;
    reflect: (pos: Vector) => number;
    roughness: number;
}

interface Thing {
    intersect(ray: Ray): Intersection;
    normal(pos: Vector): Vector;
    surface: Surface;
}

interface Light {
    pos: Vector;
    color: Color;
}

interface Scene {
    things: Thing[];
    lights: Light[];
    camera: Camera;
}

class Sphere implements Thing {
    public radius2: number;

    constructor(public center: Vector, radius: number, public surface: Surface) {
        this.radius2 = radius * radius;
    }
    normal(pos: Vector): Vector { return Vector.norm(Vector.minus(pos, this.center)); }
    intersect(ray: Ray): Intersection {
        let eo = Vector.minus(this.center, ray.start);
        let v = Vector.dot(eo, ray.dir);
        let dist = 0.0;
        if (v >= 0) {
            let disc = this.radius2 - (Vector.dot(eo, eo) - v * v);
            if (disc >= 0) {
                dist = v - Math.sqrt(disc);
            }
        }
        if (dist === 0) {
            return null;
        } else {
            return { thing: this, ray: ray, dist: dist };
        }
    }
}

class Plane implements Thing {
    constructor(public norm: Vector, public offset: number, public surface: Surface) {
    }
    public normal(pos: Vector): Vector { return this.norm; }
    public intersect(ray: Ray): Intersection {
        let denom = Vector.dot(this.norm, ray.dir);
        if (denom > 0) {
            return null;
        } else {
            let dist = (Vector.dot(this.norm, ray.start) + this.offset) / (-denom);
            return { thing: this, ray: ray, dist: dist };
        }
    }
}

module Surfaces {
    export let shiny: Surface = {
        diffuse: function (pos: Vector) { return Color.white; },
        specular: function (pos: Vector) { return Color.grey; },
        reflect: function (pos: Vector) { return 0.7; },
        roughness: 250.0
    }
    export let checkerboard: Surface = {
        diffuse: function (pos: Vector) {
            if ((Math.floor(pos.z) + Math.floor(pos.x)) % 2 !== 0) {
                return Color.white;
            } else {
                return Color.black;
            }
        },
        specular: function (pos: Vector) { return Color.white; },
        reflect: function (pos: Vector) {
            if ((Math.floor(pos.z) + Math.floor(pos.x)) % 2 !== 0) {
                return 0.1;
            } else {
                return 0.7;
            }
        },
        roughness: 150.0
    }
}


class RayTracer {
    private maxDepth = 5;

    private intersections(ray: Ray, scene: Scene) {
        let closest = +Infinity;
        let closestInter: Intersection = undefined;
        for (let i in scene.things) {
            let inter = scene.things[i].intersect(ray);
            if (inter != null && inter.dist < closest) {
                closestInter = inter;
                closest = inter.dist;
            }
        }

        return closestInter;
    }

    private testRay(ray: Ray, scene: Scene) {
        let isect = this.intersections(ray, scene);
        if (isect != null) {
            return isect.dist;
        } else {
            return undefined;
        }
    }

    private traceRay(ray: Ray, scene: Scene, depth: number): Color {
        let isect = this.intersections(ray, scene);
        if (isect === undefined) {
            return Color.background;
        } else {
            return this.shade(isect, scene, depth);
        }
    }

    private shade(isect: Intersection, scene: Scene, depth: number) {
        let d = isect.ray.dir;
        let pos = Vector.plus(Vector.times(isect.dist, d), isect.ray.start);
        let normal = isect.thing.normal(pos);
        let reflectDir = Vector.minus(d, Vector.times(2, Vector.times(Vector.dot(normal, d), normal)));
        let naturalColor = Color.plus(Color.background,
            this.getNaturalColor(isect.thing, pos, normal, reflectDir, scene));
        let reflectedColor = (depth >= this.maxDepth) ? Color.grey : this.getReflectionColor(isect.thing, pos, normal, reflectDir, scene, depth);
        return Color.plus(naturalColor, reflectedColor);
    }

    private getReflectionColor(thing: Thing, pos: Vector, normal: Vector, rd: Vector, scene: Scene, depth: number) {
        return Color.scale(thing.surface.reflect(pos), this.traceRay({ start: pos, dir: rd }, scene, depth + 1));
    }

    private getNaturalColor(thing: Thing, pos: Vector, norm: Vector, rd: Vector, scene: Scene) {
        const addLight = (col: Color, light: Light) => {
            let ldis = Vector.minus(light.pos, pos);
            let livec = Vector.norm(ldis);
            let neatIsect = this.testRay({ start: pos, dir: livec }, scene);
            let isInShadow = (neatIsect === undefined) ? false : (neatIsect <= Vector.mag(ldis));
            if (isInShadow) {
                return col;
            } else {
                let illum = Vector.dot(livec, norm);
                let lcolor = (illum > 0) ? Color.scale(illum, light.color)
                    : Color.defaultColor;
                let specular = Vector.dot(livec, Vector.norm(rd));
                let scolor = (specular > 0) ? Color.scale(Math.pow(specular, thing.surface.roughness), light.color)
                    : Color.defaultColor;
                return Color.plus(col, Color.plus(Color.times(thing.surface.diffuse(pos), lcolor),
                    Color.times(thing.surface.specular(pos), scolor)));
            }
        }
        
        return scene.lights.reduce(addLight, Color.defaultColor);
    }

    render(scene: Scene, screenWidth: number, screenHeight: number) {
        const getPoint = (x: number, y: number, camera: Camera) => {
            const recenterX = (x: number) => (x - (screenWidth / 2.0)) / 2.0 / screenWidth;
            const recenterY = (y: number) => - (y - (screenHeight / 2.0)) / 2.0 / screenHeight;
            return Vector.norm(Vector.plus(camera.forward, Vector.plus(Vector.times(recenterX(x), camera.right), Vector.times(recenterY(y), camera.up))));
        }

        for (let y = 0; y < screenHeight; y++) {
            for (let x = 0; x < screenWidth; x++) {
                let color = this.traceRay({ start: scene.camera.pos, dir: getPoint(x, y, scene.camera) }, scene, 0);
                let c = Color.toDrawingColor(color);
                //ctx.fillStyle = "rgb(" + String(c.r) + ", " + String(c.g) + ", " + String(c.b) + ")";
                //ctx.fillRect(x, y, x + 1, y + 1);
            }
        }
    }
}


function defaultScene(): Scene {
    return {
        things: [new Plane(new Vector(0.0, 1.0, 0.0), 0.0, Surfaces.checkerboard),
        new Sphere(new Vector(0.0, 1.0, -0.25), 1.0, Surfaces.shiny),
        new Sphere(new Vector(-6, 3.0, -5.0), 1.0, Surfaces.shiny),
        // new Sphere(new Vector(-6, 8.0, 15.0), 5.0, Surfaces.shiny),
        new Sphere(new Vector(-1.0, 0.5, 1.5), 0.5, Surfaces.shiny)],

        lights: [{ pos: new Vector(-2.0, 2.5, 0.0), color: new Color(0.49, 0.07, 0.07) },
        { pos: new Vector(1.5, 2.5, 1.5), color: new Color(0.07, 0.07, 0.49) },
        { pos: new Vector(1.5, 2.5, -1.5), color: new Color(0.07, 0.49, 0.071) },
        { pos: new Vector(0.0, 3.5, 0.0), color: new Color(0.21, 0.21, 0.35) }],
        camera: new Camera(new Vector(3.0, 2.0, 4.0), new Vector(-1.0, 0.5, 0.0))
    };
}

function exec() {
    let rayTracer = new RayTracer();
    rayTracer.render(defaultScene(), 256, 256);
}


function main() {
    print("start...");
    exec();
    print("done.");
}
