// The ray tracer of idiomatic.rs with no trait objects and no function pointers: a thing and a
// surface are enums and every call is a `match`, so all of it is static dispatch. This is the
// program rtbench_static.ts mirrors.

#[derive(Clone, Copy)]
struct Vector { x: f64, y: f64, z: f64 }

impl Vector {
    fn new(x: f64, y: f64, z: f64) -> Self { Vector { x, y, z } }
    fn times(k: f64, v: Vector) -> Vector { Vector::new(k * v.x, k * v.y, k * v.z) }
    fn minus(a: Vector, b: Vector) -> Vector { Vector::new(a.x - b.x, a.y - b.y, a.z - b.z) }
    fn plus(a: Vector, b: Vector) -> Vector { Vector::new(a.x + b.x, a.y + b.y, a.z + b.z) }
    fn dot(a: Vector, b: Vector) -> f64 { a.x * b.x + a.y * b.y + a.z * b.z }
    fn mag(v: Vector) -> f64 { (v.x * v.x + v.y * v.y + v.z * v.z).sqrt() }
    fn norm(v: Vector) -> Vector {
        let mag = Vector::mag(v);
        let div = if mag == 0.0 { f64::INFINITY } else { 1.0 / mag };
        Vector::times(div, v)
    }
    fn cross(a: Vector, b: Vector) -> Vector {
        Vector::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
}

#[derive(Clone, Copy)]
struct Color { r: f64, g: f64, b: f64 }

const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0 };
const GREY: Color = Color { r: 0.5, g: 0.5, b: 0.5 };
const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0 };
const BACKGROUND: Color = BLACK;
const DEFAULT_COLOR: Color = BLACK;

impl Color {
    fn scale(k: f64, v: Color) -> Color { Color { r: k * v.r, g: k * v.g, b: k * v.b } }
    fn plus(a: Color, b: Color) -> Color { Color { r: a.r + b.r, g: a.g + b.g, b: a.b + b.b } }
    fn times(a: Color, b: Color) -> Color { Color { r: a.r * b.r, g: a.g * b.g, b: a.b * b.b } }
}

#[derive(Clone, Copy)]
struct Ray { start: Vector, dir: Vector }

struct Light { pos: Vector, color: Color }

struct Camera { pos: Vector, forward: Vector, right: Vector, up: Vector }

impl Camera {
    fn new(pos: Vector, look_at: Vector) -> Self {
        let down = Vector::new(0.0, -1.0, 0.0);
        let forward = Vector::norm(Vector::minus(look_at, pos));
        let right = Vector::times(1.5, Vector::norm(Vector::cross(forward, down)));
        let up = Vector::times(1.5, Vector::norm(Vector::cross(forward, right)));
        Camera { pos, forward, right, up }
    }
}

#[derive(Clone, Copy)]
enum Surface { Shiny, Checkerboard }

fn is_white_square(pos: Vector) -> bool { (pos.z.floor() + pos.x.floor()) % 2.0 != 0.0 }

impl Surface {
    fn diffuse(self, pos: Vector) -> Color {
        match self {
            Surface::Shiny => WHITE,
            Surface::Checkerboard => if is_white_square(pos) { WHITE } else { BLACK },
        }
    }
    fn specular(self) -> Color {
        match self {
            Surface::Shiny => GREY,
            Surface::Checkerboard => WHITE,
        }
    }
    fn reflect(self, pos: Vector) -> f64 {
        match self {
            Surface::Shiny => 0.7,
            Surface::Checkerboard => if is_white_square(pos) { 0.1 } else { 0.7 },
        }
    }
    fn roughness(self) -> f64 {
        match self {
            Surface::Shiny => 250.0,
            Surface::Checkerboard => 150.0,
        }
    }
}

enum Thing {
    Sphere { center: Vector, radius2: f64, surface: Surface },
    Plane { norm: Vector, offset: f64, surface: Surface },
}

impl Thing {
    fn surface(&self) -> Surface {
        match *self {
            Thing::Sphere { surface, .. } | Thing::Plane { surface, .. } => surface,
        }
    }

    fn normal(&self, pos: Vector) -> Vector {
        match *self {
            Thing::Sphere { center, .. } => Vector::norm(Vector::minus(pos, center)),
            Thing::Plane { norm, .. } => norm,
        }
    }

    // The distance along the ray to the thing, if the ray hits it.
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        match *self {
            Thing::Sphere { center, radius2, .. } => {
                let eo = Vector::minus(center, ray.start);
                let v = Vector::dot(eo, ray.dir);
                let mut dist = 0.0;
                if v >= 0.0 {
                    let disc = radius2 - (Vector::dot(eo, eo) - v * v);
                    if disc >= 0.0 { dist = v - disc.sqrt(); }
                }
                if dist == 0.0 { None } else { Some(dist) }
            }
            Thing::Plane { norm, offset, .. } => {
                let denom = Vector::dot(norm, ray.dir);
                if denom > 0.0 { None } else { Some((Vector::dot(norm, ray.start) + offset) / (-denom)) }
            }
        }
    }
}

struct Scene { things: Vec<Thing>, lights: Vec<Light>, camera: Camera }

// What a ray hit first: the thing and the distance to it.
struct Hit<'a> { thing: &'a Thing, dist: f64 }

const MAX_DEPTH: i32 = 5;

fn intersections<'a>(ray: &Ray, scene: &'a Scene) -> Option<Hit<'a>> {
    let mut closest: Option<Hit<'a>> = None;
    for thing in &scene.things {
        if let Some(dist) = thing.intersect(ray) {
            if closest.as_ref().map_or(true, |hit| dist < hit.dist) {
                closest = Some(Hit { thing, dist });
            }
        }
    }
    closest
}

fn trace_ray(ray: &Ray, scene: &Scene, depth: i32) -> Color {
    match intersections(ray, scene) {
        None => BACKGROUND,
        Some(hit) => shade(&hit, ray, scene, depth),
    }
}

fn shade(hit: &Hit, ray: &Ray, scene: &Scene, depth: i32) -> Color {
    let thing = hit.thing;
    let d = ray.dir;
    let pos = Vector::plus(Vector::times(hit.dist, d), ray.start);
    let normal = thing.normal(pos);
    let reflect_dir = Vector::minus(d, Vector::times(2.0, Vector::times(Vector::dot(normal, d), normal)));
    let natural_color = Color::plus(BACKGROUND, get_natural_color(thing, pos, normal, reflect_dir, scene));
    let reflected_color = if depth >= MAX_DEPTH {
        GREY
    } else {
        get_reflection_color(thing, pos, reflect_dir, scene, depth)
    };
    Color::plus(natural_color, reflected_color)
}

fn get_reflection_color(thing: &Thing, pos: Vector, rd: Vector, scene: &Scene, depth: i32) -> Color {
    Color::scale(thing.surface().reflect(pos), trace_ray(&Ray { start: pos, dir: rd }, scene, depth + 1))
}

fn get_natural_color(thing: &Thing, pos: Vector, norm: Vector, rd: Vector, scene: &Scene) -> Color {
    let surface = thing.surface();
    let mut col = DEFAULT_COLOR;
    for light in &scene.lights {
        let ldis = Vector::minus(light.pos, pos);
        let livec = Vector::norm(ldis);
        // in shadow: something is nearer along the ray to the light than the light itself
        if let Some(hit) = intersections(&Ray { start: pos, dir: livec }, scene) {
            if hit.dist <= Vector::mag(ldis) { continue; }
        }

        let illum = Vector::dot(livec, norm);
        let lcolor = if illum > 0.0 { Color::scale(illum, light.color) } else { DEFAULT_COLOR };
        let spec = Vector::dot(livec, Vector::norm(rd));
        let scolor = if spec > 0.0 { Color::scale(spec.powf(surface.roughness()), light.color) } else { DEFAULT_COLOR };
        col = Color::plus(col, Color::plus(Color::times(surface.diffuse(pos), lcolor),
            Color::times(surface.specular(), scolor)));
    }
    col
}

fn get_point(x: f64, y: f64, camera: &Camera, screen_width: f64, screen_height: f64) -> Vector {
    let recenter_x = (x - (screen_width / 2.0)) / 2.0 / screen_width;
    let recenter_y = -(y - (screen_height / 2.0)) / 2.0 / screen_height;
    Vector::norm(Vector::plus(camera.forward,
        Vector::plus(Vector::times(recenter_x, camera.right), Vector::times(recenter_y, camera.up))))
}

fn legalize(d: f64) -> f64 { if d > 1.0 { 1.0 } else { d } }

fn render(scene: &Scene, screen_width: f64, screen_height: f64) -> f64 {
    let camera = &scene.camera;
    let mut sum = 0.0;
    let mut y = 0.0;
    while y < screen_height {
        let mut x = 0.0;
        while x < screen_width {
            let ray = Ray { start: camera.pos, dir: get_point(x, y, camera, screen_width, screen_height) };
            let color = trace_ray(&ray, scene, 0);
            sum += (legalize(color.r) * 255.0).floor() + (legalize(color.g) * 255.0).floor()
                + (legalize(color.b) * 255.0).floor();
            x += 1.0;
        }
        y += 1.0;
    }
    sum
}

fn default_scene() -> Scene {
    Scene {
        things: vec![
            Thing::Plane { norm: Vector::new(0.0, 1.0, 0.0), offset: 0.0, surface: Surface::Checkerboard },
            Thing::Sphere { center: Vector::new(0.0, 1.0, -0.25), radius2: 1.0, surface: Surface::Shiny },
            Thing::Sphere { center: Vector::new(-6.0, 3.0, -5.0), radius2: 1.0, surface: Surface::Shiny },
            Thing::Sphere { center: Vector::new(-1.0, 0.5, 1.5), radius2: 0.25, surface: Surface::Shiny },
        ],
        lights: vec![
            Light { pos: Vector::new(-2.0, 2.5, 0.0), color: Color { r: 0.49, g: 0.07, b: 0.07 } },
            Light { pos: Vector::new(1.5, 2.5, 1.5), color: Color { r: 0.07, g: 0.07, b: 0.49 } },
            Light { pos: Vector::new(1.5, 2.5, -1.5), color: Color { r: 0.07, g: 0.49, b: 0.071 } },
            Light { pos: Vector::new(0.0, 3.5, 0.0), color: Color { r: 0.21, g: 0.21, b: 0.35 } },
        ],
        camera: Camera::new(Vector::new(3.0, 2.0, 4.0), Vector::new(-1.0, 0.5, 0.0)),
    }
}

fn main() {
    println!("start...");
    let size: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512.0);
    let sum = render(&default_scene(), size, size);
    println!("checksum: {}", sum);
    println!("done.");
}
