// Idiomatic Rust port of tslang/test/tester/tests/raytrace.ts:
// Vector/Color are Copy values, things are Box<dyn Thing>, surfaces hold fn pointers.

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
    fn to_drawing_color(c: Color) -> (f64, f64, f64) {
        let legalize = |d: f64| if d > 1.0 { 1.0 } else { d };
        ((legalize(c.r) * 255.0).floor(), (legalize(c.g) * 255.0).floor(), (legalize(c.b) * 255.0).floor())
    }
}

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
struct Ray { start: Vector, dir: Vector }

struct Intersection<'a> { thing: &'a dyn Thing, ray: Ray, dist: f64 }

struct Surface {
    diffuse: fn(Vector) -> Color,
    specular: fn(Vector) -> Color,
    reflect: fn(Vector) -> f64,
    roughness: f64,
}

trait Thing {
    fn intersect(&self, ray: Ray) -> Option<Intersection<'_>>;
    fn normal(&self, pos: Vector) -> Vector;
    fn surface(&self) -> &Surface;
}

struct Light { pos: Vector, color: Color }

struct Scene { things: Vec<Box<dyn Thing>>, lights: Vec<Light>, camera: Camera }

struct Sphere { center: Vector, radius2: f64, surface: &'static Surface }

impl Thing for Sphere {
    fn normal(&self, pos: Vector) -> Vector { Vector::norm(Vector::minus(pos, self.center)) }
    fn intersect(&self, ray: Ray) -> Option<Intersection<'_>> {
        let eo = Vector::minus(self.center, ray.start);
        let v = Vector::dot(eo, ray.dir);
        let mut dist = 0.0;
        if v >= 0.0 {
            let disc = self.radius2 - (Vector::dot(eo, eo) - v * v);
            if disc >= 0.0 { dist = v - disc.sqrt(); }
        }
        if dist == 0.0 { None } else { Some(Intersection { thing: self, ray, dist }) }
    }
    fn surface(&self) -> &Surface { self.surface }
}

struct Plane { norm: Vector, offset: f64, surface: &'static Surface }

impl Thing for Plane {
    fn normal(&self, _pos: Vector) -> Vector { self.norm }
    fn intersect(&self, ray: Ray) -> Option<Intersection<'_>> {
        let denom = Vector::dot(self.norm, ray.dir);
        if denom > 0.0 {
            None
        } else {
            let dist = (Vector::dot(self.norm, ray.start) + self.offset) / (-denom);
            Some(Intersection { thing: self, ray, dist })
        }
    }
    fn surface(&self) -> &Surface { self.surface }
}

static SHINY: Surface = Surface {
    diffuse: |_| WHITE,
    specular: |_| GREY,
    reflect: |_| 0.7,
    roughness: 250.0,
};

static CHECKERBOARD: Surface = Surface {
    diffuse: |pos| if (pos.z.floor() + pos.x.floor()) % 2.0 != 0.0 { WHITE } else { BLACK },
    specular: |_| WHITE,
    reflect: |pos| if (pos.z.floor() + pos.x.floor()) % 2.0 != 0.0 { 0.1 } else { 0.7 },
    roughness: 150.0,
};

struct RayTracer { max_depth: i32 }

impl RayTracer {
    fn intersections<'a>(&self, ray: Ray, scene: &'a Scene) -> Option<Intersection<'a>> {
        let mut closest = f64::INFINITY;
        let mut closest_inter = None;
        for thing in &scene.things {
            if let Some(inter) = thing.intersect(ray) {
                if inter.dist < closest {
                    closest = inter.dist;
                    closest_inter = Some(inter);
                }
            }
        }
        closest_inter
    }

    fn test_ray(&self, ray: Ray, scene: &Scene) -> Option<f64> {
        self.intersections(ray, scene).map(|i| i.dist)
    }

    fn trace_ray(&self, ray: Ray, scene: &Scene, depth: i32) -> Color {
        match self.intersections(ray, scene) {
            None => BACKGROUND,
            Some(isect) => self.shade(&isect, scene, depth),
        }
    }

    fn shade(&self, isect: &Intersection, scene: &Scene, depth: i32) -> Color {
        let d = isect.ray.dir;
        let pos = Vector::plus(Vector::times(isect.dist, d), isect.ray.start);
        let normal = isect.thing.normal(pos);
        let reflect_dir = Vector::minus(d, Vector::times(2.0, Vector::times(Vector::dot(normal, d), normal)));
        let natural_color = Color::plus(BACKGROUND, self.get_natural_color(isect.thing, pos, normal, reflect_dir, scene));
        let reflected_color = if depth >= self.max_depth {
            GREY
        } else {
            self.get_reflection_color(isect.thing, pos, normal, reflect_dir, scene, depth)
        };
        Color::plus(natural_color, reflected_color)
    }

    fn get_reflection_color(&self, thing: &dyn Thing, pos: Vector, _normal: Vector, rd: Vector, scene: &Scene, depth: i32) -> Color {
        Color::scale((thing.surface().reflect)(pos), self.trace_ray(Ray { start: pos, dir: rd }, scene, depth + 1))
    }

    fn get_natural_color(&self, thing: &dyn Thing, pos: Vector, norm: Vector, rd: Vector, scene: &Scene) -> Color {
        let add_light = |col: Color, light: &Light| {
            let ldis = Vector::minus(light.pos, pos);
            let livec = Vector::norm(ldis);
            let neat_isect = self.test_ray(Ray { start: pos, dir: livec }, scene);
            let is_in_shadow = match neat_isect { None => false, Some(d) => d <= Vector::mag(ldis) };
            if is_in_shadow {
                col
            } else {
                let illum = Vector::dot(livec, norm);
                let lcolor = if illum > 0.0 { Color::scale(illum, light.color) } else { DEFAULT_COLOR };
                let specular = Vector::dot(livec, Vector::norm(rd));
                let scolor = if specular > 0.0 {
                    Color::scale(specular.powf(thing.surface().roughness), light.color)
                } else {
                    DEFAULT_COLOR
                };
                Color::plus(col, Color::plus(Color::times((thing.surface().diffuse)(pos), lcolor),
                    Color::times((thing.surface().specular)(pos), scolor)))
            }
        };
        scene.lights.iter().fold(DEFAULT_COLOR, add_light)
    }

    fn render(&self, scene: &Scene, screen_width: f64, screen_height: f64) -> f64 {
        let get_point = |x: f64, y: f64, camera: &Camera| {
            let recenter_x = |x: f64| (x - (screen_width / 2.0)) / 2.0 / screen_width;
            let recenter_y = |y: f64| -(y - (screen_height / 2.0)) / 2.0 / screen_height;
            Vector::norm(Vector::plus(camera.forward, Vector::plus(Vector::times(recenter_x(x), camera.right), Vector::times(recenter_y(y), camera.up))))
        };
        let mut sum = 0.0;
        let mut y = 0.0;
        while y < screen_height {
            let mut x = 0.0;
            while x < screen_width {
                let color = self.trace_ray(Ray { start: scene.camera.pos, dir: get_point(x, y, &scene.camera) }, scene, 0);
                let c = Color::to_drawing_color(color);
                sum += c.0 + c.1 + c.2;
                x += 1.0;
            }
            y += 1.0;
        }
        sum
    }
}

fn default_scene() -> Scene {
    Scene {
        things: vec![
            Box::new(Plane { norm: Vector::new(0.0, 1.0, 0.0), offset: 0.0, surface: &CHECKERBOARD }),
            Box::new(Sphere { center: Vector::new(0.0, 1.0, -0.25), radius2: 1.0, surface: &SHINY }),
            Box::new(Sphere { center: Vector::new(-6.0, 3.0, -5.0), radius2: 1.0, surface: &SHINY }),
            Box::new(Sphere { center: Vector::new(-1.0, 0.5, 1.5), radius2: 0.25, surface: &SHINY }),
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
    let ray_tracer = RayTracer { max_depth: 5 };
    let size: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512.0);
    let sum = ray_tracer.render(&default_scene(), size, size);
    println!("checksum: {}", sum);
    println!("done.");
}
