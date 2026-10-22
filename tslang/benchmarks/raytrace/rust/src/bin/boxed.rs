// Rust port of raytrace.ts that keeps the TypeScript allocation pattern: every Vector, Color,
// Ray and Intersection is a fresh reference-counted heap object (Rc), things are Rc<dyn Thing>,
// surface members are Rc<dyn Fn> closures. Same algorithm as idiomatic.rs.

use std::rc::Rc;

struct Vector { x: f64, y: f64, z: f64 }
type V = Rc<Vector>;

fn vec3(x: f64, y: f64, z: f64) -> V { Rc::new(Vector { x, y, z }) }

impl Vector {
    fn times(k: f64, v: &V) -> V { vec3(k * v.x, k * v.y, k * v.z) }
    fn minus(a: &V, b: &V) -> V { vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    fn plus(a: &V, b: &V) -> V { vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    fn dot(a: &V, b: &V) -> f64 { a.x * b.x + a.y * b.y + a.z * b.z }
    fn mag(v: &V) -> f64 { (v.x * v.x + v.y * v.y + v.z * v.z).sqrt() }
    fn norm(v: &V) -> V {
        let mag = Vector::mag(v);
        let div = if mag == 0.0 { f64::INFINITY } else { 1.0 / mag };
        Vector::times(div, v)
    }
    fn cross(a: &V, b: &V) -> V {
        vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }
}

struct Color { r: f64, g: f64, b: f64 }
type C = Rc<Color>;

fn rgb(r: f64, g: f64, b: f64) -> C { Rc::new(Color { r, g, b }) }

struct Colors { white: C, grey: C, black: C, background: C, default_color: C }

impl Color {
    fn scale(k: f64, v: &C) -> C { rgb(k * v.r, k * v.g, k * v.b) }
    fn plus(a: &C, b: &C) -> C { rgb(a.r + b.r, a.g + b.g, a.b + b.b) }
    fn times(a: &C, b: &C) -> C { rgb(a.r * b.r, a.g * b.g, a.b * b.b) }
    fn to_drawing_color(c: &C) -> Rc<(f64, f64, f64)> {
        let legalize = |d: f64| if d > 1.0 { 1.0 } else { d };
        Rc::new(((legalize(c.r) * 255.0).floor(), (legalize(c.g) * 255.0).floor(), (legalize(c.b) * 255.0).floor()))
    }
}

struct Camera { pos: V, forward: V, right: V, up: V }

impl Camera {
    fn new(pos: V, look_at: V) -> Rc<Self> {
        let down = vec3(0.0, -1.0, 0.0);
        let forward = Vector::norm(&Vector::minus(&look_at, &pos));
        let right = Vector::times(1.5, &Vector::norm(&Vector::cross(&forward, &down)));
        let up = Vector::times(1.5, &Vector::norm(&Vector::cross(&forward, &right)));
        Rc::new(Camera { pos, forward, right, up })
    }
}

struct Ray { start: V, dir: V }
struct Intersection { thing: Rc<dyn Thing>, ray: Rc<Ray>, dist: f64 }

struct Surface {
    diffuse: Rc<dyn Fn(&V) -> C>,
    specular: Rc<dyn Fn(&V) -> C>,
    reflect: Rc<dyn Fn(&V) -> f64>,
    roughness: f64,
}

trait Thing {
    fn intersect(self: Rc<Self>, ray: &Rc<Ray>) -> Option<Rc<Intersection>>;
    fn normal(&self, pos: &V) -> V;
    fn surface(&self) -> &Rc<Surface>;
}

struct Light { pos: V, color: C }
struct Scene { things: Vec<Rc<dyn Thing>>, lights: Vec<Rc<Light>>, camera: Rc<Camera> }

struct Sphere { center: V, radius2: f64, surface: Rc<Surface> }

impl Thing for Sphere {
    fn normal(&self, pos: &V) -> V { Vector::norm(&Vector::minus(pos, &self.center)) }
    fn intersect(self: Rc<Self>, ray: &Rc<Ray>) -> Option<Rc<Intersection>> {
        let eo = Vector::minus(&self.center, &ray.start);
        let v = Vector::dot(&eo, &ray.dir);
        let mut dist = 0.0;
        if v >= 0.0 {
            let disc = self.radius2 - (Vector::dot(&eo, &eo) - v * v);
            if disc >= 0.0 { dist = v - disc.sqrt(); }
        }
        if dist == 0.0 { None } else { Some(Rc::new(Intersection { thing: self, ray: ray.clone(), dist })) }
    }
    fn surface(&self) -> &Rc<Surface> { &self.surface }
}

struct Plane { norm: V, offset: f64, surface: Rc<Surface> }

impl Thing for Plane {
    fn normal(&self, _pos: &V) -> V { self.norm.clone() }
    fn intersect(self: Rc<Self>, ray: &Rc<Ray>) -> Option<Rc<Intersection>> {
        let denom = Vector::dot(&self.norm, &ray.dir);
        if denom > 0.0 {
            None
        } else {
            let dist = (Vector::dot(&self.norm, &ray.start) + self.offset) / (-denom);
            Some(Rc::new(Intersection { thing: self, ray: ray.clone(), dist }))
        }
    }
    fn surface(&self) -> &Rc<Surface> { &self.surface }
}

struct RayTracer { max_depth: i32, colors: Rc<Colors> }

impl RayTracer {
    fn intersections(&self, ray: &Rc<Ray>, scene: &Scene) -> Option<Rc<Intersection>> {
        let mut closest = f64::INFINITY;
        let mut closest_inter = None;
        for thing in &scene.things {
            if let Some(inter) = thing.clone().intersect(ray) {
                if inter.dist < closest {
                    closest = inter.dist;
                    closest_inter = Some(inter);
                }
            }
        }
        closest_inter
    }

    fn test_ray(&self, ray: &Rc<Ray>, scene: &Scene) -> Option<f64> {
        self.intersections(ray, scene).map(|i| i.dist)
    }

    fn trace_ray(&self, ray: &Rc<Ray>, scene: &Scene, depth: i32) -> C {
        match self.intersections(ray, scene) {
            None => self.colors.background.clone(),
            Some(isect) => self.shade(&isect, scene, depth),
        }
    }

    fn shade(&self, isect: &Rc<Intersection>, scene: &Scene, depth: i32) -> C {
        let d = isect.ray.dir.clone();
        let pos = Vector::plus(&Vector::times(isect.dist, &d), &isect.ray.start);
        let normal = isect.thing.normal(&pos);
        let reflect_dir = Vector::minus(&d, &Vector::times(2.0, &Vector::times(Vector::dot(&normal, &d), &normal)));
        let natural_color = Color::plus(&self.colors.background,
            &self.get_natural_color(&isect.thing, &pos, &normal, &reflect_dir, scene));
        let reflected_color = if depth >= self.max_depth {
            self.colors.grey.clone()
        } else {
            self.get_reflection_color(&isect.thing, &pos, &normal, &reflect_dir, scene, depth)
        };
        Color::plus(&natural_color, &reflected_color)
    }

    fn get_reflection_color(&self, thing: &Rc<dyn Thing>, pos: &V, _normal: &V, rd: &V, scene: &Scene, depth: i32) -> C {
        let ray = Rc::new(Ray { start: pos.clone(), dir: rd.clone() });
        Color::scale((thing.surface().reflect)(pos), &self.trace_ray(&ray, scene, depth + 1))
    }

    fn get_natural_color(&self, thing: &Rc<dyn Thing>, pos: &V, norm: &V, rd: &V, scene: &Scene) -> C {
        let add_light = |col: C, light: &Rc<Light>| -> C {
            let ldis = Vector::minus(&light.pos, pos);
            let livec = Vector::norm(&ldis);
            let neat_isect = self.test_ray(&Rc::new(Ray { start: pos.clone(), dir: livec.clone() }), scene);
            let is_in_shadow = match neat_isect { None => false, Some(d) => d <= Vector::mag(&ldis) };
            if is_in_shadow {
                col
            } else {
                let illum = Vector::dot(&livec, norm);
                let lcolor = if illum > 0.0 { Color::scale(illum, &light.color) } else { self.colors.default_color.clone() };
                let specular = Vector::dot(&livec, &Vector::norm(rd));
                let scolor = if specular > 0.0 {
                    Color::scale(specular.powf(thing.surface().roughness), &light.color)
                } else {
                    self.colors.default_color.clone()
                };
                Color::plus(&col, &Color::plus(&Color::times(&(thing.surface().diffuse)(pos), &lcolor),
                    &Color::times(&(thing.surface().specular)(pos), &scolor)))
            }
        };
        scene.lights.iter().fold(self.colors.default_color.clone(), add_light)
    }

    fn render(&self, scene: &Scene, screen_width: f64, screen_height: f64) -> f64 {
        let get_point = |x: f64, y: f64, camera: &Rc<Camera>| {
            let recenter_x = |x: f64| (x - (screen_width / 2.0)) / 2.0 / screen_width;
            let recenter_y = |y: f64| -(y - (screen_height / 2.0)) / 2.0 / screen_height;
            Vector::norm(&Vector::plus(&camera.forward, &Vector::plus(&Vector::times(recenter_x(x), &camera.right), &Vector::times(recenter_y(y), &camera.up))))
        };
        let mut sum = 0.0;
        let mut y = 0.0;
        while y < screen_height {
            let mut x = 0.0;
            while x < screen_width {
                let ray = Rc::new(Ray { start: scene.camera.pos.clone(), dir: get_point(x, y, &scene.camera) });
                let color = self.trace_ray(&ray, scene, 0);
                let c = Color::to_drawing_color(&color);
                sum += c.0 + c.1 + c.2;
                x += 1.0;
            }
            y += 1.0;
        }
        sum
    }
}

fn surfaces(colors: &Rc<Colors>) -> (Rc<Surface>, Rc<Surface>) {
    let (w, g) = (colors.white.clone(), colors.grey.clone());
    let w2 = colors.white.clone();
    let shiny = Rc::new(Surface {
        diffuse: Rc::new(move |_| w.clone()),
        specular: Rc::new(move |_| g.clone()),
        reflect: Rc::new(|_| 0.7),
        roughness: 250.0,
    });
    let (w3, b3) = (colors.white.clone(), colors.black.clone());
    let checkerboard = Rc::new(Surface {
        diffuse: Rc::new(move |pos: &V| if (pos.z.floor() + pos.x.floor()) % 2.0 != 0.0 { w3.clone() } else { b3.clone() }),
        specular: Rc::new(move |_| w2.clone()),
        reflect: Rc::new(|pos: &V| if (pos.z.floor() + pos.x.floor()) % 2.0 != 0.0 { 0.1 } else { 0.7 }),
        roughness: 150.0,
    });
    (shiny, checkerboard)
}

fn default_scene(colors: &Rc<Colors>) -> Scene {
    let (shiny, checkerboard) = surfaces(colors);
    let sphere = |c: V, r: f64, s: &Rc<Surface>| -> Rc<dyn Thing> { Rc::new(Sphere { center: c, radius2: r * r, surface: s.clone() }) };
    let light = |pos: V, color: C| Rc::new(Light { pos, color });
    Scene {
        things: vec![
            Rc::new(Plane { norm: vec3(0.0, 1.0, 0.0), offset: 0.0, surface: checkerboard }),
            sphere(vec3(0.0, 1.0, -0.25), 1.0, &shiny),
            sphere(vec3(-6.0, 3.0, -5.0), 1.0, &shiny),
            sphere(vec3(-1.0, 0.5, 1.5), 0.5, &shiny),
        ],
        lights: vec![
            light(vec3(-2.0, 2.5, 0.0), rgb(0.49, 0.07, 0.07)),
            light(vec3(1.5, 2.5, 1.5), rgb(0.07, 0.07, 0.49)),
            light(vec3(1.5, 2.5, -1.5), rgb(0.07, 0.49, 0.071)),
            light(vec3(0.0, 3.5, 0.0), rgb(0.21, 0.21, 0.35)),
        ],
        camera: Camera::new(vec3(3.0, 2.0, 4.0), vec3(-1.0, 0.5, 0.0)),
    }
}

fn main() {
    println!("start...");
    let black = rgb(0.0, 0.0, 0.0);
    let colors = Rc::new(Colors {
        white: rgb(1.0, 1.0, 1.0),
        grey: rgb(0.5, 0.5, 0.5),
        background: black.clone(),
        default_color: black.clone(),
        black,
    });
    let ray_tracer = RayTracer { max_depth: 5, colors: colors.clone() };
    let size: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512.0);
    let sum = ray_tracer.render(&default_scene(&colors), size, size);
    println!("checksum: {}", sum);
    println!("done.");
}
