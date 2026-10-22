// C++ port of rtbench.ts that keeps the TypeScript allocation pattern, like rust/src/bin/boxed.rs:
// every Vector, Color, Ray and Intersection is a fresh std::shared_ptr (make_shared, atomic
// counts), things are shared_ptr<Thing>, surface members are std::function. Same algorithm as
// idiomatic.cpp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

struct Vector;
using V = std::shared_ptr<const Vector>;
struct Vector {
    double x, y, z;
    Vector(double x, double y, double z) : x(x), y(y), z(z) {}
    static V make(double x, double y, double z) { return std::make_shared<const Vector>(x, y, z); }
    static V times(double k, const V &v) { return make(k * v->x, k * v->y, k * v->z); }
    static V minus(const V &a, const V &b) { return make(a->x - b->x, a->y - b->y, a->z - b->z); }
    static V plus(const V &a, const V &b) { return make(a->x + b->x, a->y + b->y, a->z + b->z); }
    static double dot(const V &a, const V &b) { return a->x * b->x + a->y * b->y + a->z * b->z; }
    static double mag(const V &v) { return std::sqrt(v->x * v->x + v->y * v->y + v->z * v->z); }
    static V norm(const V &v) {
        double m = mag(v);
        double div = (m == 0) ? std::numeric_limits<double>::infinity() : 1.0 / m;
        return times(div, v);
    }
    static V cross(const V &a, const V &b) {
        return make(a->y * b->z - a->z * b->y, a->z * b->x - a->x * b->z, a->x * b->y - a->y * b->x);
    }
};

struct Color;
using C = std::shared_ptr<const Color>;
struct Color {
    double r, g, b;
    Color(double r, double g, double b) : r(r), g(g), b(b) {}
    static C make(double r, double g, double b) { return std::make_shared<const Color>(r, g, b); }
    static C scale(double k, const C &v) { return make(k * v->r, k * v->g, k * v->b); }
    static C plus(const C &a, const C &b) { return make(a->r + b->r, a->g + b->g, a->b + b->b); }
    static C times(const C &a, const C &b) { return make(a->r * b->r, a->g * b->g, a->b * b->b); }
};

static const C white = Color::make(1.0, 1.0, 1.0);
static const C grey = Color::make(0.5, 0.5, 0.5);
static const C black = Color::make(0.0, 0.0, 0.0);
static const C background = black;
static const C defaultColor = black;

struct DrawingColor { double r, g, b; };

static DrawingColor toDrawingColor(const C &c) {
    auto legalize = [](double d) { return d > 1 ? 1.0 : d; };
    return {std::floor(legalize(c->r) * 255), std::floor(legalize(c->g) * 255), std::floor(legalize(c->b) * 255)};
}

struct Camera {
    V pos, forward, right, up;
    Camera(V pos, const V &lookAt) : pos(std::move(pos)) {
        V down = Vector::make(0.0, -1.0, 0.0);
        forward = Vector::norm(Vector::minus(lookAt, this->pos));
        right = Vector::times(1.5, Vector::norm(Vector::cross(forward, down)));
        up = Vector::times(1.5, Vector::norm(Vector::cross(forward, right)));
    }
};

struct Ray { V start, dir; };
using R = std::shared_ptr<const Ray>;

struct Thing;
struct Intersection { std::shared_ptr<const Thing> thing; R ray; double dist; };
using I = std::shared_ptr<const Intersection>;

struct Surface {
    std::function<C(const V &)> diffuse;
    std::function<C(const V &)> specular;
    std::function<double(const V &)> reflect;
    double roughness;
};

struct Thing : std::enable_shared_from_this<Thing> {
    std::shared_ptr<const Surface> surface;
    explicit Thing(std::shared_ptr<const Surface> surface) : surface(std::move(surface)) {}
    virtual ~Thing() = default;
    virtual I intersect(const R &ray) const = 0;
    virtual V normal(const V &pos) const = 0;
};

struct Light { V pos; C color; };

struct Scene {
    std::vector<std::shared_ptr<const Thing>> things;
    std::vector<std::shared_ptr<const Light>> lights;
    std::shared_ptr<const Camera> camera;
};

struct Sphere : Thing {
    V center;
    double radius2;
    Sphere(V center, double radius, std::shared_ptr<const Surface> surface)
        : Thing(std::move(surface)), center(std::move(center)), radius2(radius * radius) {}
    V normal(const V &pos) const override { return Vector::norm(Vector::minus(pos, center)); }
    I intersect(const R &ray) const override {
        V eo = Vector::minus(center, ray->start);
        double v = Vector::dot(eo, ray->dir);
        double dist = 0;
        if (v >= 0) {
            double disc = radius2 - (Vector::dot(eo, eo) - v * v);
            if (disc >= 0) dist = v - std::sqrt(disc);
        }
        if (dist == 0) return nullptr;
        return std::make_shared<const Intersection>(Intersection{shared_from_this(), ray, dist});
    }
};

struct Plane : Thing {
    V norm;
    double offset;
    Plane(V norm, double offset, std::shared_ptr<const Surface> surface)
        : Thing(std::move(surface)), norm(std::move(norm)), offset(offset) {}
    V normal(const V &) const override { return norm; }
    I intersect(const R &ray) const override {
        double denom = Vector::dot(norm, ray->dir);
        if (denom > 0) return nullptr;
        double dist = (Vector::dot(norm, ray->start) + offset) / (-denom);
        return std::make_shared<const Intersection>(Intersection{shared_from_this(), ray, dist});
    }
};

static bool odd(const V &pos) { return std::fmod(std::floor(pos->z) + std::floor(pos->x), 2) != 0; }

struct RayTracer {
    int maxDepth = 5;

    I intersections(const R &ray, const Scene &scene) const {
        double closest = std::numeric_limits<double>::infinity();
        I closestInter;
        for (auto &thing : scene.things) {
            I inter = thing->intersect(ray);
            if (inter && inter->dist < closest) {
                closestInter = inter;
                closest = inter->dist;
            }
        }
        return closestInter;
    }

    std::optional<double> testRay(const R &ray, const Scene &scene) const {
        I isect = intersections(ray, scene);
        if (isect) return isect->dist;
        return std::nullopt;
    }

    C traceRay(const R &ray, const Scene &scene, int depth) const {
        I isect = intersections(ray, scene);
        if (!isect) return background;
        return shade(isect, scene, depth);
    }

    C shade(const I &isect, const Scene &scene, int depth) const {
        V d = isect->ray->dir;
        V pos = Vector::plus(Vector::times(isect->dist, d), isect->ray->start);
        V normal = isect->thing->normal(pos);
        V reflectDir = Vector::minus(d, Vector::times(2, Vector::times(Vector::dot(normal, d), normal)));
        C naturalColor = Color::plus(background, getNaturalColor(*isect->thing, pos, normal, reflectDir, scene));
        C reflectedColor = (depth >= maxDepth) ? grey : getReflectionColor(*isect->thing, pos, normal, reflectDir, scene, depth);
        return Color::plus(naturalColor, reflectedColor);
    }

    C getReflectionColor(const Thing &thing, const V &pos, const V &, const V &rd, const Scene &scene, int depth) const {
        return Color::scale(thing.surface->reflect(pos), traceRay(std::make_shared<const Ray>(Ray{pos, rd}), scene, depth + 1));
    }

    C getNaturalColor(const Thing &thing, const V &pos, const V &norm, const V &rd, const Scene &scene) const {
        std::function<C(const C &, const Light &)> addLight = [&](const C &col, const Light &light) -> C {
            V ldis = Vector::minus(light.pos, pos);
            V livec = Vector::norm(ldis);
            auto neatIsect = testRay(std::make_shared<const Ray>(Ray{pos, livec}), scene);
            bool isInShadow = neatIsect ? (*neatIsect <= Vector::mag(ldis)) : false;
            if (isInShadow) return col;
            double illum = Vector::dot(livec, norm);
            C lcolor = (illum > 0) ? Color::scale(illum, light.color) : defaultColor;
            double specular = Vector::dot(livec, Vector::norm(rd));
            C scolor = (specular > 0) ? Color::scale(std::pow(specular, thing.surface->roughness), light.color) : defaultColor;
            return Color::plus(col, Color::plus(Color::times(thing.surface->diffuse(pos), lcolor),
                                                Color::times(thing.surface->specular(pos), scolor)));
        };
        C col = defaultColor;
        for (auto &light : scene.lights) col = addLight(col, *light);
        return col;
    }

    double render(const Scene &scene, double screenWidth, double screenHeight) const {
        auto getPoint = [&](double x, double y, const Camera &camera) {
            auto recenterX = [&](double x) { return (x - (screenWidth / 2.0)) / 2.0 / screenWidth; };
            auto recenterY = [&](double y) { return -(y - (screenHeight / 2.0)) / 2.0 / screenHeight; };
            return Vector::norm(Vector::plus(camera.forward, Vector::plus(Vector::times(recenterX(x), camera.right), Vector::times(recenterY(y), camera.up))));
        };
        double sum = 0;
        for (double y = 0; y < screenHeight; y++) {
            for (double x = 0; x < screenWidth; x++) {
                C color = traceRay(std::make_shared<const Ray>(Ray{scene.camera->pos, getPoint(x, y, *scene.camera)}), scene, 0);
                DrawingColor c = toDrawingColor(color);
                sum += c.r + c.g + c.b;
            }
        }
        return sum;
    }
};

static Scene defaultScene() {
    auto shiny = std::make_shared<const Surface>(Surface{
        [](const V &) { return white; },
        [](const V &) { return grey; },
        [](const V &) { return 0.7; },
        250.0});
    auto checkerboard = std::make_shared<const Surface>(Surface{
        [](const V &pos) { return odd(pos) ? white : black; },
        [](const V &) { return white; },
        [](const V &pos) { return odd(pos) ? 0.1 : 0.7; },
        150.0});
    Scene scene;
    scene.things = {std::make_shared<const Plane>(Vector::make(0.0, 1.0, 0.0), 0.0, checkerboard),
                    std::make_shared<const Sphere>(Vector::make(0.0, 1.0, -0.25), 1.0, shiny),
                    std::make_shared<const Sphere>(Vector::make(-6, 3.0, -5.0), 1.0, shiny),
                    std::make_shared<const Sphere>(Vector::make(-1.0, 0.5, 1.5), 0.5, shiny)};
    auto light = [](V pos, C color) { return std::make_shared<const Light>(Light{std::move(pos), std::move(color)}); };
    scene.lights = {light(Vector::make(-2.0, 2.5, 0.0), Color::make(0.49, 0.07, 0.07)),
                    light(Vector::make(1.5, 2.5, 1.5), Color::make(0.07, 0.07, 0.49)),
                    light(Vector::make(1.5, 2.5, -1.5), Color::make(0.07, 0.49, 0.071)),
                    light(Vector::make(0.0, 3.5, 0.0), Color::make(0.21, 0.21, 0.35))};
    scene.camera = std::make_shared<const Camera>(Vector::make(3.0, 2.0, 4.0), Vector::make(-1.0, 0.5, 0.0));
    return scene;
}

int main(int argc, char **argv) {
    std::puts("start...");
    double size = argc > 1 ? std::atof(argv[1]) : 512;
    RayTracer rayTracer;
    double sum = rayTracer.render(defaultScene(), size, size);
    std::printf("checksum: %.0f\n", sum);
    std::puts("done.");
}
