// Idiomatic C++ port of rtbench.ts: Vector/Color are plain value structs, things are
// std::unique_ptr<Thing> with virtual methods, surfaces hold function pointers.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

struct Vector {
    double x, y, z;
    static Vector times(double k, Vector v) { return {k * v.x, k * v.y, k * v.z}; }
    static Vector minus(Vector a, Vector b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    static Vector plus(Vector a, Vector b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    static double dot(Vector a, Vector b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    static double mag(Vector v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    static Vector norm(Vector v) {
        double m = mag(v);
        double div = (m == 0) ? std::numeric_limits<double>::infinity() : 1.0 / m;
        return times(div, v);
    }
    static Vector cross(Vector a, Vector b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
};

struct Color {
    double r, g, b;
    static Color scale(double k, Color v) { return {k * v.r, k * v.g, k * v.b}; }
    static Color plus(Color a, Color b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
    static Color times(Color a, Color b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
};

constexpr Color white{1.0, 1.0, 1.0};
constexpr Color grey{0.5, 0.5, 0.5};
constexpr Color black{0.0, 0.0, 0.0};
constexpr Color background = black;
constexpr Color defaultColor = black;

struct DrawingColor { double r, g, b; };

static DrawingColor toDrawingColor(Color c) {
    auto legalize = [](double d) { return d > 1 ? 1.0 : d; };
    return {std::floor(legalize(c.r) * 255), std::floor(legalize(c.g) * 255), std::floor(legalize(c.b) * 255)};
}

struct Camera {
    Vector pos, forward, right, up;
    Camera(Vector pos, Vector lookAt) : pos(pos) {
        Vector down{0.0, -1.0, 0.0};
        forward = Vector::norm(Vector::minus(lookAt, pos));
        right = Vector::times(1.5, Vector::norm(Vector::cross(forward, down)));
        up = Vector::times(1.5, Vector::norm(Vector::cross(forward, right)));
    }
};

struct Ray { Vector start, dir; };

struct Thing;
struct Intersection { const Thing *thing; Ray ray; double dist; };

struct Surface {
    Color (*diffuse)(Vector);
    Color (*specular)(Vector);
    double (*reflect)(Vector);
    double roughness;
};

struct Thing {
    virtual ~Thing() = default;
    virtual std::optional<Intersection> intersect(const Ray &ray) const = 0;
    virtual Vector normal(Vector pos) const = 0;
    const Surface *surface;
    explicit Thing(const Surface *surface) : surface(surface) {}
};

struct Light { Vector pos; Color color; };

struct Scene {
    std::vector<std::unique_ptr<Thing>> things;
    std::vector<Light> lights;
    Camera camera;
};

struct Sphere : Thing {
    Vector center;
    double radius2;
    Sphere(Vector center, double radius, const Surface *surface) : Thing(surface), center(center), radius2(radius * radius) {}
    Vector normal(Vector pos) const override { return Vector::norm(Vector::minus(pos, center)); }
    std::optional<Intersection> intersect(const Ray &ray) const override {
        Vector eo = Vector::minus(center, ray.start);
        double v = Vector::dot(eo, ray.dir);
        double dist = 0;
        if (v >= 0) {
            double disc = radius2 - (Vector::dot(eo, eo) - v * v);
            if (disc >= 0) dist = v - std::sqrt(disc);
        }
        if (dist == 0) return std::nullopt;
        return Intersection{this, ray, dist};
    }
};

struct Plane : Thing {
    Vector norm;
    double offset;
    Plane(Vector norm, double offset, const Surface *surface) : Thing(surface), norm(norm), offset(offset) {}
    Vector normal(Vector) const override { return norm; }
    std::optional<Intersection> intersect(const Ray &ray) const override {
        double denom = Vector::dot(norm, ray.dir);
        if (denom > 0) return std::nullopt;
        double dist = (Vector::dot(norm, ray.start) + offset) / (-denom);
        return Intersection{this, ray, dist};
    }
};

static bool odd(Vector pos) { return std::fmod(std::floor(pos.z) + std::floor(pos.x), 2) != 0; }

const Surface shiny{
    [](Vector) { return white; },
    [](Vector) { return grey; },
    [](Vector) { return 0.7; },
    250.0};

const Surface checkerboard{
    [](Vector pos) { return odd(pos) ? white : black; },
    [](Vector) { return white; },
    [](Vector pos) { return odd(pos) ? 0.1 : 0.7; },
    150.0};

struct RayTracer {
    int maxDepth = 5;

    std::optional<Intersection> intersections(const Ray &ray, const Scene &scene) const {
        double closest = std::numeric_limits<double>::infinity();
        std::optional<Intersection> closestInter;
        for (auto &thing : scene.things) {
            auto inter = thing->intersect(ray);
            if (inter && inter->dist < closest) {
                closestInter = inter;
                closest = inter->dist;
            }
        }
        return closestInter;
    }

    std::optional<double> testRay(const Ray &ray, const Scene &scene) const {
        auto isect = intersections(ray, scene);
        if (isect) return isect->dist;
        return std::nullopt;
    }

    Color traceRay(const Ray &ray, const Scene &scene, int depth) const {
        auto isect = intersections(ray, scene);
        if (!isect) return background;
        return shade(*isect, scene, depth);
    }

    Color shade(const Intersection &isect, const Scene &scene, int depth) const {
        Vector d = isect.ray.dir;
        Vector pos = Vector::plus(Vector::times(isect.dist, d), isect.ray.start);
        Vector normal = isect.thing->normal(pos);
        Vector reflectDir = Vector::minus(d, Vector::times(2, Vector::times(Vector::dot(normal, d), normal)));
        Color naturalColor = Color::plus(background, getNaturalColor(isect.thing, pos, normal, reflectDir, scene));
        Color reflectedColor = (depth >= maxDepth) ? grey : getReflectionColor(isect.thing, pos, normal, reflectDir, scene, depth);
        return Color::plus(naturalColor, reflectedColor);
    }

    Color getReflectionColor(const Thing *thing, Vector pos, Vector, Vector rd, const Scene &scene, int depth) const {
        return Color::scale(thing->surface->reflect(pos), traceRay({pos, rd}, scene, depth + 1));
    }

    Color getNaturalColor(const Thing *thing, Vector pos, Vector norm, Vector rd, const Scene &scene) const {
        auto addLight = [&](Color col, const Light &light) {
            Vector ldis = Vector::minus(light.pos, pos);
            Vector livec = Vector::norm(ldis);
            auto neatIsect = testRay({pos, livec}, scene);
            bool isInShadow = neatIsect ? (*neatIsect <= Vector::mag(ldis)) : false;
            if (isInShadow) return col;
            double illum = Vector::dot(livec, norm);
            Color lcolor = (illum > 0) ? Color::scale(illum, light.color) : defaultColor;
            double specular = Vector::dot(livec, Vector::norm(rd));
            Color scolor = (specular > 0) ? Color::scale(std::pow(specular, thing->surface->roughness), light.color) : defaultColor;
            return Color::plus(col, Color::plus(Color::times(thing->surface->diffuse(pos), lcolor),
                                                Color::times(thing->surface->specular(pos), scolor)));
        };
        Color col = defaultColor;
        for (auto &light : scene.lights) col = addLight(col, light);
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
                Color color = traceRay({scene.camera.pos, getPoint(x, y, scene.camera)}, scene, 0);
                DrawingColor c = toDrawingColor(color);
                sum += c.r + c.g + c.b;
            }
        }
        return sum;
    }
};

static Scene defaultScene() {
    Scene scene{{}, {}, Camera({3.0, 2.0, 4.0}, {-1.0, 0.5, 0.0})};
    scene.things.push_back(std::make_unique<Plane>(Vector{0.0, 1.0, 0.0}, 0.0, &checkerboard));
    scene.things.push_back(std::make_unique<Sphere>(Vector{0.0, 1.0, -0.25}, 1.0, &shiny));
    scene.things.push_back(std::make_unique<Sphere>(Vector{-6, 3.0, -5.0}, 1.0, &shiny));
    scene.things.push_back(std::make_unique<Sphere>(Vector{-1.0, 0.5, 1.5}, 0.5, &shiny));
    scene.lights = {{{-2.0, 2.5, 0.0}, {0.49, 0.07, 0.07}},
                    {{1.5, 2.5, 1.5}, {0.07, 0.07, 0.49}},
                    {{1.5, 2.5, -1.5}, {0.07, 0.49, 0.071}},
                    {{0.0, 3.5, 0.0}, {0.21, 0.21, 0.35}}};
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
