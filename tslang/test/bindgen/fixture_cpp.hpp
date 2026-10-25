// A C++ library for tsbindgen's @cpp classes (run-bindgen-test.cmake, MODE=cpp): its polymorphic
// classes become @cpp classes, called through their vtables and by their C++ declarations.
//
// No C++ runtime is needed to link it: no destructors, pure virtual methods, RTTI or exceptions, and
// its objects are constant-initialized (constexpr constructors), so a tslang program links its object
// alone.

#ifndef FIXTURE_CPP_HPP
#define FIXTURE_CPP_HPP

#include <stdint.h>

#ifdef _WIN32
#define FX_API __declspec(dllexport)
#else
#define FX_API __attribute__((visibility("default")))
#endif

namespace fx
{

// returned by value: a copy constructor and a destructor make it non-trivial (@cppnontrivial)
struct Handle
{
    void *p;
    Handle(void *value) : p(value)
    {
    }
    Handle(const Handle &other) : p(other.p)
    {
    }
    ~Handle()
    {
    }
};

// returned by value: trivial, two 8-byte fields
struct Pair
{
    int64_t a;
    int64_t b;
};

// a struct in a namespace, with a constructor: still a tuple, passed by reference or by a pointer
// that may be null
struct Extent
{
    int32_t width;
    int32_t height;
    constexpr Extent(int32_t width, int32_t height) : width(width), height(height)
    {
    }
};

class Shape;

// a container shaped as nvrhi::static_vector: a private array base, then a count. A field holding
// one has tsbindgen declare the specialization (SmallVec_Shape_3).
template <typename T, int N> struct Storage
{
    T items[N];
};

template <typename T, int N> struct SmallVec : private Storage<T, N>
{
    int64_t count;
    T at(int64_t index) const
    {
        return this->items[index];
    }
};

// with a union of a buffer and a pointer, as an std::string's, and a run of bitfields: raw words
struct Batch
{
    SmallVec<Shape *, 3> shapes;
    union
    {
        char buf[16];
        char *ptr;
    } name;
    uint32_t flags : 3;
    uint32_t mode : 5;
    int32_t total;
};

class FX_API Shape
{
  public:
    constexpr Shape(int64_t sides) : sides(sides)
    {
    }

    // defined (not pure): a pure virtual's slot would point at the C++ runtime's __cxa_pure_virtual
    virtual int32_t area() const;
    virtual void scale(int32_t by);
    virtual void scale(double by); // an overload: scale too, with its own @slot
    virtual Handle handle();
    virtual Pair pair(int32_t x);
    virtual int32_t fits(const Extent &extent, const Extent *limit);
    virtual int32_t sum(const Batch &batch);

    // non-virtual: called by its declaration, @cppname("int32_t fx::Shape::id(int32_t)")
    int32_t id(int32_t x);

    int64_t sides; // 8 bytes: no tail padding, which the Itanium ABI would put `side` in
};

class FX_API Square : public Shape
{
  public:
    constexpr Square(int32_t side) : Shape(4), side(side)
    {
    }

    int32_t area() const override;
    void scale(int32_t by) override;
    void scale(double by) override;
    Handle handle() override;
    Pair pair(int32_t x) override;

    virtual bool bigger(const Shape &other) const;

    int32_t side;
};

} // namespace fx

extern "C" FX_API fx::Square *fx_make_square(int32_t side);

#endif // FIXTURE_CPP_HPP
