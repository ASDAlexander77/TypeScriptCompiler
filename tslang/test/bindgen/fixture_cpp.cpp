#include "fixture_cpp.hpp"

namespace fx
{

int32_t Shape::id(int32_t x)
{
    return static_cast<int32_t>(sides * 100 + x);
}

int32_t Shape::area() const
{
    return 0;
}

void Shape::scale(int32_t)
{
}

void Shape::scale(double)
{
}

Handle Shape::handle()
{
    return Handle(nullptr);
}

Pair Shape::pair(int32_t x)
{
    return Pair{x, 0};
}

int32_t Square::area() const
{
    return side * side;
}

void Square::scale(int32_t by)
{
    side *= by;
}

void Square::scale(double by)
{
    side = static_cast<int32_t>(side * by);
}

Handle Square::handle()
{
    return Handle(this);
}

Pair Square::pair(int32_t x)
{
    return Pair{x, side};
}

bool Square::bigger(const Shape &other) const
{
    return area() > other.area();
}

} // namespace fx

// constant-initialized: no constructor runs, so nothing needs the C++ runtime's start-up
static fx::Square squares[2] = {fx::Square(0), fx::Square(0)};
static int next;

extern "C" fx::Square *fx_make_square(int32_t side)
{
    auto *square = &squares[next++ % 2];
    square->side = side;
    return square;
}
