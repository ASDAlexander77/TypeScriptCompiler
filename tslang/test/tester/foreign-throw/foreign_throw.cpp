// C++ exceptions that tslang never threw, for foreign-throw/catch.ts
struct Foreign
{
    int x;
};

static Foreign foreign{42};

extern "C" void foreign_throw_cstr()
{
    throw "boom";
}

extern "C" void foreign_throw_ptr()
{
    throw &foreign;
}

extern "C" void foreign_throw_int()
{
    throw 7;
}
