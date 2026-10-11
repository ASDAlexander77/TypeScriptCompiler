// A C++ object whose methods a @cppvtable class calls (abi_msvc.ts, abi_itanium.ts): built by the
// host's C++ compiler, so its vtable and the way its methods return records are that compiler's
// ABI - MSVC's, or the Itanium one of g++ and clang - not a vtable made by hand in tslang.

#include <cstdint>

#ifdef _WIN32
#define ABI_EXPORT extern "C" __declspec(dllexport)
#else
#define ABI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// non-trivial, as nvrhi::RefCountPtr<T>: the Itanium ABI returns it through a pointer before `this`
struct Ptr
{
    void *p;
    Ptr(void *v) : p(v) {}
    Ptr(const Ptr &other) : p(other.p) {}
    ~Ptr() {}
};

// trivial, 8 bytes, as nvrhi::Object: RAX
struct Obj
{
    void *p;
};

// trivial, 16 bytes: RAX:RDX
struct Two
{
    uint64_t a;
    void *b;
};

static char storage[64];

// MSVC returns every one of these records through a pointer after `this`
struct Methods
{
    virtual ~Methods() {}
    virtual Ptr make(int x) { return Ptr(&storage[x]); }
    virtual Obj obj(int x) { return Obj{&storage[x + 1]}; }
    virtual Two two(int x) { return Two{0x1234567890ull + x, &storage[x + 2]}; }
    virtual int plain(int a, int b) { return a * b + value; }
    int value = 7;
};

ABI_EXPORT Methods *abi_object()
{
    static Methods methods;
    return &methods;
}

// whether p is &storage[index]
ABI_EXPORT int abi_is_storage(void *p, int index)
{
    return p == &storage[index] ? 1 : 0;
}
