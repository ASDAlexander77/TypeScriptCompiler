// A C++ object whose methods a @cpp class calls (abi_msvc.ts, abi_itanium.ts): built by the
// host's C++ compiler, so its vtable and the way its methods return records are that compiler's
// ABI - MSVC's, or the Itanium one of g++ and clang - not a vtable made by hand in tslang.

#include <cstdint>
#include <new>

#ifdef _WIN32
#define ABI_EXPORT extern "C" __declspec(dllexport)
#define ABI_MEMBER_EXPORT __declspec(dllexport)
#else
#define ABI_EXPORT extern "C" __attribute__((visibility("default")))
#define ABI_MEMBER_EXPORT __attribute__((visibility("default")))
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

// records passed by value (abi_msvc.ts, abi_itanium.ts' byValue, pair), as each ABI classifies them:
struct S4 { uint16_t a; uint8_t b, c; };    // MSVC: a register, of 4 bytes; Itanium: one register
struct S8 { uint32_t a, b; };               // MSVC: a register, of 8 bytes
struct S12 { uint32_t a, b, c; };           // MSVC: a pointer to a copy; Itanium: two registers
struct S16 { uint64_t a; uint32_t b, c; };  // SysV: two registers, or the stack if two don't remain
struct S24 { uint64_t a, b, c; };           // MSVC, AAPCS64: a pointer to a copy; SysV: the stack
// and returned: MSVC through a pointer after `this`; Itanium in RAX / X0, RAX:RDX / X0:X1
struct Flags { bool a, b, c; };
struct R12 { uint32_t a, b, c; };

// MSVC returns every one of these records through a pointer after `this`
struct Methods
{
    virtual ~Methods() {}
    virtual Ptr make(int x) { return Ptr(&storage[x]); }
    virtual Obj obj(int x) { return Obj{&storage[x + 1]}; }
    virtual Two two(int x) { return Two{0x1234567890ull + x, &storage[x + 2]}; }
    virtual int plain(int a, int b) { return a * b + value; }
    // narrow arguments: clang's code takes them as extended to 32 bits by the caller
    virtual int narrow(bool b, uint8_t u, int8_t s) { return (b ? 1000 : 0) + u * 10 + s; }
    // overloads: MSVC groups them in its vtable, in reverse declaration order; the Itanium ABI
    // keeps the declaration order
    virtual int over(int x) { return 1000 + x; }
    virtual int over(double x) { return 2000 + (int)(x * 10); }
    virtual int over(int a, int b) { return 3000 + a * b; }
    // every field weighed by a prime of its own, so a field read from the wrong place shows; the
    // copies changed, which the caller's records mustn't see
    virtual uint64_t byValue(int k, S4 a, S8 b, S12 c, S16 d, S24 e, int tail)
    {
        uint64_t sum = k * 3ull + a.a * 5ull + a.b * 7ull + a.c * 11ull + b.a * 13ull + b.b * 17ull + c.a * 19ull + c.b * 23ull
            + c.c * 29ull + d.a * 31ull + d.b * 37ull + d.c * 41ull + e.a * 43ull + e.b * 47ull + e.c * 53ull + tail * 59ull;
        c.a = 0;
        e.a = 0;
        return sum;
    }
    // a 16-byte record when two registers remain (SysV)
    virtual uint64_t pair(S16 d, int tail) { return d.a * 31ull + d.b * 37ull + d.c * 41ull + tail * 59ull; }
    virtual Flags flags(int x) { return Flags{(x & 1) != 0, (x & 2) != 0, (x & 4) != 0}; }
    virtual R12 r12(int x) { return R12{uint32_t(x), uint32_t(x * 2), uint32_t(x * 3)}; }
    // non-virtual: called by its symbol, ?scaled@Methods@@QEAAHH@Z or _ZN7Methods6scaledEi
    ABI_MEMBER_EXPORT int scaled(int x);
    // and its overload, ?scaled@Methods@@QEAAHN@Z or _ZN7Methods6scaledEd
    ABI_MEMBER_EXPORT int scaled(double x);
    int value = 7;
};

int Methods::scaled(int x)
{
    return x * 100 + value;
}

int Methods::scaled(double x)
{
    return (int)(x * 1000) + value;
}

// implemented in tslang (abi_msvc.ts, abi_itanium.ts' Counter): C++ calls its methods through the
// vtable tslang made, as NVRHI calls an application's nvrhi::IMessageCallback
class Callback
{
protected:
    virtual ~Callback() {}

public:
    virtual void message(uint8_t severity, const char *text) = 0;
    virtual int count() = 0;
    virtual bool isEven(int x) = 0;
    // overloads implemented in tslang
    virtual int pick(int x) = 0;
    virtual int pick(double x) = 0;
};

// two messages, then the count
ABI_EXPORT int abi_call_callback(Callback *callback)
{
    callback->message(2, "two");
    callback->message(3, "three");
    return callback->count();
}

// C++ picks the overload: pick(int), then pick(double)
ABI_EXPORT int abi_callback_pick(Callback *callback)
{
    return callback->pick(3) * 10000 + callback->pick(0.25);
}

// a bool result, as C++ reads it
ABI_EXPORT int abi_callback_is_even(Callback *callback, int x)
{
    return callback->isEven(x) ? 1 : 0;
}

ABI_EXPORT Methods *abi_object()
{
    static Methods methods;
    return &methods;
}

// on the C++ heap, right after a word of C++'s own: under rc and own, a count tslang kept for it
// would be in that word. A retain and its release leave it as it was, so it is checked while
// tslang holds the copies (abi_guard_intact), and then C++ deletes the object.
static const uint64_t GUARD = 0x5a5a5a5a5a5a5a5aull;

ABI_EXPORT Methods *abi_heap_object()
{
    auto block = static_cast<char *>(::operator new(sizeof(uint64_t) + sizeof(Methods)));
    *reinterpret_cast<uint64_t *>(block) = GUARD;
    return new (block + sizeof(uint64_t)) Methods();
}

ABI_EXPORT int abi_guard_intact(Methods *methods)
{
    return reinterpret_cast<uint64_t *>(methods)[-1] == GUARD ? 1 : 0;
}

ABI_EXPORT void abi_delete_object(Methods *methods)
{
    methods->~Methods();
    ::operator delete(reinterpret_cast<char *>(methods) - sizeof(uint64_t));
}

// whether p is &storage[index]
ABI_EXPORT int abi_is_storage(void *p, int index)
{
    return p == &storage[index] ? 1 : 0;
}
