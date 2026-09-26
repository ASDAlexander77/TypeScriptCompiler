// One of each thing tsbindgen v1 binds, for test/bindgen/run-bindgen-test.cmake. bindgen_test.ts
// calls every function through the generated bindings; expected.txt is what it must print.

#ifndef FIXTURE_H
#define FIXTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FIXTURE_API __declspec(dllexport)
#else
#define FIXTURE_API __attribute__((visibility("default")))
#endif

#define FIXTURE_ANSWER 42
#define FIXTURE_RATIO 1.5
#define FIXTURE_NAME "fixture"

typedef struct Counter Counter; // opaque handle

typedef struct Point
{
    int32_t x;
    int32_t y;
} Point;

typedef struct Box
{
    int8_t tag;
    Point corner; // a struct inside a struct
    double scale;
    bool visible;
} Box;

typedef struct Node // points to itself: `next` becomes Opaque
{
    int32_t value;
    struct Node *next;
} Node;

enum Color
{
    RED,
    GREEN = 5,
    BLUE
};

union Value // skipped, but a pointer to it is still Opaque
{
    int32_t i;
    float f;
};

// scalars
FIXTURE_API int32_t fx_add(int32_t a, int32_t b);
FIXTURE_API double fx_scale(double v, float f);
FIXTURE_API int32_t fx_widen_s8(int8_t v);
FIXTURE_API int32_t fx_widen_u16(uint16_t v);
FIXTURE_API int8_t fx_negate_s8(int8_t v);
FIXTURE_API bool fx_not(bool b);

// strings in and out
FIXTURE_API size_t fx_len(const char *s);
FIXTURE_API const char *fx_greet(void);

// out-parameters
FIXTURE_API void fx_divmod(int32_t a, int32_t b, int32_t *quotient, int32_t *remainder);

// an opaque handle
FIXTURE_API Counter *fx_counter_new(void);
FIXTURE_API void fx_counter_inc(Counter *c);
FIXTURE_API int32_t fx_counter_get(Counter *c);
FIXTURE_API void fx_counter_free(Counter *c);

// structs by pointer, read and written by C
FIXTURE_API int32_t fx_point_sum(const Point *p);
FIXTURE_API void fx_box_fill(Box *b);
FIXTURE_API int32_t fx_list_sum(Node *head);

// an enum
FIXTURE_API int32_t fx_color_value(enum Color c);

// a callback
FIXTURE_API int32_t fx_apply(int32_t (*fn)(int32_t), int32_t v);

// varargs
FIXTURE_API int32_t fx_sum(int32_t count, ...);

// skipped declarations
FIXTURE_API void fx_value_clear(union Value *v);
static inline int32_t fx_twice_inline(int32_t x)
{
    return x * 2;
}

#endif // FIXTURE_H
