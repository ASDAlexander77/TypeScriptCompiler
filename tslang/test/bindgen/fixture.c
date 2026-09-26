#include "fixture.h"

#include <stdarg.h>
#include <stdlib.h>

struct Counter
{
    int32_t n;
};

int32_t fx_add(int32_t a, int32_t b)
{
    return a + b;
}

double fx_scale(double v, float f)
{
    return v * f;
}

int32_t fx_widen_s8(int8_t v)
{
    return v;
}

int32_t fx_widen_u16(uint16_t v)
{
    return v;
}

int8_t fx_negate_s8(int8_t v)
{
    return (int8_t)-v;
}

bool fx_not(bool b)
{
    return !b;
}

size_t fx_len(const char *s)
{
    size_t n = 0;
    while (s[n])
    {
        n++;
    }

    return n;
}

const char *fx_greet(void)
{
    return "hello from C";
}

void fx_divmod(int32_t a, int32_t b, int32_t *quotient, int32_t *remainder)
{
    *quotient = a / b;
    *remainder = a % b;
}

Counter *fx_counter_new(void)
{
    Counter *c = malloc(sizeof *c);
    c->n = 0;
    return c;
}

void fx_counter_inc(Counter *c)
{
    c->n++;
}

int32_t fx_counter_get(Counter *c)
{
    return c->n;
}

void fx_counter_free(Counter *c)
{
    free(c);
}

int32_t fx_point_sum(const Point *p)
{
    return p->x + p->y;
}

void fx_box_fill(Box *b)
{
    b->tag = -7;
    b->corner.x = 30;
    b->corner.y = -40;
    b->scale = 2.5;
    b->visible = true;
}

int32_t fx_list_sum(Node *head)
{
    int32_t sum = 0;
    for (; head; head = head->next)
    {
        sum += head->value;
    }

    return sum;
}

int32_t fx_color_value(enum Color c)
{
    return (int32_t)c;
}

int32_t fx_apply(int32_t (*fn)(int32_t), int32_t v)
{
    return fn(v);
}

int32_t fx_sum(int32_t count, ...)
{
    va_list args;
    va_start(args, count);
    int32_t sum = 0;
    for (int32_t i = 0; i < count; i++)
    {
        sum += va_arg(args, int32_t);
    }

    va_end(args);
    return sum;
}

void fx_value_clear(union Value *v)
{
    v->i = 0;
}
