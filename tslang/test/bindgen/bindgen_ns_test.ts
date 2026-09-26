// @strict-null false
/// <reference path="fixture_ns.ts" />

// bindgen_test.ts again, against `tsbindgen fixture.h --namespace Fx --strip-prefix fx_`: every
// function is reached through a TS name that is not its C symbol, which @linkname binds.

function twice(x: s32): s32 {
    return x * 2;
}

function main() {
    print("macros", Fx.FIXTURE_ANSWER, Fx.FIXTURE_RATIO, Fx.FIXTURE_NAME);
    print("add", Fx.add(2, 3));
    print("scale", Fx.scale(2.0, 1.5));
    print("narrow", Fx.widen_s8(-5), Fx.widen_u16(65535), Fx.negate_s8(5), Fx.not(true), Fx.not(false));
    print("len", Fx.len("hello"));
    print("greet", Fx.greet());

    let quotient: s32 = 0;
    let remainder: s32 = 0;
    Fx.divmod(17, 5, ReferenceOf(quotient), ReferenceOf(remainder));
    print("divmod", quotient, remainder);

    const counter = Fx.counter_new();
    Fx.counter_inc(counter);
    Fx.counter_inc(counter);
    print("counter", Fx.counter_get(counter));
    Fx.counter_free(counter);

    let point: Fx.Point = [3, 4];
    print("point", Fx.point_sum(ReferenceOf(point)));

    let box: Fx.Box = [0, [0, 0], 0.0, false];
    Fx.box_fill(ReferenceOf(box));
    print("box", box[0], box[1][0], box[1][1], box[2], box[3]);

    let second: Fx.Node = [5, null];
    let first: Fx.Node = [3, ReferenceOf(second) as Opaque];
    print("list", Fx.list_sum(ReferenceOf(first)));

    print("color", Fx.color_value(Fx.Color.BLUE));
    print("apply", Fx.apply(twice as Opaque, 21));
    print("sum", Fx.sum(3, 1, 2, 3));
}
