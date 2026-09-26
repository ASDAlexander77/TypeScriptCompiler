// @strict-null false
/// <reference path="fixture.ts" />

// Calls everything in fixture.h through the bindings tsbindgen generated from it (fixture.ts).
// run-bindgen-test.cmake compares what this prints with expected.txt.

function twice(x: s32): s32 {
    return x * 2;
}

function main() {
    print("macros", FIXTURE_ANSWER, FIXTURE_RATIO, FIXTURE_NAME);
    print("add", fx_add(2, 3));
    print("scale", fx_scale(2.0, 1.5));
    print("narrow", fx_widen_s8(-5), fx_widen_u16(65535), fx_negate_s8(5), fx_not(true), fx_not(false));
    print("len", fx_len("hello"));
    print("greet", fx_greet());

    let quotient: s32 = 0;
    let remainder: s32 = 0;
    fx_divmod(17, 5, ReferenceOf(quotient), ReferenceOf(remainder));
    print("divmod", quotient, remainder);

    const counter = fx_counter_new();
    fx_counter_inc(counter);
    fx_counter_inc(counter);
    print("counter", fx_counter_get(counter));
    fx_counter_free(counter);

    let point: Point = [3, 4];
    print("point", fx_point_sum(ReferenceOf(point)));

    let box: Box = [0, [0, 0], 0.0, false];
    fx_box_fill(ReferenceOf(box));
    print("box", box[0], box[1][0], box[1][1], box[2], box[3]);

    let second: Node = [5, null];
    let first: Node = [3, ReferenceOf(second) as Opaque];
    print("list", fx_list_sum(ReferenceOf(first)));

    print("color", fx_color_value(Color.BLUE));
    print("apply", fx_apply(twice as Opaque, 21));
    print("sum", fx_sum(3, 1, 2, 3));
}
