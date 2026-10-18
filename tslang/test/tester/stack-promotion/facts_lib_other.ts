// Another library defining `sumOf`, which keeps its argument and publishes no fact for it. Linked
// beside facts_lib, or given as an object, facts_lib's `sumOf 0` must not decide anything: the
// linker may take this definition.

let held: number[] = [];

export function sumOf(a: number[]): number {
    held = a;
    return 0;
}
