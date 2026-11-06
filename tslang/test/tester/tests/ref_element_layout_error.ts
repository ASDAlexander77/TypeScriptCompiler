// TSLANG_ISSUES 26: a reference to a number (a double) is no `Ref<f32>`: the function would read a
// `double*` as a `float*`.
declare function Fill(enable: int, dst: Ref<f32>): int;

class A {
    sensed: number[] = [0.0, 0.0, 0.0];

    run() {
        Fill(1, Ref(this.sensed[0]));
        print(this.sensed[0]);
    }
}

function main() {
    const a = new A();
    a.run();
}
