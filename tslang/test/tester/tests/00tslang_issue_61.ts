// TSLANG_ISSUES 61 (-mm=own), from native_ts_graphics' examples (nri/ray_tracing_reflection_nri.ts, ptlas.ts,
// nri/async_compute_bloom_nri.ts): module and namespace arrays nothing assigns, read into an array literal
// (`[CUBE_VERTICES, PLANE_VERTICES, CUBE_VERTICES]`), returned (`return GREEN;`) or held across calls
// (`const p = CAMERA_POSITION;`). An array global written once, by its initializer, is immortal, as a
// class's object global is (own spec 25): a read of it owns nothing and may go anywhere.

class Mat {
    r: number;
    constructor(r: number) {
        this.r = r;
    }
}

const CUBE = [1.0, 2.0, 3.0];
const PLANE = [4.0, 5.0];
const MATERIALS = [new Mat(0.5), new Mat(0.25)];

namespace Palette {
    const GREEN = [0.0, 1.0, 0.0];
    const ORANGE = [1.0, 0.5, 0.0];

    export function color(hot: boolean): number[] {
        if (hot) {
            return ORANGE;
        }

        return GREEN;
    }
}

function touch(n: int): int {
    return n + 1;
}

function lists(): number {
    const meshes = [CUBE, PLANE, CUBE];
    let total = 0.0;
    for (const mesh of meshes) {
        total += mesh[0];
    }

    return total;
}

function held(): number {
    const p = CUBE;
    touch(1);
    return p[2] + MATERIALS[1].r;
}

class Model {
    vertices: number[];
    materials: Mat[];
    constructor() {
        this.vertices = PLANE;
        this.materials = MATERIALS;
    }
}

function main() {
    assert(lists() == 6.0);
    assert(Palette.color(true)[1] == 0.5 && Palette.color(false)[1] == 1.0);
    assert(held() == 3.25);
    for (let i = 0; i < 3; i++) {
        const m = new Model();
        assert(m.vertices[1] == 5.0 && m.materials[0].r == 0.5);
    }

    assert(CUBE.length == 3 && MATERIALS.length == 2);
    print("done.");
}
