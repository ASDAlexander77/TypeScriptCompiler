interface Surface {
    roughness: number;
}

module Surfaces {
    export let shiny: Surface = {
        roughness: 250.0
    }
    export let checkerboard: Surface = {
        roughness: 150.0
    }
}

function main() {
    print("Start");

    print(Surfaces.shiny.roughness);
    print(Surfaces.checkerboard.roughness);

    assert(Surfaces.shiny.roughness == 250.0);
    assert(Surfaces.checkerboard.roughness == 150.0);

    print("done.");
}
