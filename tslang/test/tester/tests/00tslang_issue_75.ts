// TSLANG 75: a call through a global nothing assigns, given a function by its initializer, is a
// call of that function: `const f = half` (as `const f = Math.fround`) destroys nothing, so the cluster borrowed from a
// parameter's mesh outlives it.
namespace Clusters {
    function half(x: number) {
        return x / 2;
    }
    const f = half;
    function twice(x: number) {
        return x * 2;
    }
    const g = twice;

    export class Cluster {
        positions: number[] = [1, 2, 3, 4];
    }

    export class Mesh {
        clusters: Cluster[] = [new Cluster(), new Cluster()];
    }

    export function centroid(meshes: Mesh[]): number {
        const mesh = meshes[0];
        let sum = 0.0;
        for (const c of mesh.clusters) {
            let y = 0.0;
            for (let v = 0; v < c.positions.length; v++) {
                y = f(y + c.positions[v]);
                y = g(y);
            }
            sum = sum + y / c.positions.length;
        }

        return sum;
    }
}

function main() {
    const meshes = [new Clusters.Mesh()];
    const c = Clusters.centroid(meshes);
    assert(c == 5, "centroid");
    print("done.");
}
