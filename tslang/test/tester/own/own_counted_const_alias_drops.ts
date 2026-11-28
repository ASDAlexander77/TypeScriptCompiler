// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own rejects: `reset`, called through the constant `clear`, empties the mesh's clusters, so
// the cluster borrowed from them is used after it may be released (TSLANG 75 resolves the call to
// `reset`, whose facts say so).
class Cluster {
    positions: number[] = [1, 2, 3];
}

class Mesh {
    clusters: Cluster[] = [new Cluster()];
}

function reset(mesh: Mesh) {
    mesh.clusters = [];
}

const clear = reset;

function sum(meshes: Mesh[]) {
    const mesh = meshes[0];
    const c = mesh.clusters[0];
    clear(mesh);
    print(c.positions.length);
}

function main() {
    sum([new Mesh()]);
}
