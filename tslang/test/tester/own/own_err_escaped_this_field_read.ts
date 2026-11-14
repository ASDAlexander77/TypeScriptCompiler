// -mm=own rejects: the constructor gives `this` to `registry.add`, so the call in the loop may reach the
// object and overwrite `name`; the field cannot stand in for the parameter, which it took (TSLANG 80).
@final
class Registry {
    models: Model[] = [];

    add(model: Model) {
        this.models.push(model);
    }

    rename() {
        for (const model of this.models) {
            model.name = "renamed";
        }
    }
}

@final
class Model {
    constructor(public name: string, registry: Registry, n: int) {
        registry.add(this);
        for (let i = 0; i < n; i++) {
            registry.rename();
            print(`model ${name}`);
        }
    }
}

function main() {
    const registry = new Registry();
    new Model("m", registry, 2);
}
