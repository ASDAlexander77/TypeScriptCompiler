// TSLANG 74 (own-facts-objects.cmake): a value of `cond ? null : f()`, f another module's, pushed where
// it is not null and then used, reads the array's element - the merge is the function's own, as `f()`
// alone is, though the signature pass marks the imported call's result fresh only after it looks.
import { Texture, Device } from "./own_facts_tex_lib";

class Materials {
    textures: Texture[] = [];
    views: number[] = [];

    add(device: Device, names: string[]): boolean {
        let ok = true;
        for (let m = 0; m < names.length; m++) {
            const name = names[m];
            const texture = name == "" ? null : device.load(name);
            if (!texture) {
                ok = false;
            } else {
                this.textures.push(texture);
                this.views.push(texture.view(texture.format()));
            }
        }

        return ok;
    }
}

function main() {
    const materials = new Materials();
    const ok = materials.add(new Device(), ["abc", "", "x", "wxyz"]);
    print(ok, materials.textures.length, materials.views[0], materials.views[1]);
    print("done.");
}
