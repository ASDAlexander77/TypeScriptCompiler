// A class of another module of the same library extends Base, so this module's declarations
// carry Base as well; the importer read Base twice ("redefinition of symbol named 'Base..new'").
import { Base } from "./base_module";

export class Derived extends Base {
    name = "derived";
}

// read by a key known at run time in its own module: it has a keys adapter, and an importer holding one as a
// Base reads it by it (Derived, only exported, has none: it is read by Base's keys)
export class Tagged extends Base {
    tag = "tagged";
}

export function tagOf(t: Tagged, k: string): any {
    return t[k];
}
