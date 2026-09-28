// A class of another module of the same library extends Base, so this module's declarations
// carry Base as well; the importer read Base twice ("redefinition of symbol named 'Base..new'").
import { Base } from "./base_module";

export class Derived extends Base {
    name = "derived";
}
