// TSLANG 191: a JSON file imported as a module (`import d = require("./x.json")`, `import d from "./x.json"`): its
// value is what it exports. It was loaded as a shared library ("Can't open: Unknown error (0xC1)").
import data = require("./json_module_data.json");
import same from "./json_module_data.json";

function main() {
    assert(data.name == "tslang", "a string field");
    assert(data.version == 3, "a number field");
    assert(data.tags.length == 2 && data.tags[1] == "bb", "an array field");
    assert(data.nested.flag, "a nested object");
    assert(same.version == 3, "the default import");

    print("done.");
}
