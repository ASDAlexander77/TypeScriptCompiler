// TSLANG 200: a destructuring `var` of the module declaring some names again and some for the first time
// (`var { n: nb, p: pb } = r; var { n: nc, p: pb } = s;`): "redefinition of symbol named 'pb'". The names declared
// before are assigned the values the pattern gives them.
interface Robot {
    name: string;
    skills: { primary: string; secondary: string };
}

var robotB: Robot = { name: "trimmer", skills: { primary: "trimming", secondary: "edging" } };

var { name: nameB, skills: { primary: primaryB, secondary: secondaryB } } = robotB;
var { name: nameC, skills: { primary: primaryB, secondary: secondaryB } } = { name: "Edger", skills: { primary: "edging", secondary: "branch" } };

var { n: n1, p: shared } = { n: 1, p: "first" };
var { n: n2, p: shared } = { n: 2, p: "second" };
var [x1, shortName] = ["a", "b"];
var [x2, shortName] = ["c", "d"];

function main() {
    assert(nameB == "trimmer" && nameC == "Edger", "the names declared once");
    assert(primaryB == "edging" && secondaryB == "branch", "the names declared again, nested");
    assert(n1 == 1 && n2 == 2 && shared == "second", "a flat pattern");
    assert(x1 == "a" && x2 == "c" && shortName == "d", "an array pattern");
    print("done.");
}
