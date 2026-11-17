// own spec 93: an abstract field is a getter and a setter, with no slot for @weak to make weak
class Node { v = 1; }
abstract class A { @weak abstract parent: Node | null; }
function main() { print("done."); }
