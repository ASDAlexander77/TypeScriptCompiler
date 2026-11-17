class Node { v = 1; }
class Holder { @weak static last: Node | null = null; }
function main() { print(Holder.last === null); }
