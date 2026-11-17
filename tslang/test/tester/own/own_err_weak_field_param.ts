class Node { v = 1; constructor(@weak public p: Node | null) {} }
function main() { const n = new Node(null); print(n.v); }
