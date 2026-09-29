// `import './m'` pointing to a .ts file includes it with its bodies: run on its own, the program
// used to fail with "Symbols not found: [ S..new ]" - the imported class was only declared.
import './imported_class_module'

const s = new S();
assert(s.toString() == "Hi");
print("Hello World", s);

print("done.");
