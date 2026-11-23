enum Hello1 { World1, World2 }
class F { Hello2 = Hello1; }
function main() { const f = new F(); print(f.Hello2.World2); }
