import './export_class_interface'

const p = new A.Point2d(1, 2);
const v = 
    p.fromOrigin({
        x:120, 
        y:3, 
        fromOrigin(p: A.Point) { return 1.0; }   
    });

print(v);    

assert(v == 126.0);

print("done.");

