import './export_class_interface'

const p = new A.Point2d(10, 20);
const v = 
    p.fromOrigin({
        x:100, 
        y:200, 
        fromOrigin(p: A.Point) { return p.x * p.y + this.x + this.y; }   
    });

print(v);    

assert(v == 830.0);

print("done.");

