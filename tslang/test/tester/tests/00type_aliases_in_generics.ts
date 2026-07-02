type ObjectDescriptor<D, M> = {
    data?: D;
    methods?: M/* & ThisType<D & M>*/; // Type of 'this' in methods is D & M
};

function makeObject<D, M>(desc: ObjectDescriptor<D, M>): D & M {
    let data = desc.data/* || {}*/;
    let methods = desc.methods/* || {}*/;
    return { ...data, ...methods } as D & M;
}

function main() {

    let obj = makeObject({
        data: { x: 0, y: 0 },
        methods: {
            moveBy(dx: number, dy: number) {
                //this.x += dx; // Strongly typed this
                //this.y += dy; // Strongly typed this
            },
        },
    });

    obj.x = 10;
    obj.y = 20;
    obj.moveBy(5, 5);

    print("done.");
}              