
namespace __Array {
    function is<V extends T, T>(t: T): t is V
    {
        return true;
    }

    function reduce2<T, V = T>(this: T[], func: (v: V, t: T) => V, initial?: V) {
        if (initial == undefined)
        {
            if (this.length <= 0) {
                return undefined;
            }

            if (is<V, T>(this[0]))
            {
                let result = <V>this[0];
                for (let i = 1; i in this; i++) result = func(result, this[i]);
                return result;
            }

            return undefined;
        }
        else 
        {
            let result = initial;
            for (const v of this) result = func(result, v);
            return result;
        }
    }
}

class Color {
    constructor(public r: number,
                public g: number,
                public b: number) {
    }
    static black = new Color(0.0, 0.0, 0.0);
    static defaultColor = Color.black;
}

interface Light {
    color: Color;
}

interface Scene {
    lights: Light[];
}

function getNaturalColor(scene: Scene, background: Color) {
    const addLight = (col, light) => {
        var isInShadow = true;
        if (isInShadow) {
            return col;
        } else {
            return background;
        }
    }
    
    return scene.lights.reduce2(addLight, Color.defaultColor);
}

const resColor = getNaturalColor({ lights: [ { color: Color.defaultColor } ] }, Color.defaultColor);

assert(resColor.r == Color.defaultColor.r
    && resColor.g == Color.defaultColor.g
    && resColor.b == Color.defaultColor.b
);

print("done.");