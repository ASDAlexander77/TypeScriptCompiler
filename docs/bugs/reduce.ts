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
    
    return scene.lights.reduce(addLight, Color.defaultColor);
}

getNaturalColor({ lights: [] }, Color.defaultColor);
