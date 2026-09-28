import { Node } from "./cycle_class_node";

export class TextNode extends Node {
    data: string;

    constructor(d: string) {
        super();
        this.data = d;
    }
}
