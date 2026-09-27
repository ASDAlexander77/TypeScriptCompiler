// cycle_class_node and cycle_class_text import each other, and TextNode extends Node: the first
// attempt at cycle_class_text fails (Node is not declared yet) and is tried again. That attempt's
// errors used to be printed although the compile then succeeded.
import { TextNode } from "./cycle_class_text";

export class Node {
    childNodes: Node[] = [];

    get textContent(): string {
        let t = "";
        for (let i = 0; i < this.childNodes.length; i++) {
            const c = this.childNodes[i];
            if (c instanceof TextNode) {
                t += c.data;
            } else {
                t += c.textContent;
            }
        }

        return t;
    }

    addText(s: string) {
        this.childNodes.push(new TextNode(s));
    }
}
