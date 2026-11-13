import './export_vars_live'

function main() {
    assert(counter == 0, "counter before");
    assert(title == "first", "title before");
    assert(point.x == 1, "point before");
    assert(twice(3) == 6, "twice before");
    assert(holder.value == 1, "holder before");

    // what the library assigns, this module sees
    changeAll();
    assert(counter == 42, "counter after");
    assert(title == "second", "title after");
    assert(point.x == 10 && point.y == 20, "point after");
    assert(twice(3) == 60, "twice after");
    assert(holder.value == 2, "holder after");

    // what this module assigns, the library sees
    counter = 7;
    counter++;
    title = "third";
    point.x = 11;
    assert(counterHere() == 8, "counter assigned here");
    assert(titleHere() == "third", "title assigned here");
    assert(pointXHere() == 11, "point field assigned here");

    point = { x: 30, y: 40 };
    assert(pointXHere() == 30, "point assigned here");

    print("done.");
}
