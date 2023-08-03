const say = (speach: string) => print(speach);
const sayT = <T>(speach: T) => print(speach);

function main() {

    say("Hello");
    sayT("Hello");

    print("done.");
}