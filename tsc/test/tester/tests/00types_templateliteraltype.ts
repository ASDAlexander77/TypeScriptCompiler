type World = "world";

type Greeting = `hello ${World}`;

type EmailLocaleIDs = "welcome_email" | "email_heading";
type FooterLocaleIDs = "footer_title" | "footer_sendoff";

type AllLocaleIDs = `${EmailLocaleIDs | FooterLocaleIDs}_id`;

type PropEventSource<Type> = {
    on(eventName: `${string & keyof Type}Changed`, callback: (newValue: any) => void): void;
};

type person = {
    firstName: "Saoirse",
    lastName: "Ronan",
    age: 26
};

type personEvents = PropEventSource<person>;


type Cases<T extends string> = `${Uppercase<T>} ${Lowercase<T>} ${Capitalize<T>} ${Uncapitalize<T>}`;

type TCA1 = Cases<'bar'>;  // 'BAR bar Bar bar'
type TCA2 = Cases<'BAR'>;  // 'BAR bar BAR bAR'

type Partial<T> = {
    [P in keyof T]?: T[P];
};
type PartialPerson = Partial<Person>;

function main() {

    let a: Greeting;
    let b: AllLocaleIDs;
    let c: personEvents;
    let d: TCA1;
    let e: TCA2;

    print("done.");
}