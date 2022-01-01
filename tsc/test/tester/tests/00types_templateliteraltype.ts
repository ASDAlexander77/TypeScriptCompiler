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

function main() {

    let a: Greeting;
    let b: AllLocaleIDs;
    let c: personEvents;

    print("done.");
}