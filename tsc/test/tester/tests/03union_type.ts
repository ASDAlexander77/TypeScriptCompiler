type NetworkLoadingState = {
    state: "loading";
};

type NetworkFailedState = {
    state: "failed";
    code: number;
};

type NetworkSuccessState = {
    state: "success";
    response: {
        title: string;
        duration: number;
        summary: string;
    };
};

type NetworkState =
    | NetworkLoadingState
    | NetworkFailedState
    | NetworkSuccessState;


function logger(state: NetworkState): string {
    switch (state.state) {
        case "loading":
            return "Downloading...";
        case "failed":
            // The type must be NetworkFailedState here,
            // so accessing the `code` field is safe
            return `Error ${state.code} downloading`;
        case "success":
            return `Downloaded ${state.response.title} - ${state.response.summary}`;
        default:
            return "<error>";
    }
}

function main() {
    assert(logger({ state: "loading" }) == "Downloading...");
    assert(logger({ state: "failed", code: 1.0 }) == "Error 1 downloading");
    assert(logger({ state: "success", response: { title: "title", duration: 10.0, summary: "summary" } }) == "Downloaded title - summary");
    assert(logger({ state: "???" }) == "<error>");
    print("done.");
}
