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

    if (state.state == "loading") {
        return "Downloading...";
    }

    if (state.state == "failed") {
        return `Error ${state.code} downloading`;
    }

    if (state.state == "success") {
        return `Downloaded ${state.response.title} - ${state.response.summary}`;
    }

    return "<error>";
}

function main() {
    assert(logger({ state: "loading" }) == "Downloading...");
    assert(logger({ state: "failed", code: 1.0 }) == "Error 1 downloading");
    assert(logger({ state: "success", response: { title: "title", duration: 10.0, summary: "summary" } }) == "Downloaded title - summary");
    assert(logger({ state: "???" }) == "<error>");
    print("done.");
}
