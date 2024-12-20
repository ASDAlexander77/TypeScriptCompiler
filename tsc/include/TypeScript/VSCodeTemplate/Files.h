#define NODE_MODULE_TSNC_PATH "node_modules/tsnc"
#define DOT_VSCODE_PATH ".vscode"

#define TSCONFIG_JSON_DATA "{\r\n    \"compilerOptions\": {\r\n      \"target\": \"es2017\",\r\n      \"lib\": [\"dom\", \"dom.iterable\", \"esnext\"],\r\n      \"allowJs\": true,\r\n      \"skipLibCheck\": true,\r\n      \"strict\": true,\r\n      \"noEmit\": true,\r\n      \"esModuleInterop\": true,\r\n      \"module\": \"esnext\",\r\n      \"moduleResolution\": \"bundler\",\r\n      \"resolveJsonModule\": true,\r\n      \"isolatedModules\": true,\r\n      \"jsx\": \"preserve\",\r\n      \"incremental\": true,\r\n      \"types\": [\"tsnc\"]\r\n    },\r\n    \"include\": [\"<<PROJECT>>.ts\"],\r\n    \"exclude\": [\"node_modules\"]\r\n  }"
#define TSNC_INDEX_D_TS "declare function print(...args: (string | number)[]) : void;\r\ndeclare function assert(cond: boolean, msg?: string) : void;\r\ndeclare type int = any;"
#define TASKS_JSON_DATA "{}"
#define LAUNCH_JSON_DATA "{}"