// tsconfig.json as the project file: what to compile and with which options.
//
// tsc ignores the keys it does not know at the root of tsconfig.json, so tslang keeps its options
// in a section of its own beside "compilerOptions", as ts-node does:
//
//   {
//     "extends": "./base.json",
//     "compilerOptions": { "strict": true, "outDir": "out" },
//     "files": ["main.ts"],
//     "tslang": { "emit": "exe", "mm": "rc", "opt": true, "lib": ["x.lib"] }
//   }
//
// - The keys of "tslang" are the command-line option names (without dashes), so every option works
//   with no table of its own. A list option takes an array. An unknown key is an error.
// - "tslang.entry", else the only name in "files", else the only literal .ts in "include", is the
//   file to compile: tslang compiles one file and what it imports, not a set of files.
// - Of "compilerOptions" only those that mean something here are read: strictNullChecks (or strict)
//   and outFile / outDir. The rest are tsc's and ignored, as tsc ignores the "tslang" section.
// - An option given on the command line wins over the file, for a list too (it is not merged).
// - Relative paths are relative to the file that names them, including in a file "extends" reads.
//
// The file is read with -p <file or folder>, or, when tslang is given no input file at all, from
// ./tsconfig.json when it is there and names an entry file. With an input file and no -p nothing
// is read, so a script that names its files keeps working as before.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cl = llvm::cl;

extern cl::OptionCategory TypeScriptCompilerCategory;

extern cl::opt<std::string> inputFilename;
extern cl::opt<std::string> outputFilename;
extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> verbose;

std::string getDefaultOutputFileName(enum Action emitAction);

cl::opt<std::string> projectFile("p", cl::desc("Read the program's options from a tsconfig.json (or <folder>/tsconfig.json): its 'tslang' section "
                                                "holds tslang options by their command-line names (\"mm\": \"rc\", \"lib\": [...]), and 'tslang.entry', "
                                                "'files' or 'include' names the file to compile when none is given. Options on the command line win "
                                                "over the file's. Without -p and without an input file, ./tsconfig.json is read when it names one"),
                                 cl::value_desc("file or folder"), cl::cat(TypeScriptCompilerCategory));
cl::alias projectFileAlias("project", cl::desc("Alias for -p"), cl::aliasopt(projectFile));

namespace
{

// options whose value is a path: made absolute against the folder of the file that sets them
const llvm::StringSet<> pathOptions = {"o", "default-lib-path", "gc-lib-path", "gc-shared-lib-path", "llvm-lib-path",
                                       "tslang-lib-path", "emsdk-sysroot-path", "android-ndk-path", "jit-cache-dir",
                                       "object-filename", "obj", "own-facts"};
// ... and those that may be a path or a name the linker or the loader looks up: a path only when
// that file is there
const llvm::StringSet<> pathOrNameOptions = {"lib", "shared-libs"};
// what a project file must not do: name another project, or create one
const llvm::StringSet<> refusedOptions = {"p", "project", "new", "cmake", "install-default-lib", "help", "help-hidden", "version"};

struct ProjectConfig
{
    std::optional<std::string> entry;
    std::optional<std::vector<std::string>> files;
    std::optional<std::vector<std::string>> include;
    std::optional<bool> strict;
    std::optional<bool> strictNullChecks;
    std::optional<std::string> outFile;
    std::optional<std::string> outDir;
    // option name -> its values, in order; one value for an option that is not a list
    std::map<std::string, std::vector<std::string>> options;

    // `over`'s settings replace these, key by key, as a file's replace those of the file it extends
    void mergeFrom(ProjectConfig &&over)
    {
        if (over.entry) entry = std::move(over.entry);
        if (over.files) files = std::move(over.files);
        if (over.include) include = std::move(over.include);
        if (over.strict) strict = over.strict;
        if (over.strictNullChecks) strictNullChecks = over.strictNullChecks;
        if (over.outFile) outFile = std::move(over.outFile);
        if (over.outDir) outDir = std::move(over.outDir);
        for (auto &[name, values] : over.options)
        {
            options[name] = std::move(values);
        }
    }
};

llvm::raw_ostream &error()
{
    return llvm::WithColor::error(llvm::errs(), "tslang");
}

// tsconfig.json is JSONC: comments and trailing commas, which llvm::json does not take. Both become
// spaces, so the positions in a parse error stay those of the file.
std::string stripJsonc(llvm::StringRef text)
{
    if (text.starts_with("\xEF\xBB\xBF"))
    {
        text = text.drop_front(3);
    }

    std::string out(text);
    auto n = out.size();

    // comments
    for (size_t i = 0; i < n; i++)
    {
        auto c = out[i];
        if (c == '"')
        {
            for (i++; i < n && out[i] != '"'; i++)
            {
                if (out[i] == '\\')
                {
                    i++;
                }
            }
        }
        else if (c == '/' && i + 1 < n && out[i + 1] == '/')
        {
            for (; i < n && out[i] != '\n'; i++)
            {
                out[i] = ' ';
            }
        }
        else if (c == '/' && i + 1 < n && out[i + 1] == '*')
        {
            out[i++] = ' ';
            out[i++] = ' ';
            for (; i < n && !(out[i] == '*' && i + 1 < n && out[i + 1] == '/'); i++)
            {
                if (out[i] != '\n')
                {
                    out[i] = ' ';
                }
            }

            if (i < n)
            {
                out[i++] = ' ';
                out[i] = ' ';
            }
        }
    }

    // trailing commas
    for (size_t i = 0; i < n; i++)
    {
        auto c = out[i];
        if (c == '"')
        {
            for (i++; i < n && out[i] != '"'; i++)
            {
                if (out[i] == '\\')
                {
                    i++;
                }
            }
        }
        else if (c == ',')
        {
            auto next = i + 1;
            while (next < n && llvm::isSpace(out[next]))
            {
                next++;
            }

            if (next < n && (out[next] == '}' || out[next] == ']'))
            {
                out[i] = ' ';
            }
        }
    }

    return out;
}

std::string absolutePath(llvm::StringRef baseDir, llvm::StringRef path)
{
    if (llvm::sys::path::is_absolute(path))
    {
        return path.str();
    }

    llvm::SmallString<256> result(baseDir);
    llvm::sys::path::append(result, path);
    llvm::sys::path::remove_dots(result, true);
    return std::string(result);
}

// the text of a scalar for an option's value: what it would be on the command line
std::optional<std::string> scalarText(const llvm::json::Value &value)
{
    if (auto s = value.getAsString())
    {
        return s->str();
    }

    if (auto b = value.getAsBoolean())
    {
        return std::string(*b ? "true" : "false");
    }

    if (auto i = value.getAsInteger())
    {
        return std::to_string(*i);
    }

    if (auto d = value.getAsNumber())
    {
        std::string text;
        llvm::raw_string_ostream os(text);
        os << *d;
        return text;
    }

    return std::nullopt;
}

class ProjectLoader
{
    llvm::SmallVector<std::string> chain; // the files being read, to find an "extends" cycle

  public:
    // read `path` and the files it extends; false after an error, which is printed
    bool load(llvm::StringRef path, ProjectConfig &config)
    {
        llvm::SmallString<256> realPath;
        if (llvm::sys::fs::real_path(path, realPath))
        {
            realPath = path;
        }

        if (llvm::is_contained(chain, std::string(realPath)))
        {
            error() << path << ": 'extends' comes back to this file\n";
            return false;
        }

        auto fileOrErr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
        if (auto ec = fileOrErr.getError())
        {
            error() << "cannot read the project file '" << path << "': " << ec.message() << "\n";
            return false;
        }

        auto parsed = llvm::json::parse(stripJsonc((*fileOrErr)->getBuffer()));
        if (!parsed)
        {
            error() << path << ": " << llvm::toString(parsed.takeError()) << "\n";
            return false;
        }

        auto *root = parsed->getAsObject();
        if (!root)
        {
            error() << path << ": a project file holds an object\n";
            return false;
        }

        chain.push_back(std::string(realPath));
        auto ok = loadObject(path, *root, config);
        chain.pop_back();
        return ok;
    }

  private:
    bool loadObject(llvm::StringRef path, const llvm::json::Object &root, ProjectConfig &config)
    {
        llvm::SmallString<256> dir(path);
        llvm::sys::path::remove_filename(dir);
        if (dir.empty())
        {
            dir = ".";
        }

        // what it extends first, then this file over it; of several, a later one wins
        if (auto *extends = root.get("extends"))
        {
            std::vector<std::string> bases;
            if (auto s = extends->getAsString())
            {
                bases.push_back(s->str());
            }
            else if (auto *array = extends->getAsArray())
            {
                for (auto &item : *array)
                {
                    auto s = item.getAsString();
                    if (!s)
                    {
                        error() << path << ": 'extends' holds file names\n";
                        return false;
                    }

                    bases.push_back(s->str());
                }
            }
            else
            {
                error() << path << ": 'extends' is a file name or an array of them\n";
                return false;
            }

            for (auto &base : bases)
            {
                llvm::StringRef baseRef(base);
                if (!baseRef.starts_with("./") && !baseRef.starts_with("../") && !baseRef.starts_with(".\\")
                    && !baseRef.starts_with("..\\") && !llvm::sys::path::is_absolute(baseRef))
                {
                    error() << path << ": 'extends': \"" << base
                            << "\" names a package, which tslang does not look up; give the file's path, starting with ./ or ../\n";
                    return false;
                }

                auto basePath = absolutePath(dir, base);
                if (!llvm::sys::fs::exists(basePath) && !llvm::StringRef(basePath).ends_with(".json"))
                {
                    basePath += ".json";
                }

                ProjectConfig baseConfig;
                if (!load(basePath, baseConfig))
                {
                    return false;
                }

                config.mergeFrom(std::move(baseConfig));
            }
        }

        ProjectConfig own;
        if (!readNames(path, dir, root, "files", own.files) || !readNames(path, dir, root, "include", own.include))
        {
            return false;
        }

        if (auto *compilerOptions = root.get("compilerOptions"))
        {
            auto *object = compilerOptions->getAsObject();
            if (!object)
            {
                error() << path << ": 'compilerOptions' is an object\n";
                return false;
            }

            own.strict = object->getBoolean("strict");
            own.strictNullChecks = object->getBoolean("strictNullChecks");
            if (auto outFile = object->getString("outFile"))
            {
                own.outFile = absolutePath(dir, *outFile);
            }

            if (auto outDir = object->getString("outDir"))
            {
                own.outDir = absolutePath(dir, *outDir);
            }
        }

        if (auto *section = root.get("tslang"))
        {
            auto *object = section->getAsObject();
            if (!object)
            {
                error() << path << ": 'tslang' is an object of options, such as { \"mm\": \"rc\" }\n";
                return false;
            }

            if (!readOptions(path, dir, *object, own))
            {
                return false;
            }
        }

        config.mergeFrom(std::move(own));
        return true;
    }

    bool readNames(llvm::StringRef path, llvm::StringRef dir, const llvm::json::Object &root, llvm::StringRef key,
                   std::optional<std::vector<std::string>> &names)
    {
        auto *value = root.get(key);
        if (!value)
        {
            return true;
        }

        auto *array = value->getAsArray();
        if (!array)
        {
            error() << path << ": '" << key << "' is an array of file names\n";
            return false;
        }

        names.emplace();
        for (auto &item : *array)
        {
            auto s = item.getAsString();
            if (!s)
            {
                error() << path << ": '" << key << "' is an array of file names\n";
                return false;
            }

            names->push_back(absolutePath(dir, *s));
        }

        return true;
    }

    bool readOptions(llvm::StringRef path, llvm::StringRef dir, const llvm::json::Object &section, ProjectConfig &config)
    {
        auto &registered = cl::getRegisteredOptions();

        // in name order, so that what is reported first does not depend on a hash
        std::vector<llvm::StringRef> names;
        for (auto &item : section)
        {
            names.push_back(item.first);
        }

        llvm::sort(names);

        for (auto name : names)
        {
            auto &value = *section.get(name);

            if (name == "entry")
            {
                auto s = value.getAsString();
                if (!s)
                {
                    error() << path << ": 'tslang.entry' is the name of the file to compile\n";
                    return false;
                }

                config.entry = absolutePath(dir, *s);
                continue;
            }

            auto found = registered.find(name);
            if (found == registered.end() || refusedOptions.contains(name))
            {
                error() << path << ": 'tslang' section: '" << name << "' is not "
                        << (found == registered.end() ? "a tslang option" : "an option a project file can set") << "\n";
                return false;
            }

            auto *option = found->second;
            if (option->isPositional() || option->isSink() || option->isConsumeAfter())
            {
                error() << path << ": 'tslang' section: '" << name << "' is not an option a project file can set\n";
                return false;
            }

            std::vector<std::string> values;
            if (auto *array = value.getAsArray())
            {
                if (option->getNumOccurrencesFlag() != cl::ZeroOrMore && option->getNumOccurrencesFlag() != cl::OneOrMore
                    && !(option->getMiscFlags() & cl::CommaSeparated))
                {
                    error() << path << ": 'tslang' section: '" << name << "' takes one value, not an array\n";
                    return false;
                }

                for (auto &item : *array)
                {
                    auto text = scalarText(item);
                    if (!text)
                    {
                        error() << path << ": 'tslang' section: '" << name << "' holds strings, numbers or booleans\n";
                        return false;
                    }

                    values.push_back(std::move(*text));
                }
            }
            else if (auto text = scalarText(value))
            {
                values.push_back(std::move(*text));
            }
            else
            {
                error() << path << ": 'tslang' section: '" << name << "' is a string, a number, a boolean or an array of them\n";
                return false;
            }

            for (auto &text : values)
            {
                if (pathOptions.contains(name))
                {
                    text = absolutePath(dir, text);
                }
                else if (pathOrNameOptions.contains(name) && !llvm::sys::path::is_absolute(text))
                {
                    auto candidate = absolutePath(dir, text);
                    if (llvm::sys::fs::exists(candidate))
                    {
                        text = candidate;
                    }
                }
            }

            config.options[name.str()] = std::move(values);
        }

        return true;
    }
};

std::optional<std::string> entryFile(const ProjectConfig &config)
{
    if (config.entry)
    {
        return config.entry;
    }

    if (config.files && config.files->size() == 1)
    {
        return config.files->front();
    }

    if (!config.files && config.include && config.include->size() == 1)
    {
        llvm::StringRef include = config.include->front();
        if (include.ends_with(".ts") && include.find_first_of("*?") == llvm::StringRef::npos)
        {
            return include.str();
        }
    }

    return std::nullopt;
}

} // namespace

// Apply the project file, if there is one to read: call it after the command line is parsed and
// before anything reads an option. `takesInput` is false for the actions that compile nothing
// (--new, --print-default-lib-dir, ...): for those ./tsconfig.json is not looked for. Returns the
// exit code of an error, else 0.
int applyProjectConfig(bool takesInput)
{
    std::string path;
    auto explicitProject = projectFile.getNumOccurrences() > 0;
    if (explicitProject)
    {
        path = projectFile.getValue();
        if (llvm::sys::fs::is_directory(path))
        {
            llvm::SmallString<256> file(path);
            llvm::sys::path::append(file, "tsconfig.json");
            path = std::string(file);
        }

        if (!llvm::sys::fs::exists(path))
        {
            error() << "-p: no project file '" << path << "'\n";
            return 1;
        }
    }
    else if (takesInput && inputFilename.getNumOccurrences() == 0 && llvm::sys::fs::exists("tsconfig.json"))
    {
        path = "tsconfig.json";
    }
    else
    {
        return 0;
    }

    ProjectConfig config;
    ProjectLoader loader;
    if (!loader.load(path, config))
    {
        return 1;
    }

    auto entry = entryFile(config);
    if (inputFilename.getNumOccurrences() == 0)
    {
        if (!entry)
        {
            if (!explicitProject)
            {
                // a tsconfig.json for the editor only: keep to what tslang did without it
                return 0;
            }

            if (takesInput)
            {
                error() << path
                        << ": no file to compile: give it as \"tslang\": { \"entry\": \"main.ts\" }, as the only name in "
                           "'files', or on the command line\n";
                return 1;
            }
        }
        else
        {
            inputFilename.setValue(*entry);
        }
    }

    if (verbose)
    {
        llvm::errs() << "tslang: project file " << path << "\n";
    }

    // compilerOptions first, so that the 'tslang' section, which names tslang's own options, wins
    std::map<std::string, std::vector<std::string>> options;
    if (auto strictNullChecks = config.strictNullChecks ? config.strictNullChecks : config.strict)
    {
        options["strict-null-checks"] = {*strictNullChecks ? "true" : "false"};
    }

    if (config.outFile)
    {
        options["o"] = {*config.outFile};
    }

    for (auto &[name, values] : config.options)
    {
        options[name] = values;
    }

    auto &registered = cl::getRegisteredOptions();
    for (auto &[name, values] : options)
    {
        auto *option = registered[name];
        // the command line wins, for a list too
        if (option->getNumOccurrences() > 0)
        {
            continue;
        }

        for (auto &value : values)
        {
            if (option->addOccurrence(0, name, value))
            {
                // the option printed what is wrong with the value
                llvm::errs() << "  (set in " << path << ")\n";
                return 1;
            }
        }
    }

    // outDir: the default output name, in that folder
    if (config.outDir && outputFilename.getNumOccurrences() == 0)
    {
        switch (emitAction)
        {
        case Action::DumpLLVMIR:
        case Action::DumpByteCode:
        case Action::DumpObj:
        case Action::DumpAssembly:
        case Action::BuildExe:
        case Action::BuildDll:
            {
                if (auto ec = llvm::sys::fs::create_directories(*config.outDir))
                {
                    error() << path << ": cannot create the outDir folder '" << *config.outDir << "': " << ec.message() << "\n";
                    return 1;
                }

                llvm::SmallString<256> output(*config.outDir);
                llvm::sys::path::append(output, llvm::sys::path::filename(getDefaultOutputFileName(emitAction)));
                outputFilename.setValue(std::string(output));
            }
            break;
        default:
            break;
        }
    }

    return 0;
}
