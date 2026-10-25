module ThemeManager

export themeState, selectTheme!, loadTheme, availableThemes

const THEME_LOCK = ReentrantLock()
const BUNDLED_DIR = normpath(joinpath(@__DIR__, "..", "assets", "themes"))
const BUNDLED_NAMES = ["Flooty Dark", "Flooty Light", "Dracula"]
const DEFAULT_NAME = "Flooty Dark"
const COLOR_KEYS = Dict(
    "background" => "background", "foreground" => "foreground",
    "cursor-color" => "cursor", "cursor-text" => "cursorAccent",
    "selection-background" => "selectionBackground",
    "selection-foreground" => "selectionForeground",
)
const PALETTE_KEYS = (
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    "brightBlack", "brightRed", "brightGreen", "brightYellow", "brightBlue",
    "brightMagenta", "brightCyan", "brightWhite",
)

configPath() = abspath(expanduser(get(ENV, "FLOOTY_THEME_CONFIG", joinpath(homedir(), ".flooty", "theme.conf"))))
themesDir() = abspath(expanduser(get(ENV, "FLOOTY_THEMES_DIR", joinpath(homedir(), ".flooty", "themes"))))

validName(name::AbstractString) = !isempty(name) && strip(name) == name &&
    name ∉ (".", "..") && !any(c -> c in ('/', '\\', '\'', '"') || iscntrl(c), name)

function themeName(name::AbstractString)::String
    validName(name) || throw(ArgumentError("Theme must be a name, not a path or quoted value: $(repr(name))"))
    return String(name)
end

function unquote(value::AbstractString)::String
    value = strip(value)
    if length(value) >= 2 && first(value) in ('\'', '"') && last(value) == first(value)
        return String(value[nextind(value, firstindex(value)):prevind(value, lastindex(value))])
    end
    return String(value)
end

function color(value::AbstractString)::String
    value = unquote(value)
    occursin(r"^#?[0-9a-fA-F]{6}$", value) || throw(ArgumentError("Invalid theme color: $(repr(value)); expected six hex digits"))
    return "#" * lowercase(startswith(value, '#') ? value[2:end] : value)
end

function parseSettings(text::AbstractString; selection::Bool = false)
    selected = DEFAULT_NAME
    colors = Dict{String,String}()
    for (lineNumber, line) in enumerate(eachline(IOBuffer(text)))
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        parts = split(line, '='; limit = 2)
        key = strip(parts[1])
        relevant = haskey(COLOR_KEYS, key) || key == "palette" || (selection && key == "theme")
        relevant || continue  # Other Ghostty settings do not affect a terminal theme.
        length(parts) == 2 || throw(ArgumentError("Missing '=' for $key on line $lineNumber"))
        value = unquote(parts[2])
        if key == "theme"
            selected = themeName(value)
        elseif key == "palette"
            entry = split(value, '='; limit = 2)
            index = tryparse(Int, strip(entry[1]))
            (length(entry) == 2 && index !== nothing && 0 <= index <= 15) ||
                throw(ArgumentError("Invalid palette entry on line $lineNumber; expected index 0..15=color"))
            colors[PALETTE_KEYS[index + 1]] = color(entry[2])
        else
            colors[COLOR_KEYS[key]] = color(value)
        end
    end
    return selected, colors
end

function themeNames(dir::String)::Vector{String}
    names = copy(BUNDLED_NAMES)
    if ispath(dir)
        isdir(dir) || throw(ArgumentError("Themes directory is not a directory: $dir"))
        append!(names, filter(name -> validName(name) && isfile(joinpath(dir, name)), readdir(dir)))
    end
    return sort!(unique!(names))
end

"Return sorted bundled and user theme names; user files shadow bundled names."
function availableThemes()::Vector{String}
    return lock(THEME_LOCK) do
        themeNames(themesDir())
    end
end

function bundledDark()::Dict{String,String}
    _, colors = parseSettings(read(joinpath(BUNDLED_DIR, DEFAULT_NAME), String))
    return colors
end

function readTheme(name::AbstractString, dir::String)::Dict{String,String}
    name = themeName(name)
    userPath = joinpath(dir, name)
    path = if isfile(userPath)
        userPath
    elseif name in BUNDLED_NAMES
        joinpath(BUNDLED_DIR, name)
    else
        throw(ArgumentError("Unknown theme: $name"))
    end
    colors = bundledDark()
    _, overrides = parseSettings(read(path, String))
    merge!(colors, overrides)
    return colors
end

"Strictly load one theme as a complete xterm color dictionary, without config overrides."
function loadTheme(name::AbstractString)::Dict{String,String}
    return lock(THEME_LOCK) do
        readTheme(name, themesDir())
    end
end

function state(name::String, colors::Dict{String,String}, names::Vector{String}, path::String, dir::String)::Dict{String,Any}
    return Dict{String,Any}("name" => name, "theme" => colors, "themes" => names,
        "configPath" => path, "themesDir" => dir)
end

readConfig(path::String) = isfile(path) ? read(path, String) :
    ispath(path) ? throw(ArgumentError("Theme config is not a file: $path")) : ""

"Read the current config; invalid startup settings return a complete default theme and error."
function themeState()::Dict{String,Any}
    return lock(THEME_LOCK) do
        path, dir = configPath(), themesDir()
        names = sort(copy(BUNDLED_NAMES))
        try
            names = themeNames(dir)
            name, overrides = parseSettings(readConfig(path); selection = true)
            colors = readTheme(name, dir)
            merge!(colors, overrides)
            return state(name, colors, names, path, dir)
        catch err
            err isa InterruptException && rethrow()
            fallback = state(DEFAULT_NAME, bundledDark(), names, path, dir)
            fallback["error"] = "Could not load theme config; using bundled Flooty Dark: " * sprint(showerror, err)
            return fallback
        end
    end
end

function selectedConfig(text::String, name::String)::String
    output = IOBuffer()
    replaced = false
    for line in eachline(IOBuffer(text); keep = true)
        key = strip(first(split(strip(line), '='; limit = 2)))
        if key == "theme"
            if !replaced
                ending = endswith(line, "\r\n") ? "\r\n" : endswith(line, "\n") ? "\n" : ""
                print(output, "theme = ", name, ending)
                replaced = true
            end
        else
            print(output, line)
        end
    end
    if !replaced
        ending = occursin("\r\n", text) ? "\r\n" : "\n"
        !isempty(text) && !endswith(text, '\n') && print(output, ending)
        print(output, "theme = ", name, ending)
    end
    return String(take!(output))
end

function saveConfig(path::String, text::String)::Nothing
    mkpath(dirname(path))
    tempPath, io = mktemp(dirname(path); cleanup = false)
    try
        write(io, text)
        flush(io)
        close(io)
        # Same-directory rename replaces the old config atomically, without deleting it first.
        Base.Filesystem.rename(tempPath, path)
    finally
        isopen(io) && close(io)
        ispath(tempPath) && rm(tempPath; force = true)
    end
    return nothing
end

"Validate and atomically persist a selection, preserving non-theme config lines and comments."
function selectTheme!(name::AbstractString)::Dict{String,Any}
    return lock(THEME_LOCK) do
        name = themeName(name)
        path, dir = configPath(), themesDir()
        text = readConfig(path)
        # Ignore the old selection so a valid choice can repair an unknown selected theme.
        _, overrides = parseSettings(text)
        colors = readTheme(name, dir)
        merge!(colors, overrides)
        result = state(name, colors, themeNames(dir), path, dir)
        saveConfig(path, selectedConfig(text, name))
        return result
    end
end

end # module ThemeManager
