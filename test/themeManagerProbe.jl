# Run from the workspace: julia --startup-file=no --project=FlootyPTYLayer FlootyPTYLayer/test/themeManagerProbe.jl
using Test
include(joinpath(@__DIR__, "..", "src", "themeManager.jl"))
using .ThemeManager

@testset "ThemeManager" begin
    mktempdir() do root
        config = joinpath(root, "config", "theme.conf")
        themes = joinpath(root, "themes")
        mkpath(themes)
        withenv("FLOOTY_THEME_CONFIG" => config, "FLOOTY_THEMES_DIR" => themes) do
            expectedKeys = Set([
                "background", "foreground", "cursor", "cursorAccent",
                "selectionBackground", "selectionForeground", "black", "red", "green",
                "yellow", "blue", "magenta", "cyan", "white", "brightBlack", "brightRed",
                "brightGreen", "brightYellow", "brightBlue", "brightMagenta", "brightCyan", "brightWhite",
            ])
            @testset "Bundled themes and missing config" begin
                @test availableThemes() == sort(["Flooty Dark", "Flooty Light", "Dracula"])
                for name in availableThemes()
                    colors = loadTheme(name)
                    @test Set(keys(colors)) == expectedKeys
                    @test all(value -> occursin(r"^#[0-9a-f]{6}$", value), values(colors))
                end
                @test loadTheme("Flooty Dark")["background"] != loadTheme("Flooty Light")["background"]
                startup = themeState()
                @test startup["name"] == "Flooty Dark"
                @test startup["theme"] == loadTheme("Flooty Dark")
                @test startup["configPath"] == config
                @test startup["themesDir"] == themes
                @test !haskey(startup, "error")
                @test !ispath(config)
                @test_throws ArgumentError selectTheme!("Unknown")
                @test !ispath(config)
                @test !ispath(dirname(config))
            end

            custom = joinpath(themes, "Imported")
            write(custom, """
                # A partial Ghostty theme inherits unspecified default colors.
                background = "A1B2C3"
                cursor-color = '#223344'
                palette = 1=#112233
                palette = "1=445566"
                palette = 15="#ABCDEF"
                font-family = Ignored Font
                unrelated-flag
                """)
            @testset "Custom import, repeated palette, config precedence" begin
                imported = loadTheme("Imported")
                @test Set(keys(imported)) == expectedKeys
                @test imported["background"] == "#a1b2c3"
                @test imported["cursor"] == "#223344"
                @test imported["red"] == "#445566"
                @test imported["brightWhite"] == "#abcdef"
                @test imported["foreground"] == loadTheme("Flooty Dark")["foreground"]
                @test "Imported" in availableThemes()
                @test issorted(availableThemes())
                mkpath(dirname(config))
                write(config, "theme = 'Imported'\nbackground = #010203\npalette = 1=#040506\npalette = 1=#070809\n")
                loaded = themeState()
                @test !haskey(loaded, "error")
                @test loaded["name"] == "Imported"
                @test loaded["theme"]["background"] == "#010203"
                @test loaded["theme"]["red"] == "#070809"
                @test loadTheme("Imported")["red"] == "#445566"
                @test loadTheme("Imported")["background"] == "#a1b2c3"
            end

            @testset "User names shadow bundled themes" begin
                shadow = joinpath(themes, "Dracula")
                bundled = loadTheme("Dracula")
                write(shadow, "background = #123456\n")
                @test loadTheme("Dracula")["background"] == "#123456"
                @test count(==("Dracula"), availableThemes()) == 1
                rm(shadow)
                @test loadTheme("Dracula") == bundled
            end

            @testset "Strict failures do not persist" begin
                original = read(config, String)
                for name in ("Missing", "../Imported", "/tmp/Imported", "..", ".", "dir\\Imported", "Imported\ntheme = Dracula")
                    @test_throws ArgumentError loadTheme(name)
                    @test_throws ArgumentError selectTheme!(name)
                    @test read(config, String) == original
                end
                for content in ("background = #xyzxyz\n", "foreground = #123\n", "palette = 16=#112233\n",
                                "palette = -1=#112233\n", "palette = nope=#112233\n", "palette = 0=notacolor\n",
                                "palette = 0\n", "background\n")
                    write(joinpath(themes, "Broken"), content)
                    @test_throws ArgumentError loadTheme("Broken")
                    @test_throws ArgumentError selectTheme!("Broken")
                    @test read(config, String) == original
                end
                rm(joinpath(themes, "Broken"))
            end

            @testset "Selection preserves non-theme lines" begin
                original = "# keep this comment\r\nfont-family = My Font\r\ntheme = Imported\r\n\r\nbackground = #102030\r\ntheme = Flooty Dark\r\n# theme = not a setting\r\n"
                write(config, original)
                selected = selectTheme!("Flooty Light")
                @test read(config, String) == "# keep this comment\r\nfont-family = My Font\r\ntheme = Flooty Light\r\n\r\nbackground = #102030\r\n# theme = not a setting\r\n"
                @test selected["name"] == "Flooty Light"
                @test selected["theme"]["background"] == "#102030"
                @test selected == themeState()
                write(config, "# no selection yet\nfont-size = 14")
                selectTheme!("Dracula")
                @test read(config, String) == "# no selection yet\nfont-size = 14\ntheme = Dracula\n"
                @test themeState()["name"] == "Dracula"
                rm(config)
                @test selectTheme!("Imported")["name"] == "Imported"
                @test themeState()["name"] == "Imported"
            end

            @testset "Corrupt startup falls back completely, without writing" begin
                for content in ("theme = Unknown\n", "theme = ../Imported\n", "theme = Dracula\nbackground = bad\n",
                                "theme = Imported\npalette = 99=#123456\n")
                    write(config, content)
                    fallback = themeState()
                    @test fallback["name"] == "Flooty Dark"
                    @test fallback["theme"] == loadTheme("Flooty Dark")
                    @test read(config, String) == content
                end
                invalidOverrides = read(config, String)
                @test_throws ArgumentError selectTheme!("Dracula")
                @test read(config, String) == invalidOverrides
                write(config, "theme = Unknown\n# a valid selection can repair this\n")
                @test selectTheme!("Dracula")["name"] == "Dracula"
                @test !haskey(themeState(), "error")

                rm(config)
                mkdir(config)
                @test_throws ArgumentError selectTheme!("Dracula")
                @test isdir(config)
                rm(config)

                # A corrupt user theme named like the default must not taint the fallback.
                dark = loadTheme("Flooty Dark")
                write(joinpath(themes, "Flooty Dark"), "background = broken\n")
                write(config, "theme = Flooty Dark\nbackground = #123456\n")
                fallback = themeState()
                @test fallback["theme"] == dark
                @test read(config, String) == "theme = Flooty Dark\nbackground = #123456\n"
            end
        end
    end
end
println("THEME_MANAGER_PROBE_OK")
