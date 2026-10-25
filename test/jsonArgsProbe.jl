# Run: julia --startup-file=no --project=. test/jsonArgsProbe.jl
using FlootyPTYLayer
using JSON3
import FlootyPTYLayer: jsonArgs, jsonStr, jsonInt, jsonBool, EMPTY_PAYLOAD

payload = jsonArgs(("""{"force":false,"nested":{"force":true,"items":[false,{"name":"inner"}]},"text":"quote: \\"hello\\"\\npath: C:\\\\tmp","count":12,"fraction":2.6,"numeric_string":"42"}""",))
@assert jsonBool(payload, :force, true) === false
@assert payload[:nested][:force] === true
@assert payload.nested.items[1] === false
@assert payload["nested"]["items"][2]["name"] == "inner"
@assert jsonStr(payload, :text) == "quote: \"hello\"\npath: C:\\tmp"
@assert jsonInt(payload, :count, -1) == 12
@assert jsonInt(payload, :fraction, -1) == 3
@assert jsonStr(payload, :numeric_string) == "42"
@assert jsonInt(payload, :numeric_string, -1) == -1
@assert jsonStr(payload, :count, "fallback") == "fallback"
@assert jsonBool(payload, :count, true) === true
@assert jsonStr(payload, :missing, "fallback") == "fallback"
@assert jsonArgs(JSON3.write(payload)) == payload

for invalid in ("", "{", "{\"force\":}", "[true, false]", "null", "true", "42", "\"text\"", 42, (), (nothing,))
    @assert jsonArgs(invalid) === EMPTY_PAYLOAD
end
for invalid in (
    raw"""{"text":"\q"}""",
    raw"""{"text":"\uZZZZ"}""",
    raw"""{"nested":{"text":"\q"}}""",
    raw"""{"nested":[{"text":"\uZZZZ"}]}""",
    raw"""{"nested":["\q"]}""",
    raw"""{"nested":{"\uZZZZ":true}}""",
)
    @assert jsonArgs(invalid) === EMPTY_PAYLOAD
end
println("PASS jsonArgs payload parsing and validation")
