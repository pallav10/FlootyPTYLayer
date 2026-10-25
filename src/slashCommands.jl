const SLASH_COMMAND_CATALOG = Ref{Union{Nothing,Function}}(nothing)
const SLASH_COMMAND_DISPATCHER = Ref{Union{Nothing,Function}}(nothing)
const SUBSCRIPTION_UI_HANDLER = Ref{Union{Nothing,Function}}(nothing)

function addSlashCommandHooks!(catalog::Function, dispatcher::Function, subscription::Function)
    SLASH_COMMAND_CATALOG[] = catalog
    SLASH_COMMAND_DISPATCHER[] = dispatcher
    SUBSCRIPTION_UI_HANDLER[] = subscription
    return nothing
end

isSlashCommandDraft(text::AbstractString) = startswith(lstrip(text), "/")

function dispatchAssistSlash(payload)
    dispatcher = SLASH_COMMAND_DISPATCHER[]
    dispatcher === nothing && return (handled = false,)
    return Base.invokelatest(dispatcher, jsonStr(payload, :text), jsonStr(payload, :session_id))
end

function bindSlashCommands!(wv)
    Webviews.bind(wv, "flootySlashCommands") do _
        try
            catalog = SLASH_COMMAND_CATALOG[]
            commands = catalog === nothing ? [] : Base.invokelatest(catalog)
            JSON3.write((success = true, commands = commands))
        catch err
            errorJson(sprint(showerror, err))
        end
    end
    Webviews.bind(wv, "flootySlashDispatch") do args
        try
            result = dispatchAssistSlash(jsonArgs(args))
            JSON3.write(merge((success = true,), result))
        catch err
            errorJson(sprint(showerror, err))
        end
    end
    Webviews.bind(wv, "flootySubscription") do args
        try
            handler = SUBSCRIPTION_UI_HANDLER[]
            handler === nothing && return errorJson("No subscription controller is installed.")
            result = Base.invokelatest(handler, jsonArgs(args))
            JSON3.write(merge((success = true,), result))
        catch err
            errorJson(sprint(showerror, err))
        end
    end
    return nothing
end
