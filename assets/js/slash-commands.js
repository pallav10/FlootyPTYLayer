window.createFlootySlashCommands = function ({ editor, callJulia, context, onModelChanged }) {
    const host = document.getElementById('assistEditor');
    const textarea = host.querySelector('textarea');
    const menu = document.createElement('div');
    menu.id = 'slashCommandMenu';
    menu.className = 'slash-command-menu';
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-label', 'Available actions');
    menu.hidden = true;
    host.parentElement.appendChild(menu);
    const feedback = document.createElement('div');
    feedback.className = 'slash-command-feedback';
    feedback.setAttribute('role', 'status');
    feedback.hidden = true;
    host.appendChild(feedback);
    textarea.setAttribute('aria-controls', menu.id);
    textarea.setAttribute('aria-autocomplete', 'list');
    textarea.setAttribute('aria-expanded', 'false');
    textarea.setAttribute('role', 'combobox');
    textarea.setAttribute('aria-haspopup', 'listbox');
    textarea.setAttribute('aria-label', 'Message or action');
    textarea.setAttribute('placeholder', 'Message the agent, or type / for actions');
    let commands = [], matches = [], selected = 0, dismissed = null, loadingCatalog = false;
    let dialog = null, generation = 0, timer = null, login = null, providers = [], provider = null;
    let copilotDomain = null, choosingCopilotHost = false, startingLogin = false;

    function message(text) {
        feedback.textContent = text;
        feedback.hidden = !text;
    }
    function hideMenu() {
        menu.hidden = true;
        textarea.setAttribute('aria-expanded', 'false');
        textarea.removeAttribute('aria-activedescendant');
    }
    function updateMenu() {
        const text = editor.getValue().trimStart();
        if (!/^\/[A-Za-z0-9_-]*$/.test(text) || text === dismissed) { hideMenu(); return; }
        matches = commands.filter(command => command.name.startsWith(text.slice(1).toLowerCase()));
        selected = Math.min(selected, Math.max(0, matches.length - 1));
        menu.replaceChildren();
        matches.forEach((command, index) => {
            const row = document.createElement('button');
            row.type = 'button';
            row.id = `slash-action-${index}`;
            row.setAttribute('role', 'option');
            row.setAttribute('aria-selected', String(index === selected));
            row.tabIndex = -1;
            const name = document.createElement('strong');
            name.textContent = '/' + command.name;
            const description = document.createElement('span');
            description.textContent = command.description;
            row.append(name, description);
            row.addEventListener('mousedown', event => event.preventDefault());
            row.addEventListener('click', () => execute('/' + command.name));
            menu.appendChild(row);
        });
        if (!matches.length) {
            const empty = document.createElement('p');
            empty.textContent = 'No matching actions';
            menu.appendChild(empty);
        }
        menu.hidden = false;
        textarea.setAttribute('aria-expanded', 'true');
        if (matches.length) textarea.setAttribute('aria-activedescendant', `slash-action-${selected}`);
        else textarea.removeAttribute('aria-activedescendant');
        if (!commands.length && !loadingCatalog) loadCatalog();
    }
    async function loadCatalog() {
        loadingCatalog = true;
        try {
            const result = await callJulia('flootySlashCommands', '{}');
            if (!result?.success) throw new Error(result?.error || 'Actions are unavailable.');
            commands = result.commands || [];
            updateMenu();
        } catch (error) { message(error.message); }
        finally { loadingCatalog = false; }
    }
    async function api(operation, extra = {}) {
        const result = await callJulia('flootySubscription', JSON.stringify({ operation, ...extra }));
        if (!result?.success) throw new Error(result?.error || 'Subscription action failed.');
        return result;
    }
    function element(tag, text, className) {
        const node = document.createElement(tag);
        if (text) node.textContent = text;
        if (className) node.className = className;
        return node;
    }
    function button(text, action) {
        const node = element('button', text);
        node.type = 'button';
        node.addEventListener('click', action);
        return node;
    }
    function reportError(error) {
        if (!dialog?.open) return;
        const output = dialog.querySelector('[role="alert"]');
        output.textContent = error.message;
        output.hidden = false;
    }
    function stopPolling() {
        clearTimeout(timer);
        timer = null;
    }
    function closeDialog() {
        generation++;
        stopPolling();
        if (login && !['connected', 'cancelled', 'error'].includes(login.status)) {
            api('cancel', { id: login.id }).catch(() => {});
        }
        login = null;
        startingLogin = false;
        dialog.close();
        editor.focus();
    }
    function chooseCopilotHost() {
        generation++;
        stopPolling();
        login = null;
        choosingCopilotHost = true;
        render();
        dialog.querySelector('#subscriptionGithubHost').focus();
        schedulePoll();
    }
    function render() {
        const body = dialog.querySelector('.subscription-body');
        const previousModel = body.querySelector('#subscriptionModel');
        const modelFocused = document.activeElement === previousModel;
        const copilot = provider === 'github-copilot';
        const hostForm = dialog.querySelector('.subscription-host-form');
        const showHost = copilot && (choosingCopilotHost || ['cancelled', 'error'].includes(login?.status));
        hostForm.hidden = !showHost;
        const hostInput = hostForm.querySelector('input');
        const domain = copilotDomain ?? 'github.com';
        if (hostInput.value !== domain) hostInput.value = domain;
        hostInput.disabled = startingLogin;
        hostForm.querySelector('button').disabled = startingLogin;
        dialog.querySelector('#subscriptionBack').hidden = !provider;
        body.replaceChildren();
        if (!provider) {
            body.appendChild(element('p', 'Choose a subscription. Sign-in and model availability depend on your account.'));
            providers.forEach(item => {
                const connectedHost = item.id === 'github-copilot' ? ` (${item.domain || 'github.com'})` : '';
                const suffix = item.connected ? ' — connected' + connectedHost : item.checking ? ' — checking…' : '';
                const row = button(item.name + suffix, () => {
                    provider = item.id;
                    login = null;
                    if (provider === 'github-copilot') {
                        copilotDomain ??= item.domain || 'github.com';
                        if (!item.connected) { chooseCopilotHost(); return; }
                    }
                    choosingCopilotHost = false;
                    render();
                    if (!item.connected) startLogin();
                });
                row.className = 'subscription-provider';
                row.disabled = item.checking && !item.connected;
                body.appendChild(row);
                if (item.error) body.appendChild(element('p', item.error, 'subscription-note'));
            });
            if (!providers.length) body.appendChild(element('p', 'Checking subscriptions…'));
            return;
        }
        const item = providers.find(entry => entry.id === provider);
        body.appendChild(element('h3', item?.name || provider));
        if (copilot && item?.connected)
            body.appendChild(element('p', `Connected to ${item.domain || 'github.com'}.`, 'subscription-note'));
        if (login) {
            const status = element('p', '', 'subscription-status');
            status.setAttribute('role', 'status');
            status.setAttribute('aria-live', 'polite');
            status.textContent = ({ starting: 'Starting sign-in…', waiting: 'Waiting for authorization…',
                exchanging: 'Completing sign-in…', loading_models: 'Loading available models…',
                connected: 'Connected.', cancelled: 'Sign-in cancelled.', error: 'Sign-in failed.' })[login.status] || login.status;
            body.appendChild(status);
            if (copilot) body.appendChild(element('p', `GitHub host: ${login.domain || copilotDomain || 'github.com'}`, 'subscription-note'));
            if (login.instructions) body.appendChild(element('p', login.instructions));
            if (login.error) body.appendChild(element('p', login.error, 'subscription-note'));
            if (login.userCode) {
                const label = element('label', 'Device code');
                label.htmlFor = 'subscriptionDeviceCode';
                const code = element('input');
                code.id = label.htmlFor;
                code.value = login.userCode;
                code.readOnly = true;
                code.addEventListener('focus', () => code.select());
                body.append(label, code);
            }
            if (login.authorizationUrl && login.status === 'waiting') {
                body.appendChild(button('Open sign-in page', () => api('open', { id: login.id }).catch(reportError)));
            }
            if (login.kind === 'paste' && login.status === 'waiting') {
                const form = element('form');
                const label = element('label', 'Full redirect URL');
                label.htmlFor = 'subscriptionRedirect';
                const input = element('input');
                input.id = label.htmlFor;
                input.type = 'url';
                input.required = true;
                input.autocomplete = 'off';
                input.spellcheck = false;
                input.setAttribute('autocapitalize', 'none');
                const submit = element('button', 'Complete sign-in');
                submit.type = 'submit';
                form.append(label, input, submit);
                form.addEventListener('submit', async event => {
                    event.preventDefault();
                    submit.disabled = true;
                    const current = generation;
                    try {
                        const result = await api('submit', { id: login.id, redirectUrl: input.value });
                        input.value = '';
                        if (current !== generation) return;
                        login = result.login;
                        render();
                        schedulePoll();
                    } catch (error) { reportError(error); submit.disabled = false; }
                });
                body.appendChild(form);
            }
        }
        if (item?.connected && (copilot || !login || login.status === 'connected' || login.status === 'error')) {
            const models = item.models || [];
            if (models.length) {
                const label = element('label', 'Model for the active session');
                label.htmlFor = 'subscriptionModel';
                const select = element('select');
                select.id = label.htmlFor;
                models.forEach(model => {
                    const option = element('option', model);
                    option.value = model;
                    select.appendChild(option);
                });
                if (models.includes(previousModel?.value)) select.value = previousModel.value;
                body.append(label, select, button('Use model', async () => {
                    try {
                        await api('use', { provider, model: select.value, ...context() });
                        onModelChanged();
                        closeDialog();
                    } catch (error) { reportError(error); }
                }));
                if (modelFocused) select.focus();
            } else body.appendChild(element('p', item.error || 'Checking available models…'));
        }
        if (!showHost && (!login || ['connected', 'cancelled', 'error'].includes(login.status))) {
            body.appendChild(button(copilot ? 'Sign in with another GitHub host' :
                item?.connected ? 'Sign in again' : 'Retry sign-in', copilot ? chooseCopilotHost : startLogin));
        }
    }
    function schedulePoll() {
        stopPolling();
        timer = setTimeout(poll, 700);
    }
    async function poll() {
        const current = generation;
        try {
            const result = login ? await api('status', { id: login.id }) : await api('providers');
            if (current !== generation || !dialog.open) return;
            const changedProviders = result.providers && JSON.stringify(result.providers) !== JSON.stringify(providers);
            if (result.providers) providers = result.providers;
            const changed = result.login && JSON.stringify(result.login) !== JSON.stringify(login);
            if (result.login) login = result.login;
            const editingRedirect = document.activeElement?.id === 'subscriptionRedirect';
            if (changed || (changedProviders && !editingRedirect)) render();
            if ((login && !['connected', 'cancelled', 'error'].includes(login.status)) ||
                (!login && providers.some(item => item.checking))) schedulePoll();
        } catch (error) { if (current === generation) reportError(error); }
    }
    async function startLogin() {
        if (startingLogin) return;
        const current = ++generation;
        stopPolling();
        startingLogin = true;
        dialog.querySelector('[role="alert"]').hidden = true;
        if (provider === 'github-copilot') render();
        try {
            const result = await api('begin', { provider,
                ...(provider === 'github-copilot' ? { domain: copilotDomain ?? '' } : {}) });
            if (current !== generation || !dialog.open) {
                api('cancel', { id: result.login.id }).catch(() => {});
                return;
            }
            login = result.login;
            startingLogin = false;
            choosingCopilotHost = false;
            render();
            schedulePoll();
            if (provider === 'github-copilot')
                dialog.querySelector('.subscription-body input, .subscription-body button')?.focus();
        } catch (error) {
            if (current !== generation) return;
            startingLogin = false;
            render();
            reportError(error);
            if (provider === 'github-copilot') dialog.querySelector('#subscriptionGithubHost').focus();
        }
    }
    async function openLogin() {
        if (!dialog) {
            dialog = element('dialog', '', 'subscription-dialog');
            dialog.id = 'subscriptionDialog';
            dialog.setAttribute('aria-labelledby', 'subscriptionTitle');
            const title = element('h2', 'Connect a model subscription');
            title.id = 'subscriptionTitle';
            const error = element('p');
            error.setAttribute('role', 'alert');
            error.hidden = true;
            const hostForm = element('form', '', 'subscription-host-form');
            const hostLabel = element('label', 'GitHub host');
            hostLabel.htmlFor = 'subscriptionGithubHost';
            const hostInput = element('input');
            hostInput.id = hostLabel.htmlFor;
            hostInput.type = 'text';
            hostInput.autocomplete = 'off';
            hostInput.spellcheck = false;
            hostInput.setAttribute('autocapitalize', 'none');
            hostInput.setAttribute('autocorrect', 'off');
            hostInput.setAttribute('aria-describedby', 'subscriptionGithubHostHint');
            hostInput.addEventListener('input', () => { copilotDomain = hostInput.value; });
            const hint = element('p', 'Use github.com for personal and normal GitHub Copilot Business accounts. ' +
                'For GitHub Enterprise, enter your actual sign-in host, for example company.ghe.com. ' +
                'Enter a hostname or HTTPS root URL; an empty host uses github.com.', 'subscription-note');
            hint.id = 'subscriptionGithubHostHint';
            const proceed = element('button', 'Continue');
            proceed.type = 'submit';
            hostForm.append(hostLabel, hostInput, hint, proceed);
            hostForm.addEventListener('submit', event => {
                event.preventDefault();
                if (provider === 'github-copilot') startLogin();
            });
            const back = button('Choose another subscription', () => {
                generation++;
                stopPolling();
                if (login) api('cancel', { id: login.id }).catch(() => {});
                login = null;
                provider = null;
                choosingCopilotHost = false;
                startingLogin = false;
                render();
                schedulePoll();
            });
            back.id = 'subscriptionBack';
            dialog.append(title, error, element('div', '', 'subscription-body'), hostForm, back, button('Close', closeDialog));
            dialog.addEventListener('cancel', event => { event.preventDefault(); closeDialog(); });
            dialog.addEventListener('keydown', event => event.stopPropagation());
            document.body.appendChild(dialog);
        }
        if (dialog.open) return;
        const current = ++generation;
        login = null;
        provider = null;
        providers = [];
        copilotDomain = null;
        choosingCopilotHost = false;
        startingLogin = false;
        dialog.querySelector('[role="alert"]').hidden = true;
        render();
        dialog.showModal();
        try {
            const result = await api('refresh');
            if (current !== generation || !dialog.open) return;
            providers = result.providers || [];
            render();
            schedulePoll();
        } catch (error) { if (current === generation) reportError(error); }
    }
    async function execute(text) {
        hideMenu();
        message('');
        try {
            const result = await callJulia('flootySlashDispatch', JSON.stringify({ text, ...context() }));
            if (!result?.success) throw new Error(result?.error || 'Action failed.');
            if (result.action === 'error') { message(result.message); return; }
            if (result.handled) editor.setValue('');
            if (result.action === 'login') await openLogin();
        } catch (error) { message(error.message); }
    }
    editor.on('input', () => {
        if (editor.getValue().trimStart() !== dismissed) dismissed = null;
        selected = 0;
        message('');
        updateMenu();
    });
    return {
        submit(text) {
            if (!/^\/(?:[A-Za-z][A-Za-z0-9_-]*)?(?:\s[\s\S]*)?$/.test(text.trim())) return false;
            execute(text);
            return true;
        },
        keydown(event) {
            if (menu.hidden) return false;
            if (event.key === 'Escape') { dismissed = editor.getValue().trimStart(); hideMenu(); }
            else if (matches.length && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
                selected = (selected + (event.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length;
                updateMenu();
            } else if (matches.length && event.key === 'Tab') {
                editor.setValue('/' + matches[selected].name);
                hideMenu();
            } else if (matches.length && event.key === 'Enter' && !event.shiftKey) {
                execute('/' + matches[selected].name);
            } else return false;
            event.preventDefault();
            event.stopPropagation();
            return true;
        }
    };
};
