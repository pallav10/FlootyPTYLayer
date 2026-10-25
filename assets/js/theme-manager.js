(() => {
    'use strict';

    const paletteKeys = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
        'brightBlack', 'brightRed', 'brightGreen', 'brightYellow', 'brightBlue',
        'brightMagenta', 'brightCyan', 'brightWhite'];
    const colorKeys = ['background', 'foreground', 'cursor', 'cursorAccent',
        'selectionBackground', 'selectionForeground', ...paletteKeys];
    const cssName = key => key.replace(/[A-Z]/g, letter => '-' + letter.toLowerCase());
    const rgb = color => [1, 3, 5].map(offset => parseInt(color.slice(offset, offset + 2), 16));
    const hex = channels => '#' + channels.map(value => Math.round(value).toString(16).padStart(2, '0')).join('');
    const mix = (first, second, amount) => hex(rgb(first).map((value, index) =>
        value * (1 - amount) + rgb(second)[index] * amount));
    const luminance = color => rgb(color).map(value => {
        const channel = value / 255;
        return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
    }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    const contrast = (first, second) => {
        const a = luminance(first), b = luminance(second);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    };

    let state = null;
    let current = {};
    let busy = false;
    let errorMessage = '';

    function showError(message) {
        errorMessage = String(message || '');
        const error = document.getElementById('themeError');
        if (error) {
            error.textContent = errorMessage;
            error.hidden = !errorMessage;
        }
    }

    function renderControls() {
        const select = document.getElementById('themeSelect');
        const reload = document.getElementById('themeReload');
        if (select) {
            select.replaceChildren();
            if (state) {
                for (const name of state.themes) {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    select.appendChild(option);
                }
                select.value = state.name;
                select.title = 'Theme configuration: ' + state.configPath;
            }
            select.disabled = busy || !state;
            select.setAttribute('aria-busy', String(busy));
        }
        if (reload) reload.disabled = busy;
        showError(errorMessage);
    }

    function apply(next) {
        if (!next || typeof next.name !== 'string' || !Array.isArray(next.themes) ||
            !next.themes.every(name => typeof name === 'string') || !next.theme ||
            !colorKeys.every(key => /^#[0-9a-f]{6}$/i.test(next.theme[key]))) {
            throw new Error('Invalid theme state received');
        }
        const theme = Object.fromEntries(colorKeys.map(key => [key, next.theme[key]]));
        const root = document.documentElement;
        const setColor = (name, color) => {
            root.style.setProperty('--' + name, color);
            root.style.setProperty('--' + name + '-rgb', rgb(color).join(', '));
        };
        // ANSI bright colors can be nearly white on light themes; chrome text
        // keeps their hue but moves toward the theme's most legible ink.
        const ink = [theme.foreground, ...paletteKeys.map(key => theme[key])]
            .reduce((best, color) => contrast(color, theme.background) > contrast(best, theme.background) ? color : best);
        const legible = color => {
            for (let amount = 0; amount <= 10; amount++) {
                const candidate = mix(color, ink, amount / 10);
                if (contrast(candidate, theme.background) >= 4.5) return candidate;
            }
            return ink;
        };
        setColor('term-bg', theme.background);
        setColor('term-fg', theme.foreground);
        setColor('term-muted', legible(mix(theme.background, theme.foreground, 0.65)));
        setColor('term-cursor', theme.cursor);
        setColor('term-cursor-accent', theme.cursorAccent);
        setColor('term-selection-bg', theme.selectionBackground);
        setColor('term-selection-fg', theme.selectionForeground);
        for (const key of paletteKeys) {
            setColor('term-' + cssName(key), theme[key]);
            setColor('ui-' + cssName(key), legible(theme[key]));
        }
        root.style.colorScheme = luminance(theme.background) > 0.179 ? 'light' : 'dark';
        current = theme;
        state = { ...next, theme };
        for (const info of Object.values(window.flootyTerminals || {})) {
            if (info.term) info.term.options.theme = current;
        }
        showError(next.error || '');
        renderControls();
        return state;
    }

    async function request(binding, payload) {
        if (busy) throw new Error('A theme change is already in progress');
        busy = true;
        renderControls();
        try {
            if (typeof window.callJulia !== 'function') throw new Error('Native theme controls are not available in this browser');
            const response = await window.callJulia(binding, JSON.stringify(payload));
            if (!response || !response.success || response.error) {
                throw new Error(response && response.error || 'Unable to load theme');
            }
            return apply(response);
        } catch (error) {
            showError(error.message || error);
            throw error;
        } finally {
            busy = false;
            renderControls();
        }
    }

    window.FlootyThemeManager = {
        get current() { return current; },
        apply,
        reload: () => request('flootyThemeGet', {}),
        select: name => request('flootyThemeSet', { name })
    };

    // theme.js is synchronous and server-generated; native RPC is needed only
    // for explicit changes, never for the first paint or browser-only startup.
    try {
        apply(window.FLOOTY_THEME_STATE);
    } catch (error) {
        showError(error.message || error);
    }
    document.addEventListener('DOMContentLoaded', () => {
        renderControls();
        const select = document.getElementById('themeSelect');
        const reload = document.getElementById('themeReload');
        // request() already displays failures and restores the selected value.
        if (select) select.addEventListener('change', () => window.FlootyThemeManager.select(select.value).catch(() => {}));
        if (reload) reload.addEventListener('click', () => window.FlootyThemeManager.reload().catch(() => {}));
    });
})();
