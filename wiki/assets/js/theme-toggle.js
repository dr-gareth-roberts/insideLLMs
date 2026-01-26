(() => {
  const STORAGE_KEY = "insidellms.docs.theme";
  const THEMES = ["light", "dark"];

  const safeGet = (key) => {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const safeSet = (key, value) => {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      // Ignore storage failures (e.g. in private browsing).
    }
  };

  const normalizeTheme = (value) => (THEMES.includes(value) ? value : null);

  const systemTheme = () =>
    window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ? "dark" : "light";

  const getInitialTheme = () => normalizeTheme(safeGet(STORAGE_KEY)) ?? systemTheme();

  const setTheme = (theme) => {
    const normalized = normalizeTheme(theme);
    if (!normalized) {
      return;
    }

    if (window.jtd && typeof window.jtd.setTheme === "function") {
      window.jtd.setTheme(normalized);
      return;
    }

    const link = document.querySelector('link[rel="stylesheet"][href*="just-the-docs"]');
    if (!link) {
      return;
    }

    link.href = link.href.replace(
      /just-the-docs(-[^/]+)?\.css(\?.*)?$/,
      `just-the-docs-${normalized}.css$2`,
    );
  };

  const updateToggle = (button, theme) => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    const label = nextTheme === "dark" ? "Dark mode" : "Light mode";

    button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    button.setAttribute("aria-label", `Switch to ${nextTheme} mode`);
    button.title = `Switch to ${nextTheme} mode`;

    const labelEl = button.querySelector(".insidellms-theme-toggle__label");
    if (labelEl) {
      labelEl.textContent = label;
    } else {
      button.textContent = label;
    }
  };

  const init = () => {
    let theme = getInitialTheme();
    setTheme(theme);

    const button = document.querySelector("[data-insidellms-theme-toggle]");
    if (!button) {
      return;
    }

    updateToggle(button, theme);

    button.addEventListener("click", () => {
      theme = theme === "dark" ? "light" : "dark";
      safeSet(STORAGE_KEY, theme);
      setTheme(theme);
      updateToggle(button, theme);
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
