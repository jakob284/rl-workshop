// Coding page: library selector and copy buttons for code examples.
document.querySelectorAll(".toolkit-tabs [data-lib]").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".toolkit-tabs [data-lib]").forEach((t) => {
      const on = t === tab;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on);
    });
    document.querySelectorAll(".toolkit-panel").forEach((p) => { p.hidden = p.id !== "lib-" + tab.dataset.lib; });
  });
});
document.querySelectorAll(".copy-code").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const code = btn.closest(".toolkit-code").querySelector("code").textContent;
    try { await navigator.clipboard.writeText(code); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select and copy"; }
    setTimeout(() => { btn.textContent = "Copy"; }, 1500);
  });
});
