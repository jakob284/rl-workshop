// Coding page: page views with pager, library selectors, copy buttons.
const VIEWS = ["Libraries", "Notebooks"];
let activeView = 0;
function setView(index, scroll = true) {
  activeView = Math.max(0, Math.min(VIEWS.length - 1, Number(index)));
  document.querySelectorAll("[data-view-panel]").forEach((p) => p.classList.toggle("active", Number(p.dataset.viewPanel) === activeView));
  document.querySelectorAll(".basics-nav [data-view]").forEach((b) => b.classList.toggle("active", Number(b.dataset.view) === activeView));
  document.getElementById("view-position").textContent = `${activeView + 1} / ${VIEWS.length}`;
  document.getElementById("previous-view").disabled = activeView === 0;
  const next = document.getElementById("next-view");
  next.disabled = activeView === VIEWS.length - 1;
  next.textContent = next.disabled ? "Next" : `Next: ${VIEWS[activeView + 1]}`;
  history.replaceState(null, "", activeView === 1 ? "#notebooks" : location.pathname);
  if (scroll) window.scrollTo({ top: 0, behavior: "instant" });
}
document.querySelectorAll(".basics-nav [data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
document.getElementById("previous-view").addEventListener("click", () => setView(activeView - 1));
document.getElementById("next-view").addEventListener("click", () => setView(activeView + 1));
setView(location.hash === "#notebooks" ? 1 : 0, false);

document.querySelectorAll(".lib-switch [data-lib]").forEach((tab) => {
  tab.addEventListener("click", () => {
    const section = tab.closest(".theory-section");
    section.querySelectorAll(".lib-switch [data-lib]").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    section.querySelectorAll(".lib-panel").forEach((p) => { p.hidden = p.id !== "lib-" + tab.dataset.lib; });
  });
});

document.querySelectorAll(".copy-code").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const code = btn.closest(".code-box").querySelector("code").textContent;
    try { await navigator.clipboard.writeText(code); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select and copy"; }
    setTimeout(() => { btn.textContent = "Copy"; }, 1500);
  });
});
