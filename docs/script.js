document.documentElement.classList.add("js");

const revealItems = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -8%", threshold: 0.08 });
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("visible"));
}

const navLinks = [...document.querySelectorAll(".site-nav a")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

if ("IntersectionObserver" in window && sections.length) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  }, { rootMargin: "-28% 0px -60%", threshold: [0, 0.2, 0.5] });
  sections.forEach((section) => sectionObserver.observe(section));
}

const dialog = document.querySelector("#figure-dialog");
const dialogImage = document.querySelector("#dialog-image");
const dialogCaption = document.querySelector("#dialog-caption");
const closeButton = dialog?.querySelector(".lightbox-close");

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!dialog || typeof dialog.showModal !== "function") return;
    dialogImage.src = button.dataset.src;
    dialogImage.alt = button.querySelector("img")?.alt || "Expanded research figure";
    dialogCaption.textContent = button.dataset.caption || "";
    dialog.showModal();
  });
});

closeButton?.addEventListener("click", () => dialog.close());
dialog?.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

const copyButton = document.querySelector("#copy-bibtex");
const bibtex = document.querySelector("#bibtex-code code");
copyButton?.addEventListener("click", async () => {
  if (!bibtex) return;
  const original = copyButton.textContent;
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyButton.textContent = "Copied";
  } catch {
    copyButton.textContent = "Select citation below";
  }
  window.setTimeout(() => {
    copyButton.textContent = original;
  }, 1800);
});
