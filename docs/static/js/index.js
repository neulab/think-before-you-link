const burger = document.querySelector(".navbar-burger");
const menu = document.querySelector("#site-menu");

function closeMenu() {
  burger?.setAttribute("aria-expanded", "false");
  burger?.classList.remove("is-active");
  menu?.classList.remove("is-active");
}

burger?.addEventListener("click", () => {
  const expanded = burger.getAttribute("aria-expanded") === "true";
  burger.setAttribute("aria-expanded", String(!expanded));
  burger.classList.toggle("is-active", !expanded);
  menu?.classList.toggle("is-active", !expanded);
});

menu?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", closeMenu);
});

const heroExample = document.querySelector("[data-hero-animation]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

if (heroExample && !reducedMotion.matches && "IntersectionObserver" in window) {
  let isVisible = false;
  let isHovered = false;

  const updateHeroAnimation = () => {
    heroExample.classList.toggle("is-paused", !isVisible || isHovered);
  };

  const heroObserver = new IntersectionObserver(
    ([entry]) => {
      isVisible = entry.isIntersecting;
      if (isVisible) heroExample.classList.add("is-animated");
      updateHeroAnimation();
    },
    { threshold: 0.18 },
  );

  heroExample.addEventListener("mouseenter", () => {
    isHovered = true;
    updateHeroAnimation();
  });

  heroExample.addEventListener("mouseleave", () => {
    isHovered = false;
    updateHeroAnimation();
  });

  heroObserver.observe(heroExample);
}

const copyButton = document.querySelector("#copy-bibtex");
const bibtex = document.querySelector("#bibtex-code");

copyButton?.addEventListener("click", async () => {
  if (!bibtex) return;

  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1600);
  } catch {
    copyButton.textContent = "Select and copy the citation";
  }
});
