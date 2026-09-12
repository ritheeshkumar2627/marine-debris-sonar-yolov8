// Varaha landing page — interaction layer (no framework)

document.addEventListener('DOMContentLoaded', () => {
  const header    = document.getElementById('siteHeader');
  const menuBtn   = document.getElementById('menuBtn');
  const mobileMenu = document.getElementById('mobileMenu');
  const playBtn   = document.getElementById('playBtn');

  // ---- solid nav background once the page scrolls past the hero ----
  const onScroll = () => {
    if (window.scrollY > 40) {
      header.classList.add('is-scrolled');
    } else {
      header.classList.remove('is-scrolled');
    }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ---- mobile menu toggle ----
  if (menuBtn && mobileMenu) {
    menuBtn.addEventListener('click', () => {
      mobileMenu.classList.toggle('hidden');
    });

    // close mobile menu after tapping a link
    mobileMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => mobileMenu.classList.add('hidden'));
    });
  }

  // ---- detection preview "play" affordance ----
  // Swap this out for real video/stream playback when the detection
  // preview feature is wired up to the backend.
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      playBtn.classList.add('opacity-0', 'pointer-events-none');
      playBtn.setAttribute('aria-hidden', 'true');
    });
  }

  // ---- routing hook for "Upload Data" links ----
  // Currently point at /upload. Wire this up to your router/backend,
  // e.g. by replacing the href or intercepting the click below.
  document.querySelectorAll('a[href="/upload"]').forEach((link) => {
    link.addEventListener('click', (event) => {
      // Example: uncomment to intercept and route client-side instead
      // event.preventDefault();
      // window.location.href = '/upload.html';
    });
  });
});
