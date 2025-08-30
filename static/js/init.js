(function() {
  /**
   * Fetches JSON data from a URL with basic error handling.
   * @param {string} url The URL to fetch.
   * @returns {Promise<any|null>} The JSON data or null on failure.
   */
  async function fetchJSON(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn('Fetch failed for', url, e);
      return null;
    }
  }

  /**
   * Fetches available models from the backend and populates the model select dropdown.
   */
  async function populateModels() {
    const sel = document.querySelector('#model-select');
    if (!sel) return;

    sel.innerHTML = '<option value="">Loading models...</option>';
    const data = await fetchJSON('/models');
    const models = Array.isArray(data) ? data : [];

    if (!models || models.length === 0) {
      // Fallback option if the API is unavailable
      sel.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = 'offline';
      opt.textContent = 'Offline';
      sel.appendChild(opt);
      sel.value = 'offline';
      return;
    }

    sel.innerHTML = '';
    models.forEach(m => {
      if (!m || !m.name) return;
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      sel.appendChild(opt);
    });

    // Select the first model by default
    if (sel.options.length > 0) {
      sel.selectedIndex = 0;
    }
  }

  /**
   * Fetches available Coqui TTS voices and populates the voice select dropdown.
   */
  async function populateCoquiVoices() {
    const sel = document.querySelector('#voice-select');
    if (!sel) return;

    const data = await fetchJSON('/tts/voices?provider=coqui');
    const voices = Array.isArray(data) ? data : [];
    sel.innerHTML = '';

    if (!voices.length) {
      const opt = document.createElement('option');
      opt.value = 'default';
      opt.textContent = 'Default';
      sel.appendChild(opt);
      return;
    }

    voices.forEach(v => {
      if (!v || !v.name) return;
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = v.display_name || v.name;
      sel.appendChild(opt);
    });
  }

  /**
   * Replaces <i data-lucide> elements with local SVG <img> tags if a matching file exists.
   * This improves performance and allows for offline icons.
   */
  function replaceLucideWithLocalIcons() {
    const fallbackMap = {
      'paperclip': 'upload',
      'plus': 'check',
      'x': 'close',
      'X': 'close',
      'send': 'play',
      'volume-2': 'speaker',
      'maximize-2': 'settings',
      'video': 'play'
    };

    document.querySelectorAll('[data-lucide]').forEach(el => {
      try {
        const name = el.getAttribute('data-lucide');
        if (!name) return;

        const normalize = (n) => n.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9_-]/g, '');
        const candidateNames = [name, fallbackMap[name] || null].filter(Boolean).map(normalize);

        // This function tries to load each candidate icon name in order.
        function tryNext(index) {
          if (index >= candidateNames.length) {
            return; // No local icon found
          }
          const iconName = candidateNames[index];
          // Use the static path where icons are served from the app (was /icons/... which caused 404s)
          const url = `/static/icons/${iconName}.svg`;
          const imgTest = new Image();

          imgTest.onload = function() {
            // Success: Icon exists. Replace the placeholder with an <img> tag.
            const img = document.createElement('img');
            img.src = url;
            img.alt = name;
            img.width = el.getAttribute('data-icon-width') || 20;
            img.height = el.getAttribute('data-icon-height') || 20;

            // Preserve classes and common accessibility attributes and other attributes except data-lucide
            if (el.className) img.className = el.className;
            Array.from(el.attributes).forEach(attr => {
              if (attr.name === 'data-lucide') return;
              if (attr.name === 'class') return; // already copied
              // don't overwrite src/alt/width/height we just set
              if (['src','alt','width','height'].includes(attr.name)) return;
              try { img.setAttribute(attr.name, attr.value); } catch (e) {}
            });

            el.replaceWith(img);
          };

          imgTest.onerror = function() {
            // Failure: Try the next candidate name.
            tryNext(index + 1);
          };

          imgTest.src = url;
        }

        tryNext(0); // Start checking with the first candidate name.

      } catch (e) {
        console.warn('Failed to replace lucide icon:', e);
      }
    });
  }

  /**
   * Sets up the logic for the settings modal window.
   */
  function setupSettingsModal() {
    const settingsBtn = document.querySelector('.settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');

    if (settingsBtn && settingsModal && closeModalBtn) {
      settingsBtn.addEventListener('click', function() {
        settingsModal.style.display = 'flex';
      });
      closeModalBtn.addEventListener('click', function() {
        settingsModal.style.display = 'none';
      });
      // Close modal by clicking on the background overlay
      settingsModal.addEventListener('click', function(e) {
        if (e.target === settingsModal) {
          settingsModal.style.display = 'none';
        }
      });
      // Close modal with the Escape key
      document.addEventListener('keydown', function(e) {
        if (settingsModal.style.display === 'flex' && e.key === 'Escape') {
          settingsModal.style.display = 'none';
        }
      });
    }
  }

  /**
   * Makes the right-panel horizontally resizable.
   */
  function setupRightPanelResizer() {
    const resizer = document.getElementById('right-resizer');
    const root = document.documentElement;
    const rightPanel = document.querySelector('.right-panel');
    const canvas = document.getElementById('avatar-canvas');
    if (!resizer || !rightPanel) return;

    const DEFAULT_WIDTH = 320;
    const MIN_WIDTH = 200;

    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

    function getMaxWidth() {
      const winWidth = window.innerWidth;
      const leftSidebarWidth = 280;
      const minMainContentWidth = 420;
      const safetyMargin = 40;
      return Math.max(MIN_WIDTH, winWidth - leftSidebarWidth - minMainContentWidth - safetyMargin);
    }

    function resizeCanvasToDisplay() {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      // Ensure size is at least 1px to avoid errors
      const size = Math.max(1, Math.floor(rect.width * (window.devicePixelRatio || 1)));
      if (canvas.width !== size) {
        canvas.width = size;
        canvas.height = size;
        canvas.style.height = rect.width + 'px';
        try {
          if (window.avatarRenderer && typeof window.avatarRenderer.setSize === 'function') {
            window.avatarRenderer.setSize(rect.width, rect.width);
          }
        } catch (e) {}
      }
    }

    function setWidth(w) {
      const clamped = clamp(Math.round(w), MIN_WIDTH, getMaxWidth());
      root.style.setProperty('--right-panel-width', clamped + 'px');
      rightPanel.style.width = clamped + 'px'; // Set inline for immediate layout
      resizeCanvasToDisplay();
    }

    let dragging = false;
    let startX = 0;
    let startWidth = 0;

    function onPointerMove(clientX) {
      if (!dragging) return;
      const delta = clientX - startX;
      const newWidth = startWidth + delta;
      setWidth(newWidth);
    }

    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      try {
        resizer.releasePointerCapture(e.pointerId);
      } catch (err) {}
    }

    // Pointer events for mouse/touch/pen
    resizer.addEventListener('pointerdown', (e) => {
      dragging = true;
      startX = e.clientX;
      startWidth = parseInt(getComputedStyle(root).getPropertyValue('--right-panel-width')) || rightPanel.clientWidth || DEFAULT_WIDTH;
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
      try {
        resizer.setPointerCapture(e.pointerId);
      } catch (err) {}
      e.preventDefault();
    });

    document.addEventListener('pointermove', (e) => onPointerMove(e.clientX));
    document.addEventListener('pointerup', (e) => endDrag(e));
    document.addEventListener('pointercancel', (e) => endDrag(e));

    // Keyboard support for accessibility
    resizer.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 40 : 10;
      const current = parseInt(getComputedStyle(root).getPropertyValue('--right-panel-width')) || DEFAULT_WIDTH;
      if (e.key === 'ArrowLeft') setWidth(current - step);
      else if (e.key === 'ArrowRight') setWidth(current + step);
      else if (e.key === 'Home') setWidth(MIN_WIDTH);
      else if (e.key === 'End') setWidth(getMaxWidth());
      else if (e.key === 'r' || e.key === 'R') setWidth(DEFAULT_WIDTH);
      else return;
      e.preventDefault();
    });

    // Double-click to reset
    resizer.addEventListener('dblclick', () => setWidth(DEFAULT_WIDTH));

    // Window resize: clamp width and resize canvas
    window.addEventListener('resize', () => {
      const current = parseInt(getComputedStyle(root).getPropertyValue('--right-panel-width')) || DEFAULT_WIDTH;
      setWidth(current); // This will re-clamp the width based on the new max width
    });

    // Initial sync on load
    setTimeout(() => {
      setWidth(parseInt(getComputedStyle(root).getPropertyValue('--right-panel-width')) || DEFAULT_WIDTH);
    }, 50);
  }

  /**
   * Main initialization function that runs when the DOM is ready.
   */
  async function main() {
    // Populate dropdowns from APIs
    await Promise.all([
      populateModels(),
      populateCoquiVoices()
    ]);

    // Set up UI components
    replaceLucideWithLocalIcons();
    setupSettingsModal();
    setupRightPanelResizer();

    // Initialize Lucide icons: try immediately, then retry a few times in case the lucide script
    // is loaded after this script (index.html loads lucide after init.js by default).
    (function initLucide(retries = 20, interval = 100) {
      const attempt = () => {
        if (window.lucide) {
          try {
            if (typeof lucide.createIcons === 'function') {
              lucide.createIcons();
            } else if (typeof lucide.replace === 'function') {
              lucide.replace();
            }
          } catch (e) {
            console.warn('Lucide init error', e);
          }
          return;
        }
        if (retries-- > 0) {
          setTimeout(attempt, interval);
        }
      };
      attempt();
    })();
  }

  // Run the main function after the document is loaded.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }

})();