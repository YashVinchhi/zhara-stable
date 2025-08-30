# Zhara UI Specification (Reconstruction-Ready)

Purpose: This document precisely describes the current UI so another AI or developer can recreate the same interface and behavior from scratch without inspecting the repo.

Status: Matches the current shipped UI at zhara/static/index.html + css/styles.css + js/init.js + js/chat.js (dark background, three-panel layout, soft green accents, Coqui-only TTS).


1) Technology and global constraints
- Stack: HTML5, CSS (no framework), Vanilla JS (ES6+).
- External assets:
  - Fonts: Google Fonts Inter (weights 400/500/600/700).
  - Icons: Material Design Icons CDN (e.g., mdi mdi-send, mdi mdi-microphone).
  - highlight.js 11.9.0 (GitHub theme) for code blocks.
  - three.js r128 (lightweight canvas hook for avatar; no complex scene required by default).
- Content Security Policy (CSP) for index.html (meta tag):
  - default-src 'self'
  - script-src 'self' https://cdnjs.cloudflare.com
  - style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net
  - font-src https://fonts.gstatic.com https://cdn.jsdelivr.net
  - img-src 'self' data:
  - media-src 'self'
- Favicon: /static/favicon.svg and fallback /static/favicon.ico
- Files loaded:
  - /static/css/styles.css
  - /static/js/init.js (populates models and voices)
  - /static/js/chat.js (main UI logic)
  - highlight.js (cdn) and three.js (cdn)


2) High-level layout
- Full viewport, three vertical panels on a dark background with card-like surfaces.
- Left: Sessions sidebar (fixed width 280px), dark gray background.
- Center: Main chat panel (flex-1), large white card containing messages and the input.
- Right: Settings/Avatar panel (fixed width 320px), white card stack.
- Panel spacing: 18px gap between columns; 18px outer padding around whole app.
- All cards have rounded corners and soft shadows to appear floating above background.


3) Color palette and tokens (CSS variables)
- Backgrounds:
  - --bg-dark: #1f232a (app background)
  - --sidebar-bg: #262b33 (left sidebar)
  - --panel-card: #ffffff (main/right cards)
  - --panel-subtle: #f7f9fb (off-white input background section)
- Borders & text:
  - --border-color: #e6e8ec
  - --text-primary: #101418
  - --text-secondary: #475067
  - --text-muted: #8a93a6
- Bubbles:
  - --bubble-user: #e9f8f1 (subtle green tint)
  - --bubble-user-border: #bfead7
  - --bubble-bot: #f5f7fa (gentle gray)
  - --bubble-bot-border: #e6e8ec
- Accent greens:
  - --green: #34d399
  - --green-strong: #10b981
- Shadows:
  - --shadow-card: 0 10px 24px rgba(0,0,0,0.18), 0 2px 6px rgba(0,0,0,0.08)
  - --shadow-soft: 0 6px 16px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.04)
- Radii:
  - --radius: 14px
  - --radius-pill: 999px


4) Typography
- Family: Inter, system UI fallbacks.
- Base font-size: 16px.
- Weights used: 400 (normal), 500 (medium), 600 (semibold), 700 (bold).


5) DOM structure (required IDs/classes)
<html>
  <body>
    <div class="app-container">
      <!-- Left sidebar -->
      <div class="sessions-sidebar">
        <div class="sidebar-header">
          <button class="new-chat-btn">New Chat</button>
        </div>
        <div class="sessions-list"><!-- session items --></div>
        <div class="sidebar-footer"><!-- optional user info --></div>
      </div>

      <!-- Center main content -->
      <div class="main-content">
        <button class="sidebar-toggle" title="Toggle sidebar"></button>
        <div class="interaction-zone">
          <div class="chat-container"><!-- messages --></div>
          <div class="input-container initial">
            <div class="input-wrapper">
              <button class="mic-btn" title="Voice Input"></button>
              <textarea class="input-box" placeholder="Ask anything..." rows="1"></textarea>
              <button class="upload-btn" title="Upload File"></button>
              <button class="send-btn" title="Send Message"></button>
            </div>
          </div>
        </div>
      </div>

      <!-- Right panel -->
      <div class="right-panel">
        <div class="avatar-section">
          <div class="avatar-card">
            <canvas id="avatar-canvas" width="300" height="300"></canvas>
            <div class="avatar-status">Ready</div>
          </div>
          <div class="avatar-controls">
            <button id="mute-btn" class="control-btn" title="Toggle Mute"></button>
            <button id="fullscreen-btn" class="control-btn" title="Fullscreen"></button>
            <button id="avatar-toggle-btn" class="control-btn" title="Toggle Avatar"></button>
          </div>
        </div>
        <div class="model-settings">
          <h3 class="settings-title">AI Settings</h3>
          <div class="settings-row">
            <label for="model-select">LLM Model:</label>
            <select id="model-select"><option>Loading models...</option></select>
          </div>
          <div class="settings-row">
            <label for="voice-select">Voice (Coqui):</label>
            <select id="voice-select"><option value="default">Default</option></select>
          </div>
          <div class="settings-row">
            <label for="avatar-select">Avatar Style:</label>
            <select id="avatar-select">
              <option value="default">Default</option>
              <option value="professional">Professional</option>
              <option value="casual">Casual</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <!-- Hidden file input and audio element -->
    <input type="file" id="file-input" style="display:none;" accept=".txt,.pdf,.doc,.docx,.json" />
    <audio id="tts-audio" preload="none"></audio>
  </body>
</html>

Notes:
- Class and ID names are required as above for JS bindings.
- Icons are applied via <i class="mdi mdi-iconname"></i> or plain SVG; actual labels are set via title attributes.


6) Left sidebar (sessions)
- Container: .sessions-sidebar
  - Width: 280px, background: var(--sidebar-bg), border-radius: var(--radius), shadow: var(--shadow-soft).
- Header: .sidebar-header with .new-chat-btn
  - Button visuals: gradient from #a7f3d0 to #34d399, bold text, white card shadow.
- Sessions list: .sessions-list holds multiple .session-item
  - .session-item: rounded 12px, semi-transparent white on hover, transition translateY(-1px).
  - Inside: .session-content (clickable to switch), .session-actions (delete button)
  - Avatar circle: .session-avatar 32x32 white circle with initial.
  - Active session: .session-item.active has slightly stronger white overlay.
- Delete button: .delete-session-btn small rounded button (✕) with hover background.


7) Center panel (chat)
- .interaction-zone: white card, rounded var(--radius), shadow var(--shadow-card), column layout.
- .chat-container: vertical flex list with 12px gap and generous padding.
- Messages:
  - Wrapper: .message.user or .message.bot
  - Bubble: .message-content (max-width ~70%)
    - User bubble: background var(--bubble-user), border var(--bubble-user-border), aligned to right.
    - Bot bubble: background var(--bubble-bot), border var(--bubble-bot-border), aligned to left.
  - Time: .message-time (11px, muted color) below text inside bubble.
  - Code blocks inside bot messages: <pre><code> styled dark (#111827) with border #374151, rounded 10px, padding 12px; syntax highlighted by highlight.js (GitHub theme loaded via CDN).
- Typing indicator:
  - Element: <div class="message bot typing"><div class="message-content"><div class="typing-indicator">Thinking<span class="typing-dots"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span></div></div></div>
  - .typing-dot animates with simple bounce (1s ease-in-out, staggered 0.15s).

- Input area:
  - .input-container (white/off-white strip) > .input-wrapper (rounded pill, light shadow)
  - Children: .mic-btn, .input-box (textarea auto-grow up to 140px), .upload-btn, .send-btn
  - Button size: 40x40, white background, 1px border, rounded 10-12px, hover background #f3f6fa.
  - Placeholder rotates every 4s across: "Ask anything…", "Explain a concept in simple terms…", "Help me debug an error…", "Outline a plan for…".


8) Right panel (avatar + settings)
- .avatar-section: white card with .avatar-card and controls.
  - Canvas: #avatar-canvas 300x300 logical size (display scales to container), sits in a subtle card.
  - Status: .avatar-status text below canvas ("Ready", "Listening", "Thinking", "Speaking", "Disabled").
  - Controls: #mute-btn, #fullscreen-btn, #avatar-toggle-btn (40x40 buttons, like input icons).
  - Disabled state: canvas gets class .disabled (reduced opacity, grayscale) and status shows "Disabled".
- .model-settings: white card with three select rows: #model-select, #voice-select, #avatar-select.
- Headings: .settings-title (14px, bold, secondary color).


9) Responsive behavior
- Breakpoint: max-width: 1100px
  - Sidebar becomes overlay: .sessions-sidebar transforms off-canvas (translateX(-110%)) and toggles .open via button .sidebar-toggle (40x40 green button at top-left of center card).
  - Sidebar toggle appears only on small widths.


10) Data flow and endpoints (required behaviors)
- init.js responsibilities:
  - Populate #model-select via GET /models. If none, add <option value="offline">Offline</option> and select it.
  - Populate #voice-select via GET /tts/voices?provider=coqui. If none, set single option "Default".
- chat.js responsibilities:
  - Session lifecycle:
    - Load sessions via GET /api/sessions and render .session-item list.
    - Create session via POST /api/sessions {title:"New chat"} and switch.
    - Switch session via GET /api/sessions/{id} and render messages.
    - Delete via DELETE /api/sessions/{id}.
    - Suggest title via POST /api/sessions/{id}/suggest_title then PUT /api/sessions/{id} to save.
  - Messaging:
    - On send: POST /ask with { text, model: #model-select.value, session_id, tts_model: #voice-select.value || "default" }.
    - Show typing indicator while awaiting response.
    - On success: append bot message; if audio_url present, set #tts-audio.src and play; on ended, set status to Idle.
  - File upload:
    - Clicking .upload-btn triggers hidden #file-input; POST /api/stt with audio file; resulting text is appended to the input.
  - Avatar controls:
    - #mute-btn toggles active class.
    - #fullscreen-btn enters/leaves fullscreen on #avatar-canvas.
    - #avatar-toggle-btn toggles #avatar-canvas.disabled and updates status.
  - Local persistence: localStorage keys zh_model, zh_voice, zh_avatar; restored on load after init.js fills selects.


11) Accessibility and semantics
- Chat log region: .chat-container functions as role=log (optional) with polite updates.
- Buttons have title attributes.
- Selects have corresponding <label for="...">.
- Visual focus rings via browser default are acceptable; controls have 1px borders and hover feedback.


12) Animations and transitions
- Subtle button hover lifts (translateY(-1px)).
- Typing dots bounce.
- Input placeholder rotates every 4s.
- No heavy transitions; keep snappy and responsive.


13) Iconography
- Material Design Icons via CDN. Examples:
  - .send-btn: <i class="mdi mdi-send"></i>
  - .mic-btn: <i class="mdi mdi-microphone"></i>
  - .upload-btn: <i class="mdi mdi-paperclip"></i>
  - Sidebar menu: <i class="mdi mdi-menu"></i>
- Alternatively, inline SVGs may be used; visual footprint should remain identical in size and color.


14) Exact sizes and spacing (summary)
- App padding: 18px around columns; gap between columns: 18px.
- Left width: 280px; Right width: 320px; Center flex: 1.
- Buttons (mic/upload/send/control): 40x40px.
- Session avatar: 32x32px circle.
- Input textarea: min-height ~40px; auto-grows up to 140px.
- Bubbles: max-width ~70%; border radius ~14px.


15) Visual style constraints
- The overall feeling is minimal, soft, and calm. Use white/off-white cards over a dark gray backdrop.
- Accents limited to soft green for emphasis (New Chat button, small mobile toggle button).
- Shadows soft but present, to create floating cards.
- Rounded corners consistent across components.


16) Reconstruction acceptance criteria
- On desktop width (>1100px), three-column layout appears with correct widths and gaps.
- Left sidebar new chat button shows a green gradient and clearly stands out.
- Session items look like dark, rounded tiles; active item is subtly brighter.
- Center card is white; user messages right-aligned green-tinted; bot messages left-aligned light gray; code blocks in dark card-within-card style.
- Input is a frosted, rounded pill with four controls (mic, textarea, upload, send).
- Right panel shows avatar canvas card and three selects; status text changes with actions.
- Model dropdown is populated from /models; Voice dropdown from /tts/voices?provider=coqui.
- Small screens: sidebar toggles via green button and slides over the content.


17) Optional enhancements parity (non-blocking)
- highlight.js auto-detects languages; GitHub theme is applied.
- three.js canvas reserved; basic idle animation (gentle float) acceptable or disabled.


18) Non-goals
- Do not reintroduce multiple TTS providers; UI must only show Coqui voices.
- Avoid additional quick settings in the center panel; selectors live in the right panel only.


19) Example message rendering rules
- Parse triple backtick code blocks: ```lang\n...\n``` and render inside <pre><code class="language-lang"></code></pre>.
- Text outside code blocks is appended as plain text with preserved line breaks (white-space: pre-wrap).


20) Assets and paths
- CSS: /static/css/styles.css
- JS: /static/js/init.js, /static/js/chat.js
- Fonts & icons via CDN (CSP allows these hosts).


With the above, an AI should be able to generate the same HTML structure, CSS theme, and JS behaviors to reproduce the current UI exactly, including data bindings, selectors, and interaction details. 
