class ChatUI {
  constructor() {
    // State
    this.initialized = false;
    this.chatHistory = [];
    this.currentAudio = null;
    this.currentSessionId = null;
    this.placeholderTimer = null;

    // Elements
    this.initializeElements();

    // Events
    this.setupEventListeners();

    // 3D (optional/no-op)
    this.setupThreeJS();

    // Data
    this.loadSessions();
    // Removed unconditional createNewSession(); loadSessions() handles creating when none exist

    // UX
    this.setupDynamicPlaceholder();
  }

  initializeElements() {
    this.inputContainer = document.querySelector('.input-container');
    this.chatContainer = document.querySelector('.chat-container');
    this.inputBox = document.querySelector('.input-box');
    this.fileInput = document.querySelector('#file-input');
    this.modelSelect = document.querySelector('#model-select');
    this.voiceSelect = document.querySelector('#voice-select');
    this.avatarSelect = document.querySelector('#avatar-select');
    this.sendBtn = document.querySelector('.send-btn');

    // Sidebar and sessions
    this.sidebar = document.querySelector('.sessions-sidebar');
    this.sessionsList = document.querySelector('.sessions-list');
    this.newChatBtn = document.querySelector('.new-chat-btn');
    this.sidebarToggle = document.querySelector('.sidebar-toggle');
  }

  setupEventListeners() {
    if (this.inputBox) {
      this.inputBox.addEventListener('focus', () => this.transitionToChat());
      this.inputBox.addEventListener('input', () => { this.autoGrowInput(); this.updateSendButtonState(); });
      this.inputBox.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
      this.inputBox.addEventListener('blur', () => this.setupDynamicPlaceholder());
    }

    if (this.fileInput) {
      this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    }

    if (this.modelSelect) {
      this.modelSelect.addEventListener('change', () => this.handleModelChange());
    }

    if (this.voiceSelect) {
      this.voiceSelect.addEventListener('change', () => this.handleVoiceChange());
    }

    if (this.avatarSelect) {
      this.avatarSelect.addEventListener('change', () => this.handleAvatarChange());
    }

    if (this.newChatBtn) {
      this.newChatBtn.addEventListener('click', () => this.createNewSession());
    }

    if (this.sidebarToggle) {
      this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
    }

    const sendBtn = this.sendBtn || document.querySelector('.send-btn');
    if (sendBtn) sendBtn.addEventListener('click', () => this.sendMessage());

    const micBtn = document.querySelector('.mic-btn');
    if (micBtn) micBtn.addEventListener('click', () => this.handleVoiceInput());

    // Avatar controls
    const muteBtn = document.getElementById('mute-btn');
    if (muteBtn) muteBtn.addEventListener('click', () => this.toggleMute());
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (fullscreenBtn) fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
    const avatarToggleBtn = document.getElementById('avatar-toggle-btn');
    if (avatarToggleBtn) avatarToggleBtn.addEventListener('click', () => this.toggleAvatarEnabled());

    const uploadBtn = document.querySelector('.upload-btn');
    if (uploadBtn) uploadBtn.addEventListener('click', () => this.fileInput?.click());

    // Initial state for send button
    this.updateSendButtonState();
  }

  updateSendButtonState() {
    const btn = this.sendBtn || document.querySelector('.send-btn');
    if (!btn) return;
    const hasText = (this.inputBox?.value || '').trim().length > 0;
    if (hasText) {
      btn.classList.add('enabled');
      btn.removeAttribute('disabled');
    } else {
      btn.classList.remove('enabled');
      btn.setAttribute('disabled', 'true');
    }
  }

  // ------- UI helpers -------
  autoGrowInput() {
    if (!this.inputBox) return;
    this.inputBox.style.height = 'auto';
    this.inputBox.style.height = Math.min(this.inputBox.scrollHeight, 140) + 'px';
  }

  toggleSidebar() {
    if (this.sidebar) this.sidebar.classList.toggle('open');
  }

  transitionToChat() {
    if (this.initialized || !this.inputContainer) return;
    this.inputContainer.classList.remove('initial');
    this.inputContainer.classList.add('chat');
    this.initialized = true;
  }

  setupThreeJS() {
    // Optional hook; keep minimal for CSP and footprint
    // You can add avatar rendering here if needed
  }

  setupDynamicPlaceholder() {
    if (!this.inputBox) return;
    const ideas = [
      'Ask anything…',
      'Explain a concept in simple terms…',
      'Help me debug an error…',
      'Outline a plan for…'
    ];
    let i = 0;
    clearInterval(this.placeholderTimer);
    this.placeholderTimer = setInterval(() => {
      i = (i + 1) % ideas.length;
      this.inputBox.placeholder = ideas[i];
    }, 4000);
  }

  updateAvatarStatus(mode) {
    const statusEl = document.querySelector('.avatar-status');
    if (!statusEl) return;
    const map = { listening: 'Listening', thinking: 'Thinking', speaking: 'Speaking', idle: 'Ready', disabled: 'Disabled' };
    statusEl.textContent = map[mode] || 'Ready';
  }

  toggleAvatarEnabled() {
    const canvas = document.getElementById('avatar-canvas');
    const disabled = canvas?.classList.toggle('disabled');
    this.updateAvatarStatus(disabled ? 'disabled' : 'idle');
  }

  toggleMute() {
    const btn = document.getElementById('mute-btn');
    btn?.classList.toggle('active');
  }

  toggleFullscreen() {
    const canvas = document.getElementById('avatar-canvas');
    if (!canvas) return;
    if (!document.fullscreenElement) canvas.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  // ------- Sessions -------
  async loadSessions() {
    try {
      const res = await fetch('/api/sessions');
      const arr = await res.json();
      const sessions = Array.isArray(arr) ? arr : [];
      this.renderSessionsList(sessions);
      // Auto-select latest session if available; otherwise create one
      if (sessions.length > 0) {
        // Pick the most recently updated session
        const sorted = [...sessions].sort((a,b) => new Date(b.last_updated) - new Date(a.last_updated));
        const target = sorted[0];
        if (!this.currentSessionId || this.currentSessionId !== target.session_id) {
          await this.switchToSession(target.session_id);
        }
      } else {
        await this.createNewSession();
      }
    } catch (e) {
      console.warn('loadSessions failed', e);
    }
  }

  async createNewSession() {
    try {
      const res = await fetch('/api/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'New chat' }) });
      const session = await res.json();
      this.currentSessionId = session.session_id;
      await this.switchToSession(session.session_id);
      await this.loadSessions();
    } catch (e) {
      console.warn('createNewSession failed', e);
    }
  }

  async switchToSession(sessionId) {
    if (!sessionId || this.currentSessionId === sessionId) return;
    this.currentSessionId = sessionId;
    this.chatHistory = [];
    this.clearChatContainer();
    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      const data = await res.json();
      const messages = data?.messages || [];
      for (const m of messages) {
        if (m.user_message) this.addMessageToChat(m.user_message, 'user');
        if (m.ai_response) this.addMessageToChat(m.ai_response, 'bot');
      }
      this.updateActiveSession(sessionId);
    } catch (e) {
      console.warn('switchToSession failed', e);
    }
  }

  async deleteSession(sessionId) {
    if (!sessionId) return;
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      if (this.currentSessionId === sessionId) await this.createNewSession();
      await this.loadSessions();
    } catch (e) {
      console.warn('deleteSession failed', e);
    }
  }

  renderSessionsList(sessions) {
    if (!this.sessionsList) return;
    this.sessionsList.innerHTML = '';
    sessions.forEach(s => {
      const item = document.createElement('div');
      item.className = `session-item ${s.session_id === this.currentSessionId ? 'active' : ''}`;
      const initial = (s.title || 'N')[0]?.toUpperCase() || 'N';
      const last = s.last_updated ? new Date(s.last_updated).toLocaleDateString() : '';
      item.innerHTML = `
        <div class="session-content" data-id="${s.session_id}">
          <div class="session-avatar">${initial}</div>
          <div class="session-texts">
            <div class="session-title">${s.title}</div>
            <div class="session-info">${s.message_count || 0} messages · ${last}</div>
          </div>
        </div>
        <div class="session-actions">
          <button class="delete-session-btn" title="More" aria-label="More options"><i data-lucide="X" aria-hidden="false"></i></button>
        </div>`;
      item.querySelector('.session-content')?.addEventListener('click', () => this.switchToSession(s.session_id));
      item.querySelector('.delete-session-btn')?.addEventListener('click', (e) => { e.stopPropagation(); this.deleteSession(s.session_id); });
      this.sessionsList.appendChild(item);
    });
    try {
      if (window.lucide) {
        if (typeof lucide.createIcons === 'function') lucide.createIcons();
        else if (typeof lucide.replace === 'function') lucide.replace();
      }
    } catch {}
  }

  updateActiveSession(sessionId) {
    this.sessionsList?.querySelectorAll('.session-item').forEach(el => {
      el.classList.toggle('active', el.querySelector('.session-content')?.getAttribute('data-id') === sessionId);
    });
  }

  clearChatContainer() {
    if (this.chatContainer) this.chatContainer.innerHTML = '';
  }

  // ------- Messaging -------
  async sendMessage() {
    const text = (this.inputBox?.value || '').trim();
    if (!text) return;
    if (!this.modelSelect?.value) {
      this.addMessageToChat('Please select a model first.', 'bot', true);
      return;
    }

    // Ensure session
    if (!this.currentSessionId) await this.createNewSession();

    // UI updates
    this.inputBox.value = '';
    this.autoGrowInput();
    this.updateSendButtonState();
    this.addMessageToChat(text, 'user');
    this.showTypingIndicator();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model: this.modelSelect.value,
          session_id: this.currentSessionId,
          tts_model: this.voiceSelect?.value || 'default'
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.removeTypingIndicator();
      if (data.reply) {
        this.addMessageToChat(data.reply, 'bot');
        if (data.audio_url) await this.playAudio(data.audio_url, data.viseme_url);
        await this.updateSessionAfterMessage();
        this.suggestTitleAfterReply().catch(() => {});
      } else {
        throw new Error('No reply received');
      }
    } catch (e) {
      console.error('sendMessage failed', e);
      this.removeTypingIndicator();
      this.addMessageToChat(`Error: ${e.message}`, 'bot', true);
    }
  }

  showTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message bot typing';
    row.innerHTML = `<div class="message-content"><div class="typing-indicator">Thinking<span class="typing-dots"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span></div></div>`;
    this.chatContainer?.appendChild(row);
    this.chatContainer?.scrollTo({ top: this.chatContainer.scrollHeight });
  }

  removeTypingIndicator() {
    this.chatContainer?.querySelectorAll('.message.typing').forEach(el => el.remove());
  }

  addMessageToChat(text, type, isError = false) {
    this.removeTypingIndicator();
    const wrap = document.createElement('div');
    wrap.className = `message ${type} ${isError ? 'error' : ''}`;
    const content = document.createElement('div');
    content.className = 'message-content';
    const body = document.createElement('div');
    body.className = 'message-text';
    this.renderMessageText(text, body);
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString();
    content.appendChild(body);
    content.appendChild(time);
    wrap.appendChild(content);
    this.chatContainer?.appendChild(wrap);
    this.chatContainer?.scrollTo({ top: this.chatContainer.scrollHeight });
  }

  renderMessageText(text, container) {
    // Parse triple backtick code blocks and render nicely
    const regex = /```(\w+)?\n([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = regex.exec(text)) !== null) {
      const [full, lang, code] = m;
      const before = text.slice(last, m.index);
      if (before) container.appendChild(document.createTextNode(before));
      const pre = document.createElement('pre');
      const codeEl = document.createElement('code');
      if (lang) codeEl.className = `language-${lang}`;
      codeEl.textContent = code.trim();
      pre.appendChild(codeEl);
      container.appendChild(pre);
      try { window.hljs && window.hljs.highlightElement(codeEl); } catch {}
      last = m.index + full.length;
    }
    const rest = text.slice(last);
    if (rest) container.appendChild(document.createTextNode(rest));
  }

  async playAudio(url /* , visemeUrl */) {
    try {
      const audio = document.getElementById('tts-audio');
      if (!audio) return;
      audio.src = url;
      this.currentAudio = audio;
      this.updateAvatarStatus('speaking');
      await audio.play();
      audio.onended = () => this.updateAvatarStatus('idle');
    } catch (e) {
      console.warn('Audio playback failed', e);
      this.updateAvatarStatus('idle');
    }
  }

  async updateSessionAfterMessage() {
    try {
      await this.loadSessions();
    } catch {}
  }

  async suggestTitleAfterReply() {
    try {
      if (!this.currentSessionId) return;
      const res = await fetch(`/api/sessions/${this.currentSessionId}/suggest_title`, { method: 'POST' });
      if (!res.ok) return;
      const { title } = await res.json();
      if (!title) return;
      await fetch(`/api/sessions/${this.currentSessionId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
      await this.loadSessions();
    } catch {}
  }

  // ------- Inputs and helpers -------
  handleModelChange() {
    // Persist selection
    try { localStorage.setItem('zh_model', this.modelSelect?.value || ''); } catch {}
  }

  handleVoiceChange() {
    try { localStorage.setItem('zh_voice', this.voiceSelect?.value || 'default'); } catch {}
  }

  handleAvatarChange() {
    try { localStorage.setItem('zh_avatar', this.avatarSelect?.value || 'default'); } catch {}
  }

  async handleFileUpload(e) {
    const file = e?.target?.files?.[0];
    if (!file) return;
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/stt', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const text = data?.text || '';
      if (text && this.inputBox) {
        this.inputBox.value = (this.inputBox.value + ' ' + text).trim();
        this.autoGrowInput();
        this.updateSendButtonState();
      }
    } catch (err) {
      this.addMessageToChat('Could not transcribe audio.', 'bot', true);
    } finally {
      // reset input
      e.target.value = '';
    }
  }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  // Restore saved selections after init.js populates dropdowns
  const restoreSelections = () => {
    const modelSel = document.querySelector('#model-select');
    const voiceSel = document.querySelector('#voice-select');
    try {
      const savedModel = localStorage.getItem('zh_model');
      if (savedModel && modelSel) {
        const opt = Array.from(modelSel.options).find(o => o.value === savedModel);
        if (opt) modelSel.value = savedModel;
      }
      const savedVoice = localStorage.getItem('zh_voice');
      if (savedVoice && voiceSel) {
        const opt = Array.from(voiceSel.options).find(o => o.value === savedVoice);
        if (opt) voiceSel.value = savedVoice;
      }
    } catch {}
  };

  // Wait a tick to let init.js populate selects
  setTimeout(() => {
    restoreSelections();
    // Initialize UI after restores
    new ChatUI();
  }, 300);
});
