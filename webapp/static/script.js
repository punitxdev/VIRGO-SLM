document.addEventListener("DOMContentLoaded", () => {
    const chatContainer = document.getElementById("chat-container");
    const chatHistory = document.getElementById("chat-history");
    const promptInput = document.getElementById("prompt-input");
    const sendBtn = document.getElementById("send-btn");
    const stopBtn = document.getElementById("stop-btn");
    const tempSlider = document.getElementById("temperature");
    const tempVal = document.getElementById("temp-val");
    const topkSlider = document.getElementById("top-k");
    const topkVal = document.getElementById("topk-val");
    const toppSlider = document.getElementById("top-p");
    const toppVal = document.getElementById("topp-val");
    const repSlider = document.getElementById("rep-penalty");
    const repVal = document.getElementById("rep-val");
    const modeGreedyBtn = document.getElementById("mode-greedy-btn");
    const modeSampleBtn = document.getElementById("mode-sample-btn");
    const sampleModeVal = document.getElementById("sample-mode-val");
    const maxTokensSlider = document.getElementById("max-tokens-slider");
    const maxTokensInput = document.getElementById("max-tokens");
    const tokensVal = document.getElementById("tokens-val");
    const welcomeScreen = document.getElementById("welcome-screen");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const sidebar = document.getElementById("sidebar");
    const mobileOverlay = document.getElementById("mobile-overlay");
    const newChatBtn = document.getElementById("new-chat-btn");
    const headerNewBtn = document.getElementById("header-new-btn");
    const clearHistoryBtn = document.getElementById("clear-history-btn");
    const themeToggle = document.getElementById("theme-toggle");
    const charCounter = document.getElementById("input-char-counter");
    const activeSessionTitle = document.getElementById("active-session-title");

    let chatHistoryData = [];
    let isGenerating = false;

    function getIconSrc(isLight) {
        return isLight ? "logo_icon_dark.png?v=18" : "logo_icon_white.png?v=18";
    }

    function updateLogoTheme() {
        const isLight = document.body.classList.contains("light-theme");
        const logoBanner = document.querySelector(".welcome-logo-banner");
        if (logoBanner) {
            logoBanner.src = isLight ? "logo_full_dark.png?v=18" : "logo_full_white.png?v=18";
        }
        const brandLogo = document.querySelector(".brand-logo-img");
        if (brandLogo) {
            brandLogo.src = getIconSrc(isLight);
        }
        document.querySelectorAll(".avatar-logo-img").forEach(img => {
            img.src = getIconSrc(isLight);
        });
    }

    // Theme Preference & Handler
    const savedTheme = localStorage.getItem("virgo-theme");
    if (savedTheme === "light") {
        document.body.classList.add("light-theme");
    }
    updateLogoTheme();

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            document.body.classList.toggle("light-theme");
            const isLight = document.body.classList.contains("light-theme");
            localStorage.setItem("virgo-theme", isLight ? "light" : "dark");
            updateLogoTheme();
        });
    }

    // New Chat Action
    function startNewChat() {
        chatHistory.innerHTML = "";
        chatHistoryData = [];
        if (welcomeScreen) {
            chatHistory.appendChild(welcomeScreen);
            welcomeScreen.style.display = "flex";
        }
        if (activeSessionTitle) {
            activeSessionTitle.textContent = "Current Conversation";
        }
    }

    if (newChatBtn) newChatBtn.addEventListener("click", startNewChat);
    if (headerNewBtn) headerNewBtn.addEventListener("click", startNewChat);
    if (clearHistoryBtn) clearHistoryBtn.addEventListener("click", startNewChat);

    // Mobile Sidebar Drawer
    const mobileCloseBtn = document.getElementById("mobile-close-btn");
    function closeMobileSidebar() {
        sidebar.classList.remove("open");
        if (mobileOverlay) mobileOverlay.classList.remove("active");
    }

    if (mobileCloseBtn) mobileCloseBtn.addEventListener("click", closeMobileSidebar);
    if (mobileOverlay) mobileOverlay.addEventListener("click", closeMobileSidebar);

    // Sidebar Toggle Handler
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener("click", () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.toggle("open");
                if (mobileOverlay) mobileOverlay.classList.toggle("active");
            } else {
                sidebar.classList.toggle("collapsed");
            }
        });
    }

    // Temperature Slider Listener
    if (tempSlider && tempVal) {
        tempSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            tempVal.textContent = val.toFixed(2);
        });
    }

    function updateTokenPillActiveState(val) {
        const tokenPillBtns = document.querySelectorAll(".token-pill-btn");
        tokenPillBtns.forEach(btn => {
            if (btn.dataset.tokens === String(val)) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }

    function setMaxTokensValue(val) {
        const numVal = Math.min(Math.max(parseInt(val, 10) || 128, 1), 2048);
        if (maxTokensSlider) maxTokensSlider.value = Math.min(numVal, 1024);
        if (maxTokensInput) maxTokensInput.value = numVal;
        if (tokensVal) tokensVal.textContent = numVal;
        updateTokenPillActiveState(numVal);
    }

    if (maxTokensSlider) {
        maxTokensSlider.addEventListener("input", (e) => {
            setMaxTokensValue(e.target.value);
        });
    }

    if (maxTokensInput) {
        maxTokensInput.addEventListener("input", (e) => {
            setMaxTokensValue(e.target.value);
        });
        maxTokensInput.addEventListener("change", (e) => {
            setMaxTokensValue(e.target.value);
        });
    }

    const tokenPillBtns = document.querySelectorAll(".token-pill-btn");
    tokenPillBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.dataset && btn.dataset.tokens) {
                setMaxTokensValue(btn.dataset.tokens);
            }
        });
    });

    // Model Selection Integration
    const modelSelect = document.getElementById("model-select");
    const topbarModelName = document.getElementById("topbar-model-name");
    const modelLoadingIndicator = document.getElementById("model-loading-indicator");
    const modelStatusTag = document.getElementById("model-status-tag");

    async function loadModels() {
        if (!modelSelect) return;
        try {
            const response = await fetch("/api/models");
            if (!response.ok) throw new Error("Failed to fetch models");
            const data = await response.json();
            
            modelSelect.innerHTML = "";
            let hasAvailableModel = false;
            
            data.models.forEach(m => {
                const option = document.createElement("option");
                option.value = m.id;
                const loadedStr = m.loaded ? " (Loaded)" : "";
                const availStr = !m.available ? " (Missing)" : "";
                option.textContent = `${m.name}${loadedStr}${availStr}`;
                option.disabled = !m.available;
                
                if (m.active) {
                    option.selected = true;
                    if (topbarModelName) topbarModelName.textContent = m.name;
                }
                if (m.available) hasAvailableModel = true;
                modelSelect.appendChild(option);
            });
            
            modelSelect.disabled = !hasAvailableModel;
            if (modelStatusTag) {
                modelStatusTag.textContent = "Ready";
                modelStatusTag.className = "status-tag online";
            }
            
        } catch (error) {
            console.error("Error loading models:", error);
            modelSelect.innerHTML = '<option value="" disabled>Error loading models</option>';
            if (modelStatusTag) {
                modelStatusTag.textContent = "Offline";
                modelStatusTag.className = "status-tag offline";
            }
        }
    }
    
    loadModels();
    
    if (modelSelect) {
        modelSelect.addEventListener("change", async (e) => {
            const modelId = e.target.value;
            if (!modelId) return;
            
            modelSelect.disabled = true;
            if (modelLoadingIndicator) modelLoadingIndicator.style.display = "flex";
            promptInput.disabled = true;
            sendBtn.disabled = true;
            
            try {
                const response = await fetch("/api/switch-model", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_id: modelId })
                });
                
                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || "Failed to switch model");
                }
                
                const data = await response.json();
                if (topbarModelName) topbarModelName.textContent = data.name;
                await loadModels();
                
            } catch (error) {
                console.error("Error switching model:", error);
                alert("Failed to switch model: " + error.message);
                await loadModels();
            } finally {
                modelSelect.disabled = false;
                if (modelLoadingIndicator) modelLoadingIndicator.style.display = "none";
                promptInput.disabled = false;
                sendBtn.disabled = false;
            }
        });
    }

    // Category Tabs Filter
    const categoryTabs = document.querySelectorAll(".category-tab");
    const promptCards = document.querySelectorAll(".prompt-card");

    categoryTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            categoryTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const cat = tab.dataset.category;
            promptCards.forEach(card => {
                if (cat === "all" || card.dataset.category === cat) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        });
    });

    // Welcome Prompt Card Click
    if (chatHistory) {
        chatHistory.addEventListener("click", (e) => {
            if (isGenerating) return;
            if (welcomeScreen && welcomeScreen.style.display === "none") return;
            const card = e.target.closest(".prompt-card");
            if (card && card.dataset && card.dataset.prompt) {
                promptInput.value = card.dataset.prompt;
                updateCharCounter();
                sendMessage();
            }
        });
    }

    // Character Counter & Auto-resize Textarea
    function updateCharCounter() {
        const len = promptInput.value.length;
        if (charCounter) charCounter.textContent = `${len} char${len === 1 ? '' : 's'}`;
    }

    promptInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 180) + "px";
        updateCharCounter();
    });

    // Global Keyboard Shortcuts
    document.addEventListener("keydown", (e) => {
        // Ctrl+N -> New Chat
        if (e.ctrlKey && e.key === "n") {
            e.preventDefault();
            startNewChat();
        }
        // Ctrl+\ -> Toggle Sidebar
        if (e.ctrlKey && e.key === "\\") {
            e.preventDefault();
            if (sidebarToggle) sidebarToggle.click();
        }
    });

    // Enter to Send
    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener("click", sendMessage);

    async function sendMessage() {
        if (isGenerating) return;

        const text = promptInput.value.trim();
        if (!text) return;

        isGenerating = true;
        promptInput.disabled = true;
        sendBtn.disabled = true;

        if (welcomeScreen) welcomeScreen.style.display = "none";

        // Update session title snippet if first message
        if (chatHistoryData.length === 0 && activeSessionTitle) {
            activeSessionTitle.textContent = text.length > 22 ? text.substring(0, 22) + "..." : text;
        }

        appendMessage("user", text);
        chatHistoryData.push({ role: "user", content: text });
        promptInput.value = "";
        promptInput.style.height = "28px";
        updateCharCounter();
        sendBtn.style.display = "none";
        if (stopBtn) stopBtn.style.display = "flex";

        const loadingId = appendLoading();

        const handleStop = async () => {
            try {
                await fetch("/api/stop", { method: "POST" });
            } catch (e) {
                console.error("Failed to stop generation:", e);
            }
        };
        if (stopBtn) stopBtn.addEventListener("click", handleStop);

        try {
            const temp = tempSlider ? parseFloat(tempSlider.value) : 0.0;
            const maxTokens = maxTokensInput ? (parseInt(maxTokensInput.value, 10) || 128) : (maxTokensSlider ? parseInt(maxTokensSlider.value, 10) : 128);
            const topK = topkSlider ? parseInt(topkSlider.value, 10) : (temp === 0 ? 1 : 50);
            const topP = toppSlider ? parseFloat(toppSlider.value) : (temp === 0 ? 1.0 : 0.9);
            const repPenalty = repSlider ? parseFloat(repSlider.value) : 1.1;

            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: text,
                    history: [],
                    max_tokens: maxTokens,
                    temperature: temp,
                    top_k: topK,
                    top_p: topP,
                    repetition_penalty: repPenalty
                })
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            removeLoading(loadingId);

            const responseText = (data.response && data.response.trim()) 
                ? data.response 
                : "Virgo generated a response. Feel free to continue the discussion or adjust generation parameters.";

            appendMessage("system", responseText, true, data.generation_time, text);
            chatHistoryData.push({ role: "system", content: responseText });
        } catch (error) {
            console.error("Error:", error);
            removeLoading(loadingId);
            appendMessage("system", "Apologies, an error occurred during generation: " + error.message, true);
        } finally {
            isGenerating = false;
            promptInput.disabled = false;
            if (stopBtn) stopBtn.removeEventListener("click", handleStop);
            if (stopBtn) stopBtn.style.display = "none";
            sendBtn.style.display = "flex";
            sendBtn.disabled = false;
            promptInput.focus();
        }
    }

    function appendMessage(role, content, typing = false, timeTaken = null, lastUserPrompt = "") {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;

        const isLight = document.body.classList.contains("light-theme");
        const iconSrc = getIconSrc(isLight);

        const avatar = document.createElement("div");
        if (role === "user") {
            avatar.className = "avatar user-avatar";
            avatar.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
        } else {
            avatar.className = "avatar";
            avatar.innerHTML = `<img src="${iconSrc}" alt="Virgo" class="avatar-logo-img">`;
        }

        const contentDiv = document.createElement("div");
        contentDiv.className = "content";

        const bodyDiv = document.createElement("div");
        bodyDiv.className = "markdown-body";
        contentDiv.appendChild(bodyDiv);

        let footerDiv = null;
        if (role === "system") {
            footerDiv = document.createElement("div");
            footerDiv.className = "message-footer";

            if (timeTaken !== null) {
                const timeBadge = document.createElement("span");
                timeBadge.className = "meta-badge";
                timeBadge.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> ${timeTaken}s`;
                footerDiv.appendChild(timeBadge);
            }

            // Copy Message Text Button
            const copyBtn = document.createElement("button");
            copyBtn.className = "msg-action-btn";
            copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
            copyBtn.title = "Copy Response";
            copyBtn.addEventListener("click", () => {
                navigator.clipboard.writeText(content).then(() => {
                    copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
                    setTimeout(() => {
                        copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
                    }, 2000);
                });
            });
            footerDiv.appendChild(copyBtn);

            // Regenerate / Retry Button
            if (lastUserPrompt) {
                const retryBtn = document.createElement("button");
                retryBtn.className = "msg-action-btn";
                retryBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg> Retry`;
                retryBtn.title = "Regenerate Answer";
                retryBtn.addEventListener("click", () => {
                    if (!isGenerating) {
                        promptInput.value = lastUserPrompt;
                        updateCharCounter();
                        sendMessage();
                    }
                });
                footerDiv.appendChild(retryBtn);
            }

            if (typing) footerDiv.style.opacity = "0";
            contentDiv.appendChild(footerDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();

        if (role === "user") {
            const p = document.createElement("p");
            p.textContent = content;
            bodyDiv.appendChild(p);
        } else {
            if (typing) {
                let i = 0;
                const speed = 8;
                const tempP = document.createElement("p");
                bodyDiv.appendChild(tempP);
                function typeWriter() {
                    if (i < content.length) {
                        tempP.textContent += content.charAt(i);
                        i++;
                        scrollToBottom();
                        setTimeout(typeWriter, speed);
                    } else {
                        bodyDiv.innerHTML = parseMarkdown(content);
                        attachCodeCopyListeners(bodyDiv);
                        if (footerDiv) {
                            footerDiv.style.transition = "opacity 0.4s ease";
                            footerDiv.style.opacity = "1";
                        }
                        scrollToBottom();
                    }
                }
                typeWriter();
            } else {
                bodyDiv.innerHTML = parseMarkdown(content);
                attachCodeCopyListeners(bodyDiv);
                if (footerDiv) footerDiv.style.opacity = "1";
            }
        }
    }

    function attachCodeCopyListeners(container) {
        container.querySelectorAll(".code-copy-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const codeBlock = btn.closest(".code-block-container").querySelector("code");
                if (codeBlock) {
                    navigator.clipboard.writeText(codeBlock.innerText).then(() => {
                        const originalText = btn.innerHTML;
                        btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
                        setTimeout(() => { btn.innerHTML = originalText; }, 2000);
                    });
                }
            });
        });
    }

    function parseMarkdown(text) {
        if (!text) return "";
        let formatted = String(text);

        // Pre-process broken inline SQL language tags
        formatted = formatted.replace(/(?::sq\s*l|:sql|:\s*sql|\b(?:sq\s*l|sql)\b)\s+((?:CREATE\s+TABLE|SELECT|INSERT\s+INTO|UPDATE|ALTER\s+TABLE)[\s\S]+?;)/gi, (m, code) => {
            return `\n\n\`\`\`sql\n${code.trim()}\n\`\`\`\n\n`;
        });

        // Pre-process fused code blocks
        const knownLangs = ["javascript", "typescript", "python", "html", "css", "java", "rust", "bash", "json", "yaml", "ruby", "cpp", "sql", "php", "py", "js", "ts", "sh", "c"];
        formatted = formatted.replace(/```([a-zA-Z0-9_+-]+)/gi, (match, tag) => {
            for (const lang of knownLangs) {
                if (tag.toLowerCase().startsWith(lang) && tag.length > lang.length) {
                    const rest = tag.slice(lang.length);
                    return `\`\`\`${lang}\n${rest}`;
                }
            }
            return match;
        });

        if (window.marked && typeof window.marked.parse === "function") {
            try {
                if (typeof window.marked.setOptions === "function") {
                    window.marked.setOptions({ gfm: true, breaks: true });
                }
                const parsed = window.marked.parse(formatted);
                // Wrap pre blocks with code-block-container & header
                const parserDiv = document.createElement("div");
                parserDiv.innerHTML = parsed;
                parserDiv.querySelectorAll("pre").forEach(pre => {
                    const code = pre.querySelector("code");
                    let lang = "code";
                    if (code) {
                        const match = code.className.match(/language-([a-zA-Z0-9_+-]+)/);
                        if (match) lang = match[1];
                    }
                    const container = document.createElement("div");
                    container.className = "code-block-container";
                    container.innerHTML = `
                        <div class="code-block-header">
                            <span class="code-lang-tag">${lang}</span>
                            <button class="code-copy-btn">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy code
                            </button>
                        </div>
                    `;
                    pre.parentNode.insertBefore(container, pre);
                    container.appendChild(pre);
                });
                return parserDiv.innerHTML;
            } catch (e) {
                console.error("Marked parse error:", e);
            }
        }

        // Fallback Markdown Parser
        let html = formatted
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        html = html.replace(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g, (match, lang, code) => {
            const cleanCode = code.trim();
            const langName = lang || "code";
            return `
                <div class="code-block-container">
                    <div class="code-block-header">
                        <span class="code-lang-tag">${langName}</span>
                        <button class="code-copy-btn">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy code
                        </button>
                    </div>
                    <pre><code>${cleanCode}</code></pre>
                </div>
            `;
        });
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
        return html.replace(/\n/g, '<br>');
    }

    function appendLoading() {
        const id = "loading-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message system";
        msgDiv.id = id;

        const isLight = document.body.classList.contains("light-theme");
        const iconSrc = getIconSrc(isLight);

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = `<img src="${iconSrc}" alt="Virgo" class="avatar-logo-img">`;

        const contentDiv = document.createElement("div");
        contentDiv.className = "content";
        contentDiv.innerHTML = `
            <div class="thinking-container">
                <span class="thinking-shimmer">Virgo is generating response</span>
                <span class="thinking-dots">
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                </span>
            </div>`;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
