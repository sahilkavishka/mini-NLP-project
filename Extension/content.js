/**
 * AI Review Insight - Pro Max Edition (Daraz Native Theme + Deep Scan + Smart Pros & Cons + Breakdown)
 */
(() => {
  "use strict";

  if (window.__aiReviewInsightInjected) return;
  window.__aiReviewInsightInjected = true;

  const CONFIG = {
    INIT_DELAY_MS: 2000,
    SCROLL_AMOUNT: 1200,
    SCROLL_REPEATS: 3, // Deep Scan: 3 පාරක් පල්ලෙහාට යනවා
    MIN_REVIEW_LENGTH: 15,
    REVIEW_SELECTORS: ".mod-reviews .content, .item-content .content, .review-content, .qna-content",
    TITLE_SELECTOR: ".pdp-product-title",
    POSITIVE_THRESHOLD: 50,
    CACHE_TTL_MS: 30 * 60 * 1000,
    REQUEST_TIMEOUT_MS: 20000,
  };

  const CACHE_KEY = `ai-review-insight:${location.href.split("#")[0]}`;
  const DISMISS_KEY = `ai-review-insight-dismissed:${location.href.split("#")[0]}`;
  let requestToken = 0;

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str ?? "");
    return div.innerHTML;
  }

  function sendMessage(payload, timeoutMs = CONFIG.REQUEST_TIMEOUT_MS) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Timed out waiting for backend")), timeoutMs);
      try {
        chrome.runtime.sendMessage(payload, (response) => {
          clearTimeout(timer);
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
            return;
          }
          resolve(response);
        });
      } catch (err) {
        clearTimeout(timer);
        reject(err);
      }
    });
  }

  function getCache() {
    return new Promise((resolve) => {
      try { chrome.storage.local.get([CACHE_KEY], (res) => resolve(res[CACHE_KEY] || null)); } 
      catch { resolve(null); }
    });
  }

  function setCache(data) {
    try { chrome.storage.local.set({ [CACHE_KEY]: { data, savedAt: Date.now() } }); } 
    catch { /* ignore */ }
  }

  function timeAgo(ts) {
    const mins = Math.max(1, Math.round((Date.now() - ts) / 60000));
    if (mins < 60) return `${mins} min ago`;
    const hrs = Math.round(mins / 60);
    return `${hrs} hr${hrs > 1 ? "s" : ""} ago`;
  }

  function injectStyles() {
    if (document.getElementById("ai-review-insight-styles")) return;
    const style = document.createElement("style");
    style.id = "ai-review-insight-styles";
    style.textContent = `
      :root {
        --ari-accent: #f85606; /* Daraz Orange */
        --ari-accent-dark: #d04604;
        --ari-good: #10b981;
        --ari-bad: #ef4444;
        --ari-neutral: #f59e0b;
        --ari-bg: #ffffff;
        --ari-text: #0f172a;
        --ari-text-muted: #64748b;
        --ari-border: #e2e8f0;
      }
      @keyframes ai-spin { to { transform: rotate(360deg); } }
      @keyframes ai-fade-in { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
      @keyframes circle-fill { from { stroke-dasharray: 0, 100; } }
      
      .ai-review-spinner {
        width: 32px; height: 32px;
        border: 4px solid var(--ari-border);
        border-top-color: var(--ari-accent);
        border-radius: 50%;
        animation: ai-spin 0.9s linear infinite;
      }
      .ai-review-chip {
        cursor: pointer;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 4px 4px 0 0;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        transition: 0.2s;
      }
      .ai-review-chip:hover { filter: brightness(0.95); transform: scale(1.02); }
      
      .svg-ring { width: 60px; height: 60px; transform: rotate(-90deg); }
      .svg-ring-bg { fill: none; stroke: var(--ari-border); stroke-width: 3; }
      .svg-ring-fill { fill: none; stroke: var(--ari-good); stroke-width: 3; stroke-linecap: round; animation: circle-fill 1.2s ease-out; }
      
      .ai-review-icon-btn { background: transparent; border: none; cursor: pointer; color: var(--ari-text-muted); font-size: 14px; padding: 2px 6px; border-radius: 6px; }
      .ai-review-icon-btn:hover { background: #e2e8f0; }
      
      .ai-review-toast {
        position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
        background: #1e293b; color: #fff; padding: 10px 16px; border-radius: 8px; font-size: 13px; z-index: 999999;
        animation: ai-fade-in 0.2s ease-out;
      }
    `;
    document.head.appendChild(style);
  }

  function toast(message) {
    const el = document.createElement("div");
    el.className = "ai-review-toast";
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
  }

  function buildWidget() {
    const box = document.createElement("div");
    box.id = "ai-review-insight-box";
    Object.assign(box.style, {
      padding: "16px", marginTop: "15px", marginBottom: "15px",
      backgroundColor: "var(--ari-bg)", border: "2px solid var(--ari-accent)",
      borderRadius: "12px", boxShadow: "0 10px 25px rgba(248, 86, 6, 0.15)",
      fontFamily: "Arial, sans-serif", color: "var(--ari-text)",
    });

    box.innerHTML = `
      <div style="font-weight: 800; font-size: 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
            <img src="https://laz-img-cdn.alicdn.com/images/ims-web/TB1993jbcUrBKNjSZPxXXX00pXa.png" style="height:20px;">
            <span>AI Review Insight</span>
        </div>
        <div style="display:flex; align-items:center; gap: 4px;">
          <span style="font-size: 11px; background: #fff3ee; color: var(--ari-accent); padding: 3px 8px; border-radius: 12px; font-weight:bold;">BERT AI</span>
          <button id="ai-minimize-btn" class="ai-review-icon-btn" title="Minimize">▾</button>
          <button id="ai-close-btn" class="ai-review-icon-btn" title="Dismiss">✕</button>
        </div>
      </div>
      <div id="ai-review-body">
        <button id="ai-analyze-btn" type="button" style="background: var(--ari-accent); color: white; padding: 12px 15px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; font-size: 15px; transition: 0.3s; box-shadow: 0 4px 10px rgba(248, 86, 6, 0.3);">
          🔍 Deep Scan & Analyze Reviews
        </button>
        <div id="ai-result-area" style="margin-top: 15px; text-align: center; display: none;"></div>
      </div>
    `;
    return box;
  }

  function mountWidget(box) {
    const titleArea = document.querySelector(CONFIG.TITLE_SELECTOR);
    if (titleArea && titleArea.parentNode) {
      titleArea.parentNode.insertBefore(box, titleArea.nextSibling);
    } else {
      document.body.prepend(box);
    }
  }

  function wireChromeControls(box) {
    const minimizeBtn = box.querySelector("#ai-minimize-btn");
    const closeBtn = box.querySelector("#ai-close-btn");
    const body = box.querySelector("#ai-review-body");

    minimizeBtn.addEventListener("click", () => {
      const collapsed = body.style.display === "none";
      body.style.display = collapsed ? "" : "none";
      minimizeBtn.textContent = collapsed ? "▾" : "▸";
    });
    closeBtn.addEventListener("click", () => { box.remove(); sessionStorage.setItem(DISMISS_KEY, "1"); });
  }

  function collectReviews() {
    const seen = new Set();
    document.querySelectorAll(CONFIG.REVIEW_SELECTORS).forEach((el) => {
      const text = el.innerText.trim();
      if (text.length > CONFIG.MIN_REVIEW_LENGTH) seen.add(text);
    });
    return Array.from(seen);
  }

  function jumpToKeyword(keyword) {
    const lower = keyword.toLowerCase();
    const els = document.querySelectorAll(CONFIG.REVIEW_SELECTORS);
    for (const el of els) {
      if (el.innerText.toLowerCase().includes(lower)) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return true;
      }
    }
    toast(`Couldn't find "${keyword}" in visible reviews`);
    return false;
  }

  function showLoading(resultArea, onCancel) {
    resultArea.style.display = "block";
    resultArea.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px 0;">
        <div class="ai-review-spinner"></div>
        <div id="ai-loading-label" style="margin-top: 12px; color: var(--ari-accent); font-weight: bold; font-size: 14px;">
          Deep Scanning Page... (1/${CONFIG.SCROLL_REPEATS})
        </div>
        <div style="font-size:11px; color:var(--ari-text-muted); margin-top:5px;">Extracting hidden reviews</div>
        <button id="ai-cancel-btn" style="margin-top:10px; background:none; border:1px solid var(--ari-border); padding:4px 12px; border-radius:6px; font-size:12px; cursor:pointer;">Cancel</button>
      </div>
    `;
    resultArea.querySelector("#ai-cancel-btn").addEventListener("click", onCancel);
  }

  function renderResult(box, resultArea, data, { cached = false, savedAt = null } = {}) {
    const pos = Number(data.positive_percentage) || 0;
    const neg = Number(data.negative_percentage) || 0;
    const neu = Math.max(0, 100 - pos - neg);
    
    const isGood = pos >= CONFIG.POSITIVE_THRESHOLD;
    const color = isGood ? "var(--ari-good)" : "var(--ari-bad)";
    
    // Status එක පැහැදිලි කිරීම
    let status = isGood ? "Excellent Choice" : "Think Twice";
    if (pos === 50 && neu >= 50) status = "Average / Okay";
    
    box.style.borderColor = color;

    // Smart Pros & Cons HTML Generate කිරීම
    const prosHtml = (data.top_pros && data.top_pros.length > 0) 
      ? data.top_pros.map(k => `<button type="button" class="ai-review-chip" style="background:#dcfce7; color:#166534; border:1px solid #bbf7d0;" data-keyword="${escapeHtml(k)}">+ ${escapeHtml(k)}</button>`).join("")
      : "<span style='color:#94a3b8; font-size:11px;'>Not enough data</span>";
      
    const consHtml = (data.top_cons && data.top_cons.length > 0)
      ? data.top_cons.map(k => `<button type="button" class="ai-review-chip" style="background:#fee2e2; color:#991b1b; border:1px solid #fecaca;" data-keyword="${escapeHtml(k)}">- ${escapeHtml(k)}</button>`).join("")
      : "<span style='color:#94a3b8; font-size:11px;'>No major issues found</span>";

    resultArea.innerHTML = `
      <div style="background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid var(--ari-border); display:flex; align-items:center; gap: 15px; text-align:left;">
        <svg class="svg-ring" viewBox="0 0 36 36">
          <path class="svg-ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
          <path class="svg-ring-fill" stroke="${color}" stroke-dasharray="${pos}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
        </svg>

        <div style="flex:1;">
            <div style="color: ${color}; font-size: 24px; font-weight: 900; line-height: 1.1;">
                ${pos}% <span style="font-size: 14px; font-weight: bold; color:var(--ari-text);">Positive</span>
            </div>
            <div style="font-size: 13px; font-weight: bold; color: ${color};">${isGood ? '🥇' : '⚠️'} ${status}</div>
            
            <div style="height: 6px; border-radius: 3px; background: #e2e8f0; display: flex; overflow: hidden; margin-top: 8px;">
              <div style="width:${pos}%; background: var(--ari-good);"></div>
              ${neu > 0 ? `<div style="width:${neu}%; background: var(--ari-neutral);"></div>` : ""}
              ${neg > 0 ? `<div style="width:${neg}%; background: var(--ari-bad);"></div>` : ""}
            </div>
            
            <!-- පැටලෙන්නේ නැති වෙන්න ප්‍රතිශත වෙන් කර පෙන්වීම -->
            <div style="display:flex; justify-content:space-between; font-size: 10px; color: var(--ari-text-muted); margin-top: 6px; font-weight:bold;">
              <span style="color: var(--ari-good);">👍 Good: ${pos}%</span>
              ${neu > 0 ? `<span style="color: var(--ari-neutral);">😐 Average: ${neu}%</span>` : ""}
              <span style="color: var(--ari-bad);">👎 Bad: ${neg}%</span>
            </div>
        </div>
      </div>
      
      <!-- AI Pros & Cons Columns -->
      <div style="display:flex; gap:10px; margin-top:15px; text-align:left;">
          <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; padding:12px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
              <div style="font-size:12px; font-weight:800; color:#16a34a; margin-bottom:8px; display:flex; align-items:center; gap:4px;">
                  <svg width="14" height="14" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
                  PROS
              </div>
              <div>${prosHtml}</div>
          </div>
          <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; padding:12px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
              <div style="font-size:12px; font-weight:800; color:#dc2626; margin-bottom:8px; display:flex; align-items:center; gap:4px;">
                  <svg width="14" height="14" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>
                  CONS
              </div>
              <div>${consHtml}</div>
          </div>
      </div>
      
      <div style="display:flex; justify-content: space-between; align-items:center; font-size: 11px; color: var(--ari-text-muted); margin-top: 15px; border-top: 1px dashed var(--ari-border); padding-top: 10px;">
        <span>${cached ? `Cached &middot; ${timeAgo(savedAt)}` : `Analyzed ${data.total_analyzed} extracted reviews`}</span>
        <button id="ai-rescan-btn" style="background:none; border:none; color: var(--ari-accent-dark); cursor:pointer; text-decoration: underline; font-weight:bold;">Rescan</button>
      </div>
    `;

    // වචන Click කරාම අදාල Review එකට යෑම
    resultArea.querySelectorAll(".ai-review-chip").forEach((chip) => {
      chip.addEventListener("click", () => jumpToKeyword(chip.dataset.keyword));
    });
    resultArea.querySelector("#ai-rescan-btn").addEventListener("click", () => runScan(box, { force: true }));
  }

  async function runScan(box, { force = false } = {}) {
    const btn = box.querySelector("#ai-analyze-btn");
    const resultArea = box.querySelector("#ai-result-area");
    requestToken++;
    const myToken = requestToken;

    if (!force) {
      const cached = await getCache();
      if (cached && Date.now() - cached.savedAt < CONFIG.CACHE_TTL_MS) {
        btn.style.display = "none";
        resultArea.style.display = "block";
        renderResult(box, resultArea, cached.data, { cached: true, savedAt: cached.savedAt });
        return;
      }
    }

    btn.style.display = "none";
    showLoading(resultArea, () => {
      requestToken++; btn.style.display = "block"; resultArea.style.display = "none"; toast("Scan cancelled");
    });

    const loadingLabel = resultArea.querySelector("#ai-loading-label");
    
    // Deep Scan Logic
    for (let i = 0; i < CONFIG.SCROLL_REPEATS; i++) {
        if (myToken !== requestToken) return;
        loadingLabel.textContent = `Deep Scanning Page... (${i+1}/${CONFIG.SCROLL_REPEATS})`;
        window.scrollBy({ top: CONFIG.SCROLL_AMOUNT, behavior: "smooth" });
        await new Promise(r => setTimeout(r, 800));
    }

    if (myToken !== requestToken) return;
    const reviews = collectReviews();

    if (reviews.length === 0) {
      btn.style.display = "block"; resultArea.style.display = "none";
      toast("No reviews found. Try scrolling to the bottom.");
      return;
    }

    loadingLabel.textContent = `Analyzing ${reviews.length} reviews with BERT AI...`;

    try {
      const response = await sendMessage({ action: "analyze", reviews });
      if (myToken !== requestToken) return;

      if (response && response.success) {
        setCache(response.data);
        renderResult(box, resultArea, response.data);
      } else {
        btn.style.display = "block"; resultArea.innerHTML = "❌ Connection error! Is Python Server running?";
      }
    } catch (err) {
      if (myToken !== requestToken) return;
      btn.style.display = "block"; resultArea.innerHTML = `❌ Error: ${err.message}`;
    }
  }

  function init() {
    if (sessionStorage.getItem(DISMISS_KEY) === "1") return;
    injectStyles();
    const box = buildWidget();
    mountWidget(box);
    wireChromeControls(box);
    box.querySelector("#ai-analyze-btn").addEventListener("click", () => runScan(box));
    
    getCache().then((cached) => {
      if (cached && Date.now() - cached.savedAt < CONFIG.CACHE_TTL_MS) {
        box.querySelector("#ai-analyze-btn").style.display = "none";
        box.querySelector("#ai-result-area").style.display = "block";
        renderResult(box, box.querySelector("#ai-result-area"), cached.data, { cached: true, savedAt: cached.savedAt });
      }
    });
  }

  setTimeout(init, CONFIG.INIT_DELAY_MS);
})();