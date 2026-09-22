const state = {
  selectedFile: null,
  jobs: [],
  activeJob: null,
  scoreData: null,
  diagnosisData: null,
  activeTab: "overview",
  pollTimer: null,
  previewTimers: { hurdle: null, pose: null },
};

const elements = {
  systemState: document.querySelector("#systemState"),
  dropZone: document.querySelector("#dropZone"),
  videoInput: document.querySelector("#videoInput"),
  dropTitle: document.querySelector("#dropTitle"),
  dropHint: document.querySelector("#dropHint"),
  selectedFile: document.querySelector("#selectedFile"),
  selectedName: document.querySelector("#selectedName"),
  selectedSize: document.querySelector("#selectedSize"),
  clearFile: document.querySelector("#clearFile"),
  startButton: document.querySelector("#startButton"),
  uploadError: document.querySelector("#uploadError"),
  historyList: document.querySelector("#historyList"),
  refreshHistory: document.querySelector("#refreshHistory"),
  welcomeState: document.querySelector("#welcomeState"),
  jobView: document.querySelector("#jobView"),
  jobStatusChip: document.querySelector("#jobStatusChip"),
  jobTimestamp: document.querySelector("#jobTimestamp"),
  jobFilename: document.querySelector("#jobFilename"),
  jobStageText: document.querySelector("#jobStageText"),
  fullReportLink: document.querySelector("#fullReportLink"),
  newAnalysisButton: document.querySelector("#newAnalysisButton"),
  progressCard: document.querySelector("#progressCard"),
  progressLabel: document.querySelector("#progressLabel"),
  progressPercent: document.querySelector("#progressPercent"),
  progressBar: document.querySelector("#progressBar"),
  previewCard: document.querySelector("#previewCard"),
  hurdlePreviewChannel: document.querySelector("#hurdlePreviewChannel"),
  hurdlePreviewImage: document.querySelector("#hurdlePreviewImage"),
  posePreviewChannel: document.querySelector("#posePreviewChannel"),
  posePreviewImage: document.querySelector("#posePreviewImage"),
  previewBadge: document.querySelector("#previewBadge"),
  previewHint: document.querySelector("#previewHint"),
  stageRow: document.querySelector("#stageRow"),
  errorCard: document.querySelector("#errorCard"),
  errorMessage: document.querySelector("#errorMessage"),
  logLink: document.querySelector("#logLink"),
  results: document.querySelector("#results"),
  resultContent: document.querySelector("#resultContent"),
  toast: document.querySelector("#toast"),
};

const stageConfig = [
  ["preprocessing", "预处理", 10],
  ["detecting", "目标检测", 15],
  ["segmenting", "阶段划分", 45],
  ["extracting_features", "特征计算", 58],
  ["scoring", "技术评分", 72],
  ["diagnosing", "动作诊断", 86],
];

const stageLabels = {
  queued: "等待分析资源",
  waiting_for_pipeline: "已进入分析队列",
  preparing: "正在准备视频",
  preprocessing: "正在统一视频尺寸",
  detecting: "正在识别栏架与人体关键点",
  segmenting: "正在划分动作阶段",
  extracting_features: "正在计算技术指标",
  scoring: "正在生成技术评分",
  diagnosing: "正在诊断动作问题",
  collecting_results: "正在整理分析报告",
  completed: "分析完成",
  failed: "分析失败",
  interrupted: "任务已中断",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function statusName(status) {
  return ({ queued: "排队中", running: "分析中", completed: "已完成", failed: "失败" })[status] || "未知";
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => elements.toast.classList.add("hidden"), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

async function checkHealth() {
  try {
    const health = await api("/api/health");
    elements.systemState.className = "system-state online";
    elements.systemState.lastElementChild.textContent = health.llm === "disabled" ? "分析服务在线 · 规则诊断" : "分析服务在线";
  } catch (_) {
    elements.systemState.className = "system-state error";
    elements.systemState.lastElementChild.textContent = "分析服务未连接";
  }
}

function setSelectedFile(file) {
  elements.uploadError.classList.add("hidden");
  if (!file) {
    state.selectedFile = null;
    elements.videoInput.value = "";
    elements.selectedFile.classList.add("hidden");
    elements.startButton.disabled = true;
    return;
  }
  const suffix = file.name.split(".").pop().toLowerCase();
  if (!["mp4", "mov", "avi", "mkv", "m4v"].includes(suffix)) {
    elements.uploadError.textContent = "请选择 MP4、MOV、AVI、MKV 或 M4V 视频。";
    elements.uploadError.classList.remove("hidden");
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    elements.uploadError.textContent = "视频不能超过 500 MB。";
    elements.uploadError.classList.remove("hidden");
    return;
  }
  state.selectedFile = file;
  elements.selectedName.textContent = file.name;
  elements.selectedSize.textContent = `${formatBytes(file.size)} · 等待上传`;
  elements.selectedFile.querySelector(".file-badge").textContent = suffix.toUpperCase();
  elements.selectedFile.classList.remove("hidden");
  elements.startButton.disabled = false;
}

function bindUpload() {
  elements.videoInput.addEventListener("change", () => setSelectedFile(elements.videoInput.files[0]));
  elements.clearFile.addEventListener("click", (event) => { event.preventDefault(); setSelectedFile(null); });
  ["dragenter", "dragover"].forEach(type => elements.dropZone.addEventListener(type, event => {
    event.preventDefault(); elements.dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach(type => elements.dropZone.addEventListener(type, event => {
    event.preventDefault(); elements.dropZone.classList.remove("dragging");
  }));
  elements.dropZone.addEventListener("drop", event => setSelectedFile(event.dataTransfer.files[0]));
  elements.startButton.addEventListener("click", startAnalysis);
}

async function startAnalysis() {
  if (!state.selectedFile) return;
  elements.startButton.disabled = true;
  elements.startButton.querySelector("span").textContent = "正在上传…";
  const body = new FormData();
  body.append("video", state.selectedFile);
  try {
    const job = await api("/api/jobs", { method: "POST", body });
    setSelectedFile(null);
    toast("视频已上传，分析任务已创建");
    await loadHistory(false);
    selectJob(job.id);
  } catch (error) {
    elements.uploadError.textContent = error.message;
    elements.uploadError.classList.remove("hidden");
  } finally {
    elements.startButton.querySelector("span").textContent = "开始动作分析";
    elements.startButton.disabled = !state.selectedFile;
  }
}

async function loadHistory(selectLatest = false) {
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    renderHistory();
    if (selectLatest && data.jobs.length) selectJob(data.jobs[0].id);
  } catch (error) {
    elements.historyList.innerHTML = `<div class="history-empty">${escapeHtml(error.message)}</div>`;
  }
}

function renderHistory() {
  if (!state.jobs.length) {
    elements.historyList.innerHTML = '<div class="history-empty">还没有分析记录</div>';
    return;
  }
  elements.historyList.innerHTML = state.jobs.slice(0, 12).map(job => `
    <div class="history-item ${state.activeJob?.id === job.id ? "active" : ""}">
      <button class="history-open" data-job-id="${job.id}" type="button">
        <span class="history-icon">${escapeHtml((job.stored_filename?.split(".").pop() || "VID").toUpperCase())}</span>
        <span class="history-info">
          <strong>${escapeHtml(job.original_filename)}</strong>
          <small>${formatDate(job.created_at)} · ${statusName(job.status)}</small>
        </span>
        <span class="history-status ${escapeHtml(job.status)}" aria-label="${statusName(job.status)}"></span>
      </button>
      <button class="history-delete" data-delete-job="${job.id}" type="button" aria-label="删除 ${escapeHtml(job.original_filename)}" ${["queued", "running"].includes(job.status) ? "disabled" : ""}>
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.75 4.75h8.5m-7.5 0 .5 8h5.5l.5-8m-4.75 0V3.5h3v1.25M6.5 7v3.5m3-3.5v3.5" /></svg>
      </button>
    </div>
  `).join("");
  elements.historyList.querySelectorAll("[data-job-id]").forEach(button => {
    button.addEventListener("click", () => selectJob(button.dataset.jobId));
  });
  elements.historyList.querySelectorAll("[data-delete-job]").forEach(button => {
    button.addEventListener("click", () => deleteJob(button.dataset.deleteJob));
  });
}

function showNewAnalysis() {
  window.clearTimeout(state.pollTimer);
  clearPreviewTimers();
  state.activeJob = null;
  state.scoreData = null;
  state.diagnosisData = null;
  elements.jobView.classList.add("hidden");
  elements.previewCard.classList.add("hidden");
  elements.hurdlePreviewChannel.classList.add("hidden");
  elements.posePreviewChannel.classList.add("hidden");
  elements.welcomeState.classList.remove("hidden");
  renderHistory();
  elements.dropZone.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function deleteJob(jobId) {
  const job = state.jobs.find(item => item.id === jobId);
  if (!job || !window.confirm(`删除“${job.original_filename}”的分析记录和所有结果文件？`)) return;
  try {
    await api(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (state.activeJob?.id === jobId) showNewAnalysis();
    await loadHistory(false);
    toast("分析记录已删除");
  } catch (error) {
    toast(error.message);
  }
}

async function selectJob(jobId) {
  window.clearTimeout(state.pollTimer);
  clearPreviewTimers();
  state.scoreData = null;
  state.diagnosisData = null;
  try {
    state.activeJob = await api(`/api/jobs/${jobId}`);
    renderHistory();
    renderJob();
    loadPreviews();
    if (["queued", "running"].includes(state.activeJob.status)) schedulePoll();
    if (state.activeJob.status === "completed") await loadResults();
  } catch (error) { toast(error.message); }
}

function clearPreviewTimers() {
  Object.values(state.previewTimers).forEach(timer => window.clearTimeout(timer));
  state.previewTimers = { hurdle: null, pose: null };
}

function previewIsLive() {
  return state.activeJob?.status === "running" && state.activeJob?.stage === "detecting";
}

function closeLivePreview() {
  clearPreviewTimers();
  elements.hurdlePreviewImage.removeAttribute("src");
  elements.posePreviewImage.removeAttribute("src");
  elements.hurdlePreviewChannel.classList.add("hidden");
  elements.posePreviewChannel.classList.add("hidden");
  elements.previewCard.classList.add("hidden");
}

function updatePreviewStatus() {
  elements.previewBadge.classList.remove("finished");
  elements.previewBadge.innerHTML = "<i></i> LIVE";
  elements.previewHint.textContent = "两路画面随各自模型的检测进度独立更新。";
}

function updatePreviewVisibility() {
  const hasPreview = !elements.hurdlePreviewChannel.classList.contains("hidden")
    || !elements.posePreviewChannel.classList.contains("hidden");
  elements.previewCard.classList.toggle("hidden", !hasPreview);
}

function loadPreviewKind(kind) {
  const job = state.activeJob;
  if (!job) return;
  window.clearTimeout(state.previewTimers[kind]);
  const channel = kind === "hurdle" ? elements.hurdlePreviewChannel : elements.posePreviewChannel;
  const image = kind === "hurdle" ? elements.hurdlePreviewImage : elements.posePreviewImage;
  const previewUrl = `/api/jobs/${job.id}/previews/${kind}?t=${Date.now()}`;
  const loader = new Image();
  loader.onload = () => {
    if (state.activeJob?.id !== job.id || !previewIsLive()) return;
    image.src = previewUrl;
    channel.classList.remove("hidden");
    updatePreviewStatus();
    updatePreviewVisibility();
    if (previewIsLive()) state.previewTimers[kind] = window.setTimeout(() => loadPreviewKind(kind), 250);
  };
  loader.onerror = () => {
    if (state.activeJob?.id !== job.id) return;
    if (previewIsLive()) {
      state.previewTimers[kind] = window.setTimeout(() => loadPreviewKind(kind), 350);
    } else {
      channel.classList.add("hidden");
      updatePreviewVisibility();
    }
  };
  loader.src = previewUrl;
}

function loadPreviews() {
  clearPreviewTimers();
  if (!previewIsLive()) {
    closeLivePreview();
    return;
  }
  updatePreviewStatus();
  loadPreviewKind("hurdle");
  loadPreviewKind("pose");
}

function schedulePoll() {
  state.pollTimer = window.setTimeout(async () => {
    if (!state.activeJob) return;
    try {
      state.activeJob = await api(`/api/jobs/${state.activeJob.id}`);
      const index = state.jobs.findIndex(job => job.id === state.activeJob.id);
      if (index >= 0) state.jobs[index] = state.activeJob;
      renderHistory();
      renderJob();
      loadPreviews();
      if (["queued", "running"].includes(state.activeJob.status)) schedulePoll();
      else if (state.activeJob.status === "completed") await loadResults();
    } catch (error) { toast(error.message); }
  }, 1400);
}

function renderJob() {
  const job = state.activeJob;
  if (!job) return;
  elements.welcomeState.classList.add("hidden");
  elements.jobView.classList.remove("hidden");
  elements.jobFilename.textContent = job.original_filename;
  elements.jobTimestamp.textContent = `创建于 ${formatDate(job.created_at)}`;
  elements.jobStageText.textContent = stageLabels[job.stage] || "正在处理";
  elements.jobStatusChip.textContent = statusName(job.status);
  elements.jobStatusChip.className = `status-chip ${job.status}`;
  elements.progressLabel.textContent = stageLabels[job.stage] || "正在处理";
  elements.progressPercent.textContent = `${job.progress || 0}%`;
  elements.progressBar.style.width = `${job.progress || 0}%`;
  elements.progressCard.classList.toggle("completed", job.status === "completed");
  elements.errorCard.classList.toggle("hidden", job.status !== "failed");
  elements.results.classList.toggle("hidden", job.status !== "completed");
  elements.fullReportLink.classList.add("hidden");
  if (job.status === "failed") {
    elements.errorMessage.textContent = job.error || "未知错误";
    elements.logLink.href = `/api/jobs/${job.id}/log`;
  }
  renderStages(job);
}

function renderStages(job) {
  const progress = job.progress || 0;
  elements.stageRow.innerHTML = stageConfig.map(([key, label, threshold]) => {
    const current = job.stage === key;
    const done = progress > threshold || job.status === "completed";
    return `<span class="stage-step ${current ? "current" : done ? "done" : ""}">${label}</span>`;
  }).join("");
}

async function loadResults() {
  const job = state.activeJob;
  const scoreArtifact = job.artifacts.find(item => item.name.endsWith("_result.json"));
  const diagnosisArtifact = job.artifacts.find(item => item.name.endsWith("_diagnosis.json"));
  const reportArtifact = job.artifacts.find(item => item.name.endsWith("_advice.html"));
  if (reportArtifact) {
    elements.fullReportLink.href = reportArtifact.url;
    elements.fullReportLink.classList.remove("hidden");
  }
  try {
    [state.scoreData, state.diagnosisData] = await Promise.all([
      scoreArtifact ? api(scoreArtifact.url) : null,
      diagnosisArtifact ? api(diagnosisArtifact.url) : null,
    ]);
    renderActiveTab();
  } catch (error) {
    elements.resultContent.innerHTML = `<div class="history-empty">结果读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderActiveTab() {
  document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.tab === state.activeTab));
  const renderers = { overview: renderOverview, scores: renderScores, advice: renderAdvice, files: renderFiles };
  elements.resultContent.innerHTML = renderers[state.activeTab]();
}

function renderOverview() {
  const score = state.scoreData;
  const diagnosis = state.diagnosisData;
  if (!score || !diagnosis) return '<div class="history-empty">暂无结构化分析结果</div>';
  const total = Number(score.overall_score?.score || 0);
  const stages = diagnosis.stage_diagnosis || [];
  const problems = diagnosis.top_problems || [];
  return `
    <div class="overview-grid">
      <article class="score-hero">
        <div class="score-ring" style="--score:${Math.min(total / 5 * 360, 360)}deg">
          <div class="score-value"><strong>${total.toFixed(2)}</strong><small>满分 5.00</small></div>
        </div>
        <strong>综合技术评分</strong><span>${escapeHtml(diagnosis.overall_summary)}</span>
      </article>
      <div class="summary-stack">
        <article class="summary-card">
          <div class="card-heading"><h3>阶段表现</h3><small>三个关键技术阶段</small></div>
          <div class="stage-scores">${stages.map(stage => `
            <div class="stage-score"><span>${escapeHtml(stage.stage_name)}</span><strong>${Number(stage.score).toFixed(2)}</strong><div class="mini-track"><i style="width:${Number(stage.score) / 5 * 100}%"></i></div></div>
          `).join("")}</div>
          <div class="weak-stage"><span class="flag">!</span><div><strong>优先改善：${escapeHtml(diagnosis.weakest_stage?.stage_name)}</strong><p>${escapeHtml(diagnosis.weakest_stage?.summary)}</p></div></div>
        </article>
      </div>
    </div>
    <div class="card-heading" style="margin-top:24px"><h3>优先关注的问题</h3><small>按改善优先级排序</small></div>
    <div class="problem-list">${problems.map((problem, index) => `
      <article class="problem-card"><span class="problem-number">0${index + 1}</span><h3>${escapeHtml(problem.problem_title)}</h3><p>${escapeHtml(problem.diagnosis)}</p><span class="problem-score">${Number(problem.score).toFixed(2)} 分</span></article>
    `).join("")}</div>
  `;
}

function renderScores() {
  const score = state.scoreData;
  if (!score) return '<div class="history-empty">暂无评分结果</div>';
  const stageNames = { takeoff: "起跨", flight: "腾空", landing: "下栏" };
  return `<article class="score-table-card"><div class="score-table-header"><h3>14 项动作技术评分</h3></div><div style="overflow-x:auto"><table class="score-table"><thead><tr><th>编号</th><th>技术指标</th><th>阶段</th><th class="score-bar-cell">评分</th></tr></thead><tbody>${score.item_scores.map(item => `
    <tr><td>${escapeHtml(item.task_id)}</td><td><strong>${escapeHtml(item.task_name)}</strong></td><td>${escapeHtml(stageNames[item.stage] || item.stage)}</td><td class="score-bar-cell"><div class="inline-score"><div class="mini-track"><i style="width:${Number(item.score) / 5 * 100}%"></i></div><strong>${Number(item.score).toFixed(1)}</strong></div></td></tr>
  `).join("")}</tbody></table></div></article>`;
}

function renderAdvice() {
  const diagnosis = state.diagnosisData;
  if (!diagnosis) return '<div class="history-empty">暂无训练建议</div>';
  return `<div class="advice-list">${diagnosis.top_problems.map(problem => `
    <article class="advice-card">
      <div class="advice-card-header"><h3>${escapeHtml(problem.problem_title)}</h3><span class="task-id">${escapeHtml(problem.task_id)}</span></div>
      <p>${escapeHtml(problem.impact)}</p>
      <ul class="evidence-list">${(problem.evidence || []).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <div class="card-heading" style="margin:20px 0 10px"><h3>推荐练习</h3></div>
      <ul class="drill-list">${(problem.recommended_drills || []).slice(0, 5).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </article>
  `).join("")}</div>`;
}

function renderFiles() {
  const artifacts = state.activeJob?.artifacts || [];
  const descriptions = { reports: "可在浏览器中查看", json: "结构化分析数据", csv: "可导入表格软件", manifest: "任务文件清单" };
  return `<div class="files-card">${artifacts.map(file => {
    const type = file.name.split(".").pop().toUpperCase();
    return `<a class="file-link" href="${escapeHtml(file.url)}" target="_blank" rel="noopener"><span class="file-type">${escapeHtml(type)}</span><span><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(descriptions[file.category] || "分析结果文件")}</small></span><span class="file-arrow">↗</span></a>`;
  }).join("")}</div>`;
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    state.activeTab = tab.dataset.tab;
    renderActiveTab();
  }));
}

async function init() {
  bindUpload();
  bindTabs();
  elements.refreshHistory.addEventListener("click", () => loadHistory(false));
  elements.newAnalysisButton.addEventListener("click", showNewAnalysis);
  await Promise.all([checkHealth(), loadHistory(false)]);
}

init();
