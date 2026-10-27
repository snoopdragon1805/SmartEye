import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity, AlertCircle, ArrowDownToLine, AudioLines, Bell, BookOpenCheck,
  Camera, Check, CircleHelp, Clock3, Eye, Gauge, LayoutDashboard,
  LoaderCircle, Menu, MonitorPlay, Radio, RefreshCw, Settings2, ShieldCheck,
  SlidersHorizontal, UserRound, Wifi, WifiOff, X
} from "lucide-react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { api, API_URL, WS_URL } from "./api";
import type { Metrics, Session, SessionDetail, Status } from "./types";

type Page = "overview" | "live" | "calibration" | "history" | "analytics" | "settings";
type PendingWindow = { session_id: string; window_id: number };

const navigation: Array<{ id: Page; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "live", label: "Live monitoring", icon: MonitorPlay },
  { id: "calibration", label: "Personal calibration", icon: SlidersHorizontal },
  { id: "history", label: "Session history", icon: Clock3 },
  { id: "analytics", label: "Analytics", icon: Activity },
  { id: "settings", label: "Settings", icon: Settings2 }
];

const emptyStatus: Status = {
  backend_connected: false,
  mode: "idle",
  session_active: false,
  session_id: null,
  session_elapsed_seconds: 0,
  window_id: 1,
  camera_connected: false,
  face_detected: false,
  calibration_available: false,
  ear_threshold: null,
  error: null,
  calibration_progress: 0,
  calibration_message: null
};

function formatValue(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "00:00";
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function timeLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { minute: "2-digit", second: "2-digit" });
}

function MetricCard({ label, value, unit, icon: Icon, detail }: {
  label: string; value: string; unit?: string; icon: typeof Eye; detail: string;
}) {
  return (
    <article className="metric-card" title={detail}>
      <div className="metric-top">
        <span className="metric-icon"><Icon size={16} aria-hidden="true" /></span>
        <span className="metric-label">{label}</span>
        <span className="tooltip" aria-label={detail}><CircleHelp size={14} /></span>
      </div>
      <div className="metric-value">{value}<small>{unit}</small></div>
      <p className="metric-detail">{detail}</p>
    </article>
  );
}

function TrendChart({ title, data, dataKey, color, unit, domain }: {
  title: string; data: Metrics[]; dataKey: string; color: string; unit: string;
  domain?: [number | "auto", number | "auto"];
}) {
  const rows = data.map((item) => ({
    timestamp: item.timestamp,
    value: dataKey === "ear"
      ? item.ear?.current ?? null
      : dataKey === "score"
        ? item.fatigue.score
        : dataKey === "perclos"
          ? item.eye_closure?.perclos_percent ?? null
          : item.blink?.rate_per_minute ?? null
  }));
  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div><h3>{title}</h3><p>Live measurements · recent samples</p></div>
        <span className="chart-legend"><i style={{ background: color }} />{unit}</span>
      </div>
      {rows.some((row) => row.value != null) ? (
        <div className="chart-area">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
              <CartesianGrid stroke="#1b2b3c" strokeDasharray="3 5" vertical={false} />
              <XAxis dataKey="timestamp" tickFormatter={timeLabel} minTickGap={30} tick={{ fill: "#8295aa", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis domain={domain ?? ["auto", "auto"]} tick={{ fill: "#8295aa", fontSize: 10 }} axisLine={false} tickLine={false} width={42} />
              <Tooltip
                labelFormatter={(label) => new Date(String(label)).toLocaleTimeString()}
                formatter={(value) => [value == null ? "Unavailable" : `${Number(value).toFixed(3)} ${unit}`, title]}
                contentStyle={{ background: "#0d1a2a", border: "1px solid #26384b", borderRadius: 10, color: "#eaf1f8" }}
              />
              <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} connectNulls={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : <div className="empty-chart">Waiting for real measurements…</div>}
    </section>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [status, setStatus] = useState<Status>(emptyStatus);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<Metrics[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null);
  const [profile, setProfile] = useState<Record<string, number | string> | null>(null);
  const [profileAvailable, setProfileAvailable] = useState(false);
  const [pending, setPending] = useState<PendingWindow[]>([]);
  const [labelChoice, setLabelChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [statusReceivedAt, setStatusReceivedAt] = useState(Date.now());
  const [sampleCount, setSampleCount] = useState(60);

  const refreshStatus = useCallback(async () => {
    try {
      const current = await api.status();
      setStatus(current);
      setStatusReceivedAt(Date.now());
      setError("");
    } catch (reason) {
      setStatus((previous) => ({ ...previous, backend_connected: false }));
      setError(reason instanceof Error ? reason.message : "Backend unavailable.");
    }
  }, []);

  const refreshProfile = useCallback(async () => {
    try {
      const result = await api.profile();
      setProfileAvailable(result.available);
      setProfile(result.profile);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load calibration profile.");
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.sessions());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load session history.");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void refreshProfile();
    void refreshSessions();
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [refreshProfile, refreshSessions, refreshStatus]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnect: number | undefined;
    let stopped = false;
    const connect = () => {
      if (stopped) return;
      socket = new WebSocket(WS_URL);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as {
            status?: Status; metrics?: Metrics | null; pending_windows?: PendingWindow[];
          };
          if (payload.status) {
            setStatus(payload.status);
            setStatusReceivedAt(Date.now());
          }
          if (payload.metrics) setMetrics(payload.metrics);
          if (payload.pending_windows) setPending(payload.pending_windows);
        } catch {
          setError("Received an invalid live update from the backend.");
        }
      };
      socket.onclose = () => {
        if (!stopped) reconnect = window.setTimeout(connect, 1500);
      };
      socket.onerror = () => socket?.close();
    };
    const connectionStart = window.setTimeout(connect, 0);
    const poll = window.setInterval(() => {
      void refreshStatus();
      api.metrics().then((value) => {
        if ("timestamp" in value) setMetrics(value);
      }).catch(() => undefined);
      if (status.session_active) {
        api.history().then(setHistory).catch(() => undefined);
      }
    }, 2500);
    return () => {
      stopped = true;
      window.clearTimeout(connectionStart);
      if (reconnect) window.clearTimeout(reconnect);
      window.clearInterval(poll);
      socket?.close();
    };
  }, [refreshStatus, status.session_active]);

  useEffect(() => {
    if (page === "history" || page === "analytics") void refreshSessions();
  }, [page, refreshSessions]);

  useEffect(() => {
    if (!selectedSession) {
      setSessionDetail(null);
      return;
    }
    api.session(selectedSession).then(setSessionDetail).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : "Could not load this session.");
    });
  }, [selectedSession]);

  const visibleHistory = useMemo(() => history.slice(-sampleCount), [history, sampleCount]);
  const elapsed = status.session_active
    ? status.session_elapsed_seconds + Math.max(0, clock - statusReceivedAt) / 1000
    : status.session_elapsed_seconds;
  const pageTitle = navigation.find((item) => item.id === page)?.label ?? "Overview";

  const runAction = async (action: () => Promise<unknown>, success: string) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
      setNotice(success);
      await refreshStatus();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The action failed.");
    } finally {
      setBusy(false);
    }
  };

  const startCalibration = async () => {
    const replace = profileAvailable
      ? window.confirm("A calibration profile already exists. Replace it with a new webcam calibration?")
      : false;
    if (profileAvailable && !replace) return;
    await runAction(() => api.calibrate(replace), "Calibration started.");
  };

  const submitLabel = async (windowInfo: PendingWindow) => {
    if (labelChoice === "") return;
    setBusy(true);
    setError("");
    try {
      await api.label(windowInfo.session_id, windowInfo.window_id, Number(labelChoice));
      setPending((items) => items.filter((item) =>
        item.session_id !== windowInfo.session_id || item.window_id !== windowInfo.window_id
      ));
      setLabelChoice("");
      setNotice(`Window ${windowInfo.window_id} saved with your self-report.`);
      await refreshSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save this window.");
    } finally {
      setBusy(false);
    }
  };

  const skipWindow = async (windowInfo: PendingWindow) => {
    try {
      await api.skipWindow(windowInfo.session_id, windowInfo.window_id);
      setPending((items) => items.filter((item) =>
        item.session_id !== windowInfo.session_id || item.window_id !== windowInfo.window_id
      ));
      setLabelChoice("");
      setNotice(`Window ${windowInfo.window_id} was not saved.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not dismiss this window.");
    }
  };

  const downloadCsv = async (id?: string) => {
    setError("");
    try {
      const blob = await api.exportCsv(id);
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = `smarteye-${id ?? "all-sessions"}.csv`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 1000);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "CSV export failed.");
    }
  };

  const choosePage = (target: Page) => {
    setPage(target);
    setMobileNav(false);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Eye size={21} /></div>
          <div><strong>SmartEye</strong><span>VISION MONITORING</span></div>
          <button className="icon-button mobile-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => choosePage(id)}>
              <Icon size={17} aria-hidden="true" /><span>{label}</span>
              {id === "live" && status.session_active && <i className="nav-live-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="side-status">
            <span className={`status-dot ${status.backend_connected ? "online" : "offline"}`} />
            <div><strong>Backend</strong><small>{status.backend_connected ? "Connected" : "Disconnected"}</small></div>
            {status.backend_connected ? <Wifi size={15} /> : <WifiOff size={15} />}
          </div>
          <div className="side-status">
            <span className={`status-dot ${status.session_active ? "online" : ""}`} />
            <div><strong>Session</strong><small>{status.session_active ? "Monitoring" : "Inactive"}</small></div>
            <Radio size={15} />
          </div>
          <div className="side-version"><ShieldCheck size={14} /> Prototype · local processing</div>
        </div>
      </aside>
      {mobileNav && <button className="nav-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation overlay" />}

      <main className="main-area">
        <header className="topbar">
          <div className="title-wrap">
            <button className="icon-button mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button>
            <div><div className="eyebrow">SMART EYE FATIGUE MONITOR</div><h1>{pageTitle}</h1></div>
          </div>
          <div className="topbar-right">
            <div className="header-chip"><span className={`status-dot ${status.camera_connected ? "online" : ""}`} />Camera <b>{status.camera_connected ? "Ready" : "Off"}</b></div>
            <div className="header-chip"><span className={`status-dot ${status.backend_connected ? "online" : "offline"}`} />API <b>{status.backend_connected ? "Online" : "Offline"}</b></div>
            <button className="avatar-button" aria-label="Local participant"><UserRound size={17} /></button>
          </div>
        </header>

        <div className="content">
          {(error || status.error) && (
            <div className="alert error-alert" role="alert">
              <AlertCircle size={17} /><span>{error || status.error}</span>
              <button onClick={() => { setError(""); void refreshStatus(); }} aria-label="Dismiss error"><X size={16} /></button>
            </div>
          )}
          {notice && <div className="alert success-alert" role="status"><Check size={17} /><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss message"><X size={16} /></button></div>}

          <div className="page-intro">
            <div><div className="eyebrow">{page === "live" ? "REAL-TIME ANALYSIS" : page === "calibration" ? "PERSONAL BASELINE" : page === "history" ? "SAVED DATA" : "MONITORING OVERVIEW"}</div><h2>{page === "overview" ? "Your monitoring at a glance" : pageTitle}</h2><p>{page === "calibration" ? "Calibrate your eye measurements before starting a live session." : "Live measurements from the existing SmartEye vision pipeline."}</p></div>
            <div className="page-actions">
              {status.mode === "monitoring" || status.mode === "error"
                ? <button className="button danger-button" disabled={busy} onClick={() => void runAction(api.stop, "Monitoring stopped.")}><X size={16} />Stop monitoring</button>
                : <button className="button primary-button" disabled={busy || !status.backend_connected || !profileAvailable || status.mode !== "idle"} title={!profileAvailable ? "Complete personal calibration first" : ""} onClick={() => void runAction(api.start, "Monitoring started.")}>{busy ? <LoaderCircle className="spin" size={16} /> : <MonitorPlay size={16} />}Start monitoring</button>}
              <span className={`session-pill ${status.session_active ? "session-on" : ""}`}><i />{status.session_active ? "SESSION ACTIVE" : "SESSION INACTIVE"}</span>
            </div>
          </div>

          {page === "calibration" ? (
            <CalibrationView
              profile={profile}
              available={profileAvailable}
              status={status}
              busy={busy}
              refresh={refreshProfile}
              start={startCalibration}
              cancel={() => void runAction(api.cancelCalibration, "Calibration cancellation requested.")}
            />
          ) : page === "history" ? (
            <HistoryView sessions={sessions} selected={selectedSession} detail={sessionDetail} select={setSelectedSession} download={downloadCsv} />
          ) : page === "analytics" ? (
            <AnalyticsView sessions={sessions} data={visibleHistory} sampleCount={sampleCount} setSampleCount={setSampleCount} />
          ) : page === "settings" ? (
            <SettingsView status={status} profileAvailable={profileAvailable} refresh={refreshStatus} />
          ) : (
            <MonitoringView
              metrics={metrics}
              status={status}
              history={visibleHistory}
              sampleCount={sampleCount}
              setSampleCount={setSampleCount}
              elapsed={elapsed}
              download={() => void downloadCsv(status.session_id ?? undefined)}
            />
          )}

          <footer className="disclaimer"><ShieldCheck size={14} /><span>SmartEye provides a prototype fatigue indicator for demonstration only. It is not a medical diagnosis.</span><span className="footer-separator">•</span><span>Local webcam processing</span></footer>
        </div>
      </main>

      {pending.length > 0 && (
        <div className="modal-backdrop" role="presentation">
          <section className="label-modal" role="dialog" aria-modal="true" aria-labelledby="label-title">
            <div className="modal-icon"><BookOpenCheck size={20} /></div>
            <h2 id="label-title">Window ready to save</h2>
            <p>Window {pending[0].window_id} has real measurements. Add your self-reported fatigue label; it will be stored separately from the live prediction.</p>
            <label htmlFor="self-report">How tired did you feel during this window?</label>
            <select id="self-report" value={labelChoice} onChange={(event) => setLabelChoice(event.target.value)}>
              <option value="">Select a self-report…</option>
              <option value="0">0 · Not tired</option>
              <option value="1">1 · Tired</option>
              <option value="2">2 · Very tired</option>
            </select>
            <div className="modal-actions">
              <button className="button quiet-button" disabled={busy} onClick={() => void skipWindow(pending[0])}>Skip saving</button>
              <button className="button primary-button" disabled={busy || labelChoice === ""} onClick={() => void submitLabel(pending[0])}>{busy ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}Save window</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function MonitoringView({ metrics, status, history, sampleCount, setSampleCount, elapsed, download }: {
  metrics: Metrics | null; status: Status; history: Metrics[]; sampleCount: number;
  setSampleCount: (value: number) => void; elapsed: number; download: () => void;
}) {
  const ear = metrics?.ear;
  const blink = metrics?.blink;
  const closure = metrics?.eye_closure;
  const fatigue = metrics?.fatigue;
  const available = metrics?.prediction_available && fatigue?.score != null;
  const validSeconds = metrics?.observation.valid_duration_seconds ?? 0;
  const requiredSeconds = metrics?.observation.minimum_required_seconds ?? 10;
  const predictionStatus = available
    ? fatigue.category
    : status.mode === "error"
      ? "Camera error"
      : !status.session_active
        ? "Start monitoring"
        : !status.camera_connected
          ? "Connecting camera"
          : !status.face_detected
            ? "Finding face"
            : validSeconds < requiredSeconds
              ? `Collecting ${validSeconds.toFixed(1)} / ${requiredSeconds} sec`
              : "Calculating";
  const assessmentMessage = available
    ? "Actual live pipeline output. Not a medical diagnosis."
    : status.error
      ? status.error
      : status.mode === "error"
        ? "Monitoring stopped because the camera pipeline encountered an error."
        : !status.session_active
          ? "Start monitoring to begin live fatigue scoring."
          : !status.camera_connected
            ? "Waiting for the backend to connect to the webcam."
            : !status.face_detected
              ? "Center your face in the camera view."
              : `Keep your face visible until ${requiredSeconds} seconds of valid eye measurements are collected.`;
  return (
    <>
      <div className="session-strip panel">
        <div className="session-info"><span className="live-wave"><AudioLines size={17} /></span><div><small>ACTIVE SESSION</small><strong>{status.session_id ?? "No session started"}</strong></div></div>
        <div className="session-meta"><span><Clock3 size={15} />{formatDuration(elapsed)}</span><span><Activity size={15} />Window {status.window_id}</span><span><UserRound size={15} />Participant P01</span></div>
        <button className="icon-text-button" onClick={download} disabled={!status.session_id}><ArrowDownToLine size={15} />Export CSV</button>
      </div>

      <div className="monitor-grid">
        <section className="panel camera-panel">
          <div className="panel-heading"><div><h3>Live camera</h3><p>Processed locally by the SmartEye backend</p></div><span className={`camera-state ${status.camera_connected ? "camera-on" : ""}`}><i />{status.camera_connected ? "CONNECTED" : "OFFLINE"}</span></div>
          <div className="camera-view">
            {status.camera_connected
              ? <img src={`${API_URL}/api/camera/stream`} alt="Live webcam feed processed by SmartEye" />
              : <div className="camera-empty"><div className="camera-empty-icon"><Camera size={24} /></div><strong>{status.error ? "Camera unavailable" : "Camera is not active"}</strong><span>{status.error ?? (status.session_active ? "Connecting to webcam…" : "Start monitoring to connect the webcam.")}</span></div>}
            <div className="camera-overlay"><span><i className={status.face_detected ? "face-ok" : ""} />{status.face_detected ? "FACE DETECTED" : "NO FACE"}</span><span>{metrics?.timestamp ? timeLabel(metrics.timestamp) : "—"}</span></div>
          </div>
          <div className="camera-caption"><span><i className={`status-dot ${status.face_detected ? "online" : ""}`} />{status.face_detected ? "Face tracking active" : "Waiting for a valid face measurement"}</span><span>EAR threshold {status.ear_threshold == null ? "Unavailable" : status.ear_threshold.toFixed(3)}</span></div>
        </section>

        <section className="panel fatigue-panel">
          <div className="panel-heading"><div><h3>Fatigue assessment</h3><p>Rule-based prototype score · scale 0–1</p></div><span className="assessment-icon"><Gauge size={17} /></span></div>
          <div className={`fatigue-score ${available ? `fatigue-${fatigue.category.toLowerCase()}` : ""}`}>
            <strong>{available ? formatValue(fatigue.score, 3) : "—"}</strong><span>/ 1.00</span>
          </div>
          <div className="score-track"><i style={{ width: available ? `${Math.max(0, Math.min(100, (fatigue.score ?? 0) * 100))}%` : "0%" }} /></div>
          <div className="score-label-row"><span>Prediction</span><strong className={`category category-${available ? fatigue.category.toLowerCase() : "unavailable"}`}>{predictionStatus}</strong></div>
          <div className="assessment-note"><ShieldCheck size={15} /><span>{assessmentMessage}</span></div>
          <div className="assessment-stats"><div><small>Valid observations</small><strong>{formatValue(metrics?.observation.valid_duration_seconds, 1)}<em> sec</em></strong></div><div><small>Window progress</small><strong>{formatValue(metrics?.observation.window_duration_seconds, 0)}<em> sec</em></strong></div></div>
        </section>
      </div>

      <section className="metric-section">
        <div className="section-heading"><div><div className="eyebrow">LIVE PIPELINE OUTPUT</div><h3>Eye measurements</h3></div><span className="updated-at">{metrics?.timestamp ? `Updated ${timeLabel(metrics.timestamp)}` : "No live samples"}</span></div>
        <div className="metric-grid">
          <MetricCard label="Current EAR" value={formatValue(ear?.current, 3)} icon={Eye} detail="Instantaneous eye aspect ratio measured from face landmarks." />
          <MetricCard label="Mean EAR" value={formatValue(ear?.mean, 3)} icon={Activity} detail="Mean EAR across the rolling temporal analysis window." />
          <MetricCard label="Minimum EAR" value={formatValue(ear?.minimum, 3)} icon={Eye} detail="Minimum EAR observed in the rolling temporal analysis window." />
          <MetricCard label="Maximum EAR" value={formatValue(ear?.maximum, 3)} icon={Eye} detail="Maximum EAR observed in the rolling temporal analysis window." />
          <MetricCard label="EAR deviation" value={formatValue(ear?.std, 3)} icon={Activity} detail="Population standard deviation of the valid current feature-window EAR measurements." />
          <MetricCard label="EAR range" value={formatValue(ear?.range, 3)} icon={Activity} detail="Difference between maximum and minimum rolling EAR." />
        </div>
      </section>

      <section className="metric-section">
        <div className="section-heading"><div><div className="eyebrow">BEHAVIOR SIGNALS</div><h3>Blink analysis & eye closure</h3></div></div>
        <div className="metric-grid">
          <MetricCard label="Blink count" value={blink == null ? "—" : String(blink.count)} icon={Activity} detail="Completed blinks observed in the current feature window." />
          <MetricCard label="Blink rate" value={formatValue(blink?.rate_per_minute, 1)} unit=" / min" icon={Activity} detail="Blink frequency per minute based on valid observed time." />
          <MetricCard label="Mean blink duration" value={formatValue(blink?.mean_duration_seconds, 3)} unit=" sec" icon={Clock3} detail="Mean duration of completed blinks in the current window." />
          <MetricCard label="Maximum blink duration" value={formatValue(blink?.maximum_duration_seconds, 3)} unit=" sec" icon={Clock3} detail="Longest completed blink duration in the current window." />
          <MetricCard label="PERCLOS" value={formatValue(closure?.perclos_percent, 1)} unit="%" icon={Eye} detail="Percent of observed time below the personalized EAR threshold." />
          <MetricCard label="Current closure" value={formatValue(closure?.current_duration_seconds, 2)} unit=" sec" icon={Clock3} detail="Duration of the currently tracked eye closure." />
          <MetricCard label="Long closure" value={closure == null ? "—" : closure.long_closure ? "Detected" : "No"} icon={Bell} detail="True when a current eye closure reaches the configured duration." />
        </div>
      </section>

      <div className="charts-toolbar"><div><div className="eyebrow">REAL-TIME TRENDS</div><h3>Measurement history</h3></div><label>Recent samples <select value={sampleCount} onChange={(event) => setSampleCount(Number(event.target.value))}><option value={30}>30</option><option value={60}>60</option><option value={120}>120</option><option value={240}>240</option></select></label></div>
      <div className="chart-grid">
        <TrendChart title="Eye aspect ratio" data={history} dataKey="ear" color="#48d7c0" unit="EAR" />
        <TrendChart title="Fatigue score" data={history} dataKey="score" color="#86aaff" unit="score" domain={[0, 1]} />
        <TrendChart title="PERCLOS" data={history} dataKey="perclos" color="#d3a6ff" unit="%" domain={[0, 100]} />
        <TrendChart title="Blink rate" data={history} dataKey="blink" color="#f0b768" unit="blinks/min" />
      </div>
    </>
  );
}

function CalibrationView({ profile, available, status, busy, refresh, start, cancel }: {
  profile: Record<string, number | string> | null; available: boolean; status: Status;
  busy: boolean; refresh: () => Promise<void>; start: () => Promise<void>; cancel: () => void;
}) {
  const values: Array<[string, string, string]> = [
    ["Baseline EAR", "baseline_ear", ""],
    ["Personal EAR threshold", "ear_threshold", ""],
    ["Baseline blink rate", "baseline_blink_rate", "blinks/min"],
    ["Baseline closure", "baseline_closure_duration", "sec"],
    ["Calibration duration", "calibration_duration_seconds", "sec"],
    ["Calibration date", "calibration_timestamp", ""]
  ];
  return (
    <div className="calibration-layout">
      <section className="panel calibration-card">
        <div className="panel-heading"><div><h3>Personal calibration</h3><p>Profile loaded by the live analysis pipeline</p></div><span className={`profile-badge ${available ? "available" : ""}`}><i />{available ? "PROFILE READY" : "NOT CALIBRATED"}</span></div>
        {available && profile ? <div className="profile-grid">
          {values.map(([label, key, unit]) => {
            const value = profile[key];
            const shown = typeof value === "number" ? value.toFixed(key.includes("rate") || key.includes("duration") ? 2 : 4) : value ?? "—";
            return <div className="profile-value" key={key}><small>{label}</small><strong>{shown}<em>{unit ? ` ${unit}` : ""}</em></strong></div>;
          })}
        </div> : <div className="profile-empty"><SlidersHorizontal size={26} /><strong>No saved profile available</strong><span>Start a camera calibration to establish your personal baseline.</span></div>}
        <div className="calibration-footer">
          <div className="calibration-state"><span className={`status-dot ${status.mode === "calibrating" ? "online pulse" : ""}`} /><span>{status.calibration_message ?? (status.mode === "calibrating" ? "Calibration in progress" : available ? "Profile loaded from disk" : "Waiting to calibrate")}</span></div>
          <button className="button quiet-button" onClick={() => void refresh()}><RefreshCw size={15} />Reload profile</button>
        </div>
        {status.mode === "calibrating" && <div className="progress-wrap"><div><span>Calibration progress</span><b>{Math.round(status.calibration_progress * 100)}%</b></div><div className="progress-track"><i style={{ width: `${status.calibration_progress * 100}%` }} /></div><button className="text-button" disabled={busy} onClick={cancel}>Cancel calibration</button></div>}
      </section>
      <section className="panel calibration-instructions">
        <div className="instructions-icon"><Camera size={20} /></div><div className="eyebrow">30-SECOND PROCESS</div><h3>Get a reliable baseline</h3>
        <ol><li><b>Position yourself</b><span>Face the camera with both eyes clearly visible.</span></li><li><b>Look naturally</b><span>Remain alert and blink normally while measurements are collected.</span></li><li><b>Use the personalized threshold</b><span>The saved EAR threshold configures blink, PERCLOS, and temporal analysis.</span></li></ol>
        <button className="button primary-button full-button" disabled={busy || status.mode !== "idle"} onClick={() => void start()}>{status.mode === "calibrating" ? <LoaderCircle className="spin" size={16} /> : <SlidersHorizontal size={16} />}{available ? "Replace calibration profile" : "Start calibration"}</button>
        <p className="fine-print"><ShieldCheck size={14} />Existing profiles are not replaced without confirmation.</p>
      </section>
      {status.error && <div className="alert error-alert"><AlertCircle size={17} />{status.error}</div>}
    </div>
  );
}

function HistoryView({ sessions, selected, detail, select, download }: {
  sessions: Session[]; selected: string; detail: SessionDetail | null;
  select: (id: string) => void; download: (id?: string) => void;
}) {
  return (
    <>
      <section className="panel history-panel">
        <div className="panel-heading"><div><h3>Recorded sessions</h3><p>Only windows with a submitted self-report are saved to the CSV dataset.</p></div><button className="icon-text-button" disabled={!sessions.length} onClick={() => download()}><ArrowDownToLine size={15} />Export all CSV</button></div>
        {sessions.length ? <div className="table-scroll"><table><thead><tr><th>Session / date</th><th>Participant</th><th>Windows</th><th>Predictions</th><th>Avg score</th><th>Avg EAR</th><th>Avg blink rate</th><th>Avg PERCLOS</th><th /></tr></thead><tbody>
          {sessions.map((session) => <tr key={session.session_id} className={selected === session.session_id ? "selected-row" : ""} onClick={() => select(session.session_id)}>
            <td><strong>{session.session_id}</strong><small className="table-subtext">{session.started_at ? new Date(session.started_at).toLocaleString() : "Date unavailable"}</small></td><td>{session.participant_id}</td><td>{session.windows}</td><td>{session.predictions_available} / {session.windows}</td><td>{formatValue(session.average_fatigue_score, 3)}</td><td>{formatValue(session.average_ear, 3)}</td><td>{formatValue(session.average_blink_rate, 1)}</td><td>{formatValue(session.average_perclos, 1)}{session.average_perclos == null ? "" : "%"}</td><td><button className="table-action" aria-label={`Export ${session.session_id}`} onClick={(event) => { event.stopPropagation(); void download(session.session_id); }}><ArrowDownToLine size={15} /></button></td>
          </tr>)}
        </tbody></table></div> : <div className="empty-chart"><Clock3 size={22} /><strong>No saved sessions yet</strong><span>Completed windows appear here after you add a self-report.</span></div>}
      </section>
      {selected && <section className="panel history-panel"><div className="panel-heading"><div><h3>Session windows</h3><p>{selected}</p></div><button className="icon-text-button" onClick={() => download(selected)}><ArrowDownToLine size={15} />Export session</button></div>
        {detail?.windows.length ? <div className="table-scroll"><table><thead><tr><th>Window</th><th>Self-report</th><th>Predicted status</th><th>Score</th><th>Mean EAR</th><th>Blink rate</th><th>PERCLOS</th></tr></thead><tbody>{detail.windows.map((row) => <tr key={`${row.session_id}-${row.window_id}`}><td>{row.window_id}</td><td>{row.fatigue_label || "Unavailable"}</td><td>{row.predicted_status || "Unavailable"}</td><td>{formatValue(NumberOrNull(row.fatigue_score), 3)}</td><td>{formatValue(NumberOrNull(row.mean_ear), 3)}</td><td>{formatValue(NumberOrNull(row.blink_rate), 1)}</td><td>{formatValue(NumberOrNull(row.perclos), 1)}%</td></tr>)}</tbody></table></div> : <div className="empty-chart">Loading session windows…</div>}
      </section>}
    </>
  );
}

function NumberOrNull(value: string | undefined): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function AnalyticsView({ sessions, data, sampleCount, setSampleCount }: {
  sessions: Session[]; data: Metrics[]; sampleCount: number; setSampleCount: (value: number) => void;
}) {
  const totals = sessions.reduce((sum, item) => sum + item.windows, 0);
  const scored = sessions.filter((item) => item.average_fatigue_score != null).length;
  return (
    <>
      <div className="analytics-summary">
        <MetricCard label="Saved sessions" value={String(sessions.length)} icon={Clock3} detail="Sessions present in the saved CSV records." />
        <MetricCard label="Saved windows" value={String(totals)} icon={Activity} detail="Feature windows with an explicit self-reported label." />
        <MetricCard label="Sessions with predictions" value={`${scored} / ${sessions.length}`} icon={Gauge} detail="Sessions with at least one real recorded fatigue score." />
      </div>
      <div className="charts-toolbar"><div><div className="eyebrow">LIVE SESSION HISTORY</div><h3>Real measurement trends</h3></div><label>Recent samples <select value={sampleCount} onChange={(event) => setSampleCount(Number(event.target.value))}><option value={30}>30</option><option value={60}>60</option><option value={120}>120</option><option value={240}>240</option></select></label></div>
      <div className="chart-grid">
        <TrendChart title="Eye aspect ratio" data={data} dataKey="ear" color="#48d7c0" unit="EAR" />
        <TrendChart title="Fatigue score" data={data} dataKey="score" color="#86aaff" unit="score" domain={[0, 1]} />
        <TrendChart title="PERCLOS" data={data} dataKey="perclos" color="#d3a6ff" unit="%" domain={[0, 100]} />
        <TrendChart title="Blink rate" data={data} dataKey="blink" color="#f0b768" unit="blinks/min" />
      </div>
      <p className="fine-print"><AlertCircle size={14} />Charts use actual samples from the current live session; saved CSV history is shown in Session History.</p>
    </>
  );
}

function SettingsView({ status, profileAvailable, refresh }: {
  status: Status; profileAvailable: boolean; refresh: () => Promise<void>;
}) {
  return (
    <section className="panel settings-panel"><div className="panel-heading"><div><h3>Runtime & data</h3><p>Local application configuration and service health</p></div></div>
      <div className="settings-list">
        <div><span><Wifi size={17} />Backend API</span><strong className={status.backend_connected ? "text-online" : "text-offline"}>{status.backend_connected ? "Connected · localhost:8000" : "Disconnected"}</strong></div>
        <div><span><Camera size={17} />Camera pipeline</span><strong>{status.camera_connected ? "Connected" : "Not active"}</strong></div>
        <div><span><SlidersHorizontal size={17} />Personal calibration</span><strong>{profileAvailable ? `Loaded · threshold ${status.ear_threshold?.toFixed(4) ?? "available"}` : "Profile not available"}</strong></div>
        <div><span><Activity size={17} />Live inference</span><strong>Existing rule-based fatigue score</strong></div>
        <div><span><ShieldCheck size={17} />Data storage</span><strong>dataset/fatigue_data.csv · local</strong></div>
      </div>
      <div className="settings-foot"><p>Live Monitoring uses the webcam pipeline in the Python backend. The frontend does not generate or simulate live predictions.</p><button className="button quiet-button" onClick={() => void refresh()}><RefreshCw size={15} />Refresh status</button></div>
    </section>
  );
}
