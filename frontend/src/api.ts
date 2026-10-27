import type { Metrics, Session, SessionDetail, Status } from "./types";

export const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
export const WS_URL = API_URL.replace(/^http/, "ws") + "/ws/monitoring";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  status: () => request<Status>("/api/status"),
  metrics: () => request<Metrics | { available: false }>("/api/metrics/current"),
  history: () => request<Metrics[]>("/api/metrics/history"),
  sessions: () => request<Session[]>("/api/sessions"),
  session: (id: string) => request<SessionDetail>(`/api/sessions/${encodeURIComponent(id)}`),
  profile: () => request<{ available: boolean; profile: Record<string, number | string> | null }>("/api/calibration/profile"),
  start: () => request<Status>("/api/session/start", { method: "POST" }),
  stop: () => request<Status>("/api/session/stop", { method: "POST" }),
  calibrate: (confirmReplace: boolean) => request<Status>("/api/calibration/start", {
    method: "POST",
    body: JSON.stringify({ confirm_replace: confirmReplace })
  }),
  cancelCalibration: () => request<Status>("/api/calibration/cancel", { method: "POST" }),
  label: (sessionId: string, windowId: number, fatigueLabel: number) =>
    request<{ saved: boolean }>("/api/session/window/label", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, window_id: windowId, fatigue_label: fatigueLabel })
    }),
  skipWindow: (sessionId: string, windowId: number) =>
    request<{ skipped: boolean }>("/api/session/window/skip", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, window_id: windowId })
    }),
  sessionExport: (id?: string) =>
    `${API_URL}/api/export/csv${id ? `?session_id=${encodeURIComponent(id)}` : ""}`,
  exportCsv: async (id?: string): Promise<Blob> => {
    const response = await fetch(api.sessionExport(id));
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { detail?: string } | null;
      throw new Error(body?.detail ?? `CSV export failed (${response.status})`);
    }
    return response.blob();
  }
};
