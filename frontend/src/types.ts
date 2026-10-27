export interface Status {
  backend_connected: boolean;
  mode: "idle" | "monitoring" | "calibrating" | "error" | string;
  session_active: boolean;
  session_id: string | null;
  session_elapsed_seconds: number;
  window_id: number;
  camera_connected: boolean;
  face_detected: boolean;
  calibration_available: boolean;
  ear_threshold: number | null;
  error: string | null;
  calibration_progress: number;
  calibration_message: string | null;
}

export interface Metrics {
  session_id: string | null;
  window_id: number;
  timestamp: string;
  face_detected: boolean;
  prediction_available: boolean;
  ear: null | {
    current: number | null;
    mean: number | null;
    minimum: number | null;
    maximum: number | null;
    std: number | null;
    range: number | null;
  };
  blink: null | {
    count: number;
    rate_per_minute: number | null;
    mean_duration_seconds: number | null;
    maximum_duration_seconds: number | null;
  };
  eye_closure: null | {
    perclos_percent: number | null;
    current_duration_seconds: number | null;
    long_closure: boolean;
  };
  fatigue: { score: number | null; category: string };
  observation: {
    window_duration_seconds: number;
    valid_duration_seconds: number | null;
    minimum_required_seconds: number;
  };
}

export interface Session {
  session_id: string;
  participant_id: string;
  windows: number;
  predictions_available: number;
  started_at: string | null;
  average_fatigue_score: number | null;
  average_ear: number | null;
  average_blink_rate: number | null;
  average_perclos: number | null;
  categories: Record<string, number>;
}

export interface SessionDetail {
  session_id: string;
  windows: Array<Record<string, string>>;
}
