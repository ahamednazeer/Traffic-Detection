/**
 * Traffic Detection API Client
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Detection {
  class: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  class_id: number;
}

export interface DetectionStats {
  total_objects: number;
  unique_classes: number;
  avg_confidence: number;
  class_counts: Record<string, number>;
  has_pedestrians: boolean;
  has_vehicles: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  loaded: boolean;
}

export interface ImageDetectionResponse {
  success: boolean;
  model_used: string;
  annotated_image: string;
  detections: Detection[];
  statistics: DetectionStats;
}

export interface VideoDetectionResponse {
  success: boolean;
  model_used: string;
  video_base64: string;
  video_info: {
    fps: number;
    total_frames: number;
    processed_frames: number;
    width: number;
    height: number;
    duration_seconds: number;
  };
  statistics: DetectionStats;
}

export interface ModelsResponse {
  models: ModelInfo[];
  active: string;
  class_names: string[];
  class_colors: Record<string, string>;
}

class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async health(): Promise<{ status: string; active_model: string }> {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error('API health check failed');
    return res.json();
  }

  async getModels(): Promise<ModelsResponse> {
    const res = await fetch(`${this.baseUrl}/api/models`);
    if (!res.ok) throw new Error('Failed to get models');
    return res.json();
  }

  async selectModel(model: string): Promise<{ status: string; active_model: string }> {
    const res = await fetch(`${this.baseUrl}/api/models/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    if (!res.ok) throw new Error('Failed to select model');
    return res.json();
  }

  async detectImage(
    file: File,
    confidence: number = 0.5,
    model?: string
  ): Promise<ImageDetectionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', confidence.toString());
    if (model) formData.append('model', model);

    const res = await fetch(`${this.baseUrl}/api/detect/image`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Image detection failed');
    }
    return res.json();
  }

  async detectVideo(
    file: File,
    confidence: number = 0.5,
    model?: string,
    skipFrames: number = 0
  ): Promise<VideoDetectionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', confidence.toString());
    formData.append('skip_frames', skipFrames.toString());
    if (model) formData.append('model', model);

    const res = await fetch(`${this.baseUrl}/api/detect/video`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Video detection failed');
    }
    return res.json();
  }
}

export const api = new APIClient();
export default api;
