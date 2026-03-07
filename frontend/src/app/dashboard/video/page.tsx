'use client';

import React, { useEffect, useState, useRef, useMemo } from 'react';
import ModelSelector from '@/components/ModelSelector';
import DetectionStatsPanel from '@/components/DetectionStats';
import AccidentStatus from '@/components/AccidentStatus';
import type { DetectionStats, AccidentResult, AccidentTimelineEntry, AccidentClip } from '@/lib/api';
import {
    Upload,
    Play,
    ArrowsClockwise,
    VideoCamera,
} from '@phosphor-icons/react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const CHUNK_SIZE = 1024 * 1024; // 1MB chunks

export default function VideoDetectionPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [resultVideo, setResultVideo] = useState<string | null>(null);
    const [previewFrame, setPreviewFrame] = useState<string | null>(null);
    const [stats, setStats] = useState<DetectionStats | null>(null);
    const [accident, setAccident] = useState<AccidentResult | null>(null);
    const [accidentTimeline, setAccidentTimeline] = useState<AccidentTimelineEntry[]>([]);
    const [accidentPeaks, setAccidentPeaks] = useState<AccidentTimelineEntry[]>([]);
    const [accidentClip, setAccidentClip] = useState<(AccidentClip & { url: string }) | null>(null);
    const [activeModel, setActiveModel] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [confidence, setConfidence] = useState(0.5);
    const [skipFrames, setSkipFrames] = useState(0);
    const [livePreview, setLivePreview] = useState(true);
    const [previewFps, setPreviewFps] = useState(10);
    const [clipSeconds, setClipSeconds] = useState(8);
    const [emailNotifications, setEmailNotifications] = useState(false);
    const [notifyEmail, setNotifyEmail] = useState('');
    const [defaultNotifyEmail, setDefaultNotifyEmail] = useState('');
    const [emailStatus, setEmailStatus] = useState<{ status: string; reason?: string; recipient?: string | null } | null>(null);
    const [previewFpsActual, setPreviewFpsActual] = useState(0);
    const [progress, setProgress] = useState(0);
    const [progressStatus, setProgressStatus] = useState<string | null>(null);
    const [videoInfo, setVideoInfo] = useState<any>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const cancelRef = useRef(false);
    const resultVideoRef = useRef<HTMLVideoElement>(null);
    const previewFrameCountRef = useRef(0);
    const previewLastTimeRef = useRef(Date.now());
    const pollRef = useRef<number | null>(null);
    const jobIdRef = useRef<string | null>(null);
    const progressRef = useRef(0);

    useEffect(() => {
        let cancelled = false;

        const loadDefaultNotifyEmail = async () => {
            try {
                const res = await fetch(`${API_URL}/api/health`);
                if (!res.ok) return;
                const data = await res.json();
                const envEmail = typeof data?.default_notify_email === 'string'
                    ? data.default_notify_email.trim()
                    : '';
                if (!envEmail || cancelled) return;
                setDefaultNotifyEmail(envEmail);
                setNotifyEmail((prev) => (prev.trim() ? prev : envEmail));
            } catch {
                // ignore config lookup errors
            }
        };

        loadDefaultNotifyEmail();
        return () => {
            cancelled = true;
        };
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResultVideo(null);
            setStats(null);
            setAccident(null);
            setError(null);
            setVideoInfo(null);
            setProgress(0);
            setPreviewFrame(null);
            setAccidentTimeline([]);
            setAccidentPeaks([]);
            setAccidentClip(null);
            setEmailStatus(null);
        }
    };

    const handleProcess = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);
        setProgress(0);
        setProgressStatus('Connecting...');
        setPreviewFrame(null);
        setPreviewFpsActual(0);
        setAccidentTimeline([]);
        setAccidentPeaks([]);
        setAccidentClip(null);
        setEmailStatus(null);
        jobIdRef.current = null;
        progressRef.current = 0;
        if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
        }
        previewFrameCountRef.current = 0;
        previewLastTimeRef.current = Date.now();
        cancelRef.current = false;

        try {
            const ws = new WebSocket(`${WS_URL}/api/video/process`);
            wsRef.current = ws;
            const newJobId = (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            jobIdRef.current = newJobId;

            ws.onopen = async () => {
                setProgressStatus('Sending video...');

                // Send metadata first
                ws.send(JSON.stringify({
                    size: selectedFile.size,
                    confidence,
                    skip_frames: skipFrames,
                    model: activeModel || 'accident_yolo11x',
                    job_id: newJobId,
                    preview: livePreview,
                    preview_fps: previewFps,
                    preview_width: 640,
                    clip_seconds: clipSeconds,
                    email_notifications: emailNotifications,
                    notify_email: (notifyEmail.trim() || defaultNotifyEmail).trim(),
                    filename: selectedFile.name
                }));
            };

            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);

                if (data.error) {
                    if (cancelRef.current) {
                        return;
                    }
                    setError(data.error);
                    setLoading(false);
                    ws.close();
                    return;
                }

                if (data.type === 'ready') {
                    if (data.job_id) {
                        jobIdRef.current = data.job_id;
                        if (pollRef.current) {
                            window.clearInterval(pollRef.current);
                        }
                        pollRef.current = window.setInterval(() => {
                            pollJob(data.job_id);
                        }, 5000);
                        pollJob(data.job_id);
                    }
                    // Start sending chunks
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64 = (reader.result as string).split(',')[1];

                        // Send in chunks
                        for (let i = 0; i < base64.length; i += CHUNK_SIZE) {
                            const chunk = base64.slice(i, i + CHUNK_SIZE);
                            ws.send(chunk);
                        }
                    };
                    reader.readAsDataURL(selectedFile);
                }

                if (data.type === 'upload') {
                    setProgressStatus(`Uploading: ${data.progress}%`);
                }

                if (data.type === 'status') {
                    setProgressStatus(data.message);
                    if (data.progress !== undefined) {
                        setProgress(data.progress);
                    }
                }

                if (data.type === 'progress') {
                    if (data.progress >= progressRef.current) {
                        progressRef.current = data.progress;
                        setProgress(data.progress);
                    }
                    if (data.frame && data.total) {
                        setProgressStatus(`Processing: ${data.frame}/${data.total} frames`);
                    }
                }

                if (data.type === 'preview') {
                    if (cancelRef.current) {
                        return;
                    }
                    setPreviewFrame(`data:image/jpeg;base64,${data.frame}`);
                    if (data.processed_index && data.total) {
                        const derivedProgress = Math.floor((data.processed_index / data.total) * 100);
                        if (derivedProgress >= progressRef.current) {
                            progressRef.current = derivedProgress;
                            setProgress(derivedProgress);
                            setProgressStatus(`Processing: ${data.processed_index}/${data.total} frames`);
                        }
                    }
                    previewFrameCountRef.current += 1;
                    const now = Date.now();
                    if (now - previewLastTimeRef.current >= 1000) {
                        setPreviewFpsActual(previewFrameCountRef.current);
                        previewFrameCountRef.current = 0;
                        previewLastTimeRef.current = now;
                    }
                }

                if (data.type === 'complete') {
                    if (cancelRef.current) {
                        return;
                    }
                    setResultVideo(`data:video/mp4;base64,${data.video_base64}`);
                    setStats(data.statistics);
                    setAccident(data.accident || null);
                    setAccidentTimeline(data.accident_timeline || []);
                    setAccidentPeaks(data.accident_peaks || []);
                    if (data.accident_clip && data.accident_clip.clip_base64) {
                        setAccidentClip({
                            ...data.accident_clip,
                            url: `data:video/mp4;base64,${data.accident_clip.clip_base64}`
                        });
                    } else {
                        setAccidentClip(null);
                    }
                    setEmailStatus(data.email_notification || null);
                    setVideoInfo(data.video_info);
                    setProgress(100);
                    setProgressStatus('Complete!');
                    setPreviewFrame(null);
                    setLoading(false);
                    setError(null);
                    progressRef.current = 100;
                    if (pollRef.current) {
                        window.clearInterval(pollRef.current);
                        pollRef.current = null;
                    }
                    ws.close();
                }
            };

            ws.onerror = (err) => {
                if (cancelRef.current) {
                    return;
                }
                console.error('WebSocket error:', err);
                if (loading && jobIdRef.current) {
                    setError('Connection failed. Continuing in background...');
                    setProgressStatus('Reconnecting...');
                    if (pollRef.current) {
                        window.clearInterval(pollRef.current);
                    }
                    pollRef.current = window.setInterval(() => {
                        pollJob(jobIdRef.current as string);
                    }, 3000);
                    pollJob(jobIdRef.current as string);
                } else {
                    setError('Connection failed. Is the backend running?');
                    setLoading(false);
                }
            };

            ws.onclose = () => {
                wsRef.current = null;
                if (!cancelRef.current && loading) {
                    setError('Connection closed during processing. Continuing in background...');
                    setProgressStatus('Reconnecting...');
                    if (jobIdRef.current) {
                        if (pollRef.current) {
                            window.clearInterval(pollRef.current);
                        }
                        pollRef.current = window.setInterval(() => {
                            pollJob(jobIdRef.current as string);
                        }, 3000);
                        pollJob(jobIdRef.current as string);
                    }
                }
            };

        } catch (err: any) {
            setError(err.message || 'Video processing failed');
            setLoading(false);
        }
    };

    const handleStop = () => {
        cancelRef.current = true;
        if (wsRef.current && wsRef.current.readyState < WebSocket.CLOSING) {
            wsRef.current.close();
        }
        if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
        }
        setLoading(false);
        setProgressStatus('Cancelled');
        setPreviewFrame(null);
        setPreviewFpsActual(0);
    };

    const pollJob = async (id: string) => {
        try {
            const res = await fetch(`${API_URL}/api/video/jobs/${id}`);
            if (!res.ok) {
                return;
            }
            const job = await res.json();
            if (job.preview) {
                setPreviewFrame(`data:image/jpeg;base64,${job.preview}`);
            }
            if (typeof job.progress === 'number' && job.progress >= progressRef.current) {
                progressRef.current = job.progress;
                setProgress(job.progress);
            }
            if (job.frame && job.total) {
                setProgressStatus(`Processing: ${job.frame}/${job.total} frames`);
            }
            if (job.status === 'complete' && job.result) {
                const result = job.result;
                setResultVideo(`data:video/mp4;base64,${result.video_base64}`);
                setStats(result.statistics);
                setAccident(result.accident || null);
                setAccidentTimeline(result.accident_timeline || []);
                setAccidentPeaks(result.accident_peaks || []);
                if (result.accident_clip && result.accident_clip.clip_base64) {
                    setAccidentClip({
                        ...result.accident_clip,
                        url: `data:video/mp4;base64,${result.accident_clip.clip_base64}`
                    });
                } else {
                    setAccidentClip(null);
                }
                setEmailStatus(result.email_notification || job.email_notification || null);
                setVideoInfo(result.video_info);
                setProgress(100);
                setProgressStatus('Complete!');
                setPreviewFrame(null);
                setLoading(false);
                setError(null);
                if (pollRef.current) {
                    window.clearInterval(pollRef.current);
                    pollRef.current = null;
                }
            }
            if (job.status === 'error') {
                setError(job.error || 'Processing failed');
                setLoading(false);
                if (pollRef.current) {
                    window.clearInterval(pollRef.current);
                    pollRef.current = null;
                }
            }
        } catch (err) {
            // ignore
        }
    };

    const formatTime = (seconds: number) => {
        if (!Number.isFinite(seconds)) return '--';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds - Math.floor(seconds)) * 100);
        if (mins > 0) {
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        return `${secs}.${ms.toString().padStart(2, '0')}s`;
    };

    const timelineBuckets = useMemo(() => {
        if (!accidentTimeline.length) return [];
        const bucketCount = 80;
        const buckets: { score: number; timestamp: number }[] = [];
        for (let i = 0; i < bucketCount; i++) {
            const start = Math.floor((i * accidentTimeline.length) / bucketCount);
            const end = Math.max(start + 1, Math.floor(((i + 1) * accidentTimeline.length) / bucketCount));
            let maxScore = 0;
            let maxTimestamp = accidentTimeline[start]?.timestamp ?? 0;
            for (let j = start; j < end && j < accidentTimeline.length; j++) {
                const entry = accidentTimeline[j];
                if (entry.score > maxScore) {
                    maxScore = entry.score;
                    maxTimestamp = entry.timestamp;
                }
            }
            buckets.push({ score: maxScore, timestamp: maxTimestamp });
        }
        return buckets;
    }, [accidentTimeline]);

    const jumpToTimestamp = (timestamp: number) => {
        if (!resultVideoRef.current) return;
        resultVideoRef.current.currentTime = Math.max(0, timestamp);
        resultVideoRef.current.play();
    };

    const formatEmailReason = (reason?: string) => {
        if (!reason) return 'n/a';
        return reason.replace(/_/g, ' ');
    };

    const accidentThreshold = accident?.threshold ?? 0.5;
    const effectiveNotifyEmail = (notifyEmail.trim() || defaultNotifyEmail).trim();
    const emailMissing = emailNotifications && effectiveNotifyEmail.length === 0;

    return (
        <div className="space-y-6 animate-slide-up">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-chivo font-bold text-2xl uppercase tracking-wider">
                        Accident Detection
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Process video files for accident detection
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Controls */}
                <div className="space-y-4">
                    <ModelSelector onModelChange={setActiveModel} />

                    {/* Confidence Slider */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Confidence Threshold
                        </h3>
                        <input
                            type="range"
                            min="0.01"
                            max="1"
                            step="0.01"
                            value={confidence}
                            onChange={(e) => setConfidence(parseFloat(e.target.value))}
                            className="w-full"
                            disabled={loading}
                        />
                        <div className="text-center text-blue-400 font-mono text-lg mt-2">
                            {(confidence * 100).toFixed(0)}%
                        </div>
                    </div>

                    {/* Skip Frames */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Skip Frames (Speed)
                        </h3>
                        <input
                            type="range"
                            min="0"
                            max="10"
                            step="1"
                            value={skipFrames}
                            onChange={(e) => setSkipFrames(parseInt(e.target.value))}
                            className="w-full"
                            disabled={loading}
                        />
                        <div className="text-center text-green-400 font-mono text-lg mt-2">
                            Process every {skipFrames + 1} frame(s)
                        </div>
                    </div>

                    {/* Clip Length */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Accident Clip Length
                        </h3>
                        <input
                            type="range"
                            min="5"
                            max="10"
                            step="1"
                            value={clipSeconds}
                            onChange={(e) => setClipSeconds(parseInt(e.target.value, 10))}
                            className="w-full"
                            disabled={loading}
                        />
                        <div className="text-center text-blue-400 font-mono text-lg mt-2">
                            {clipSeconds}s
                        </div>
                    </div>

                    {/* Live Preview */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Live Preview
                        </h3>
                        <label className="flex items-center justify-between text-sm text-slate-300">
                            <span>Show preview while processing</span>
                            <input
                                type="checkbox"
                                checked={livePreview}
                                onChange={(e) => setLivePreview(e.target.checked)}
                                disabled={loading}
                                className="h-4 w-4 accent-blue-500"
                            />
                        </label>
                        <div className="mt-3">
                            <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                                <span>Preview FPS</span>
                                <span>{previewFps} fps</span>
                            </div>
                            <input
                                type="range"
                                min="4"
                                max="20"
                                step="1"
                                value={previewFps}
                                onChange={(e) => setPreviewFps(parseInt(e.target.value, 10))}
                                className="w-full"
                                disabled={loading || !livePreview}
                            />
                        </div>
                    </div>

                    {/* Email Notification */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Email Notification
                        </h3>
                        <label className="flex items-center justify-between text-sm text-slate-300">
                            <span>Send one alert email if accident is detected</span>
                            <input
                                type="checkbox"
                                checked={emailNotifications}
                                onChange={(e) => setEmailNotifications(e.target.checked)}
                                disabled={loading}
                                className="h-4 w-4 accent-blue-500"
                            />
                        </label>
                        <input
                            type="email"
                            value={notifyEmail}
                            onChange={(e) => setNotifyEmail(e.target.value)}
                            placeholder={defaultNotifyEmail || 'ops@company.com'}
                            disabled={loading || !emailNotifications}
                            className="mt-3 w-full rounded-sm border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none"
                        />
                        {defaultNotifyEmail && (
                            <p className="mt-2 text-xs text-blue-400">
                                Default recipient from backend env: {defaultNotifyEmail}
                            </p>
                        )}
                        <p className="mt-2 text-xs text-slate-500">
                            Requires SMTP env vars in backend (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`).
                        </p>
                        {emailMissing && (
                            <p className="mt-2 text-xs text-red-400">
                                Enter recipient email to enable notifications.
                            </p>
                        )}
                    </div>

                    {/* Upload */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Upload Video
                        </h3>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            onChange={handleFileSelect}
                            className="hidden"
                            disabled={loading}
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="btn-secondary w-full flex items-center justify-center gap-2"
                            disabled={loading}
                        >
                            <Upload size={18} weight="duotone" />
                            {selectedFile ? selectedFile.name : 'Choose Video'}
                        </button>
                        {selectedFile && (
                            <p className="text-xs text-slate-500 mt-2">
                                Size: {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                            </p>
                        )}
                    </div>

                    {/* Process Button */}
                    <div className="flex gap-2">
                        <button
                            onClick={handleProcess}
                            disabled={!selectedFile || loading || emailMissing}
                            className="btn-primary flex-1 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <ArrowsClockwise size={18} className="animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Play size={18} weight="duotone" />
                                    Process Video
                                </>
                            )}
                        </button>
                        <button
                            onClick={handleStop}
                            disabled={!loading}
                            className="btn-secondary w-28 flex items-center justify-center gap-2"
                        >
                            Stop
                        </button>
                    </div>



                    {error && (
                        <div className="bg-red-950/30 border border-red-800 rounded-sm p-3 text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    {/* Video Info */}
                    {videoInfo && (
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Video Info
                            </h3>
                            <div className="space-y-1 text-sm">
                                <p><span className="text-slate-500">Duration:</span> {videoInfo.duration_seconds?.toFixed(1)}s</p>
                                <p><span className="text-slate-500">FPS:</span> {videoInfo.fps}</p>
                                <p><span className="text-slate-500">Frames:</span> {videoInfo.total_frames}</p>
                                <p><span className="text-slate-500">Processed:</span> {videoInfo.processed_frames}</p>
                            </div>
                        </div>
                    )}

                    {emailStatus && (
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Email Notification
                            </h3>
                            <div className="space-y-1 text-sm">
                                <p>
                                    <span className="text-slate-500">Status:</span>{' '}
                                    <span className={emailStatus.status === 'sent' ? 'text-green-400' : emailStatus.status === 'failed' ? 'text-red-400' : 'text-slate-300'}>
                                        {emailStatus.status}
                                    </span>
                                </p>
                                <p>
                                    <span className="text-slate-500">Reason:</span> {formatEmailReason(emailStatus.reason)}
                                </p>
                                {emailStatus.recipient && (
                                    <p>
                                        <span className="text-slate-500">Recipient:</span> {emailStatus.recipient}
                                    </p>
                                )}
                            </div>
                        </div>
                    )}

                    <AccidentStatus accident={accident} />
                    <DetectionStatsPanel stats={stats} />
                </div>

                {/* Right Column - Video */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Processed Video
                        </h3>
                        <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center">
                            {loading ? (
                                previewFrame ? (
                                    <div className="relative w-full h-full">
                                        <img
                                            src={previewFrame}
                                            alt="Live preview"
                                            className="w-full h-full object-contain"
                                        />
                                        <div className="absolute inset-x-0 bottom-0 bg-slate-900/70 px-3 py-2 text-xs text-slate-200 flex items-center justify-between">
                                            <span>{progressStatus || 'Processing...'}</span>
                                            <span className="flex items-center gap-3">
                                                <span>Preview {previewFpsActual} FPS</span>
                                                <span>{progress}%</span>
                                            </span>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center">
                                        <ArrowsClockwise size={40} className="text-blue-400 animate-spin mx-auto mb-2" />
                                        <p className="text-blue-400">{progressStatus}</p>
                                        <p className="text-slate-500 text-sm mt-1">{progress}% complete</p>
                                    </div>
                                )
                            ) : resultVideo ? (
                                <video
                                    ref={resultVideoRef}
                                    src={resultVideo}
                                    controls
                                    className="max-h-full max-w-full"
                                />
                            ) : (
                                <div className="text-center">
                                    <VideoCamera size={40} className="text-slate-600 mx-auto mb-2" />
                                    <p className="text-slate-600">Upload and process a video</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {(accidentTimeline.length > 0 || accidentPeaks.length > 0 || accidentClip) && (
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Accident Timeline
                            </h3>
                            {timelineBuckets.length > 0 ? (
                                <div className="space-y-3">
                                    <div className="h-20 flex items-end gap-[2px] bg-slate-900/50 rounded-sm p-2">
                                        {timelineBuckets.map((bucket, index) => {
                                            const height = Math.max(4, Math.round(bucket.score * 100));
                                            const isHot = bucket.score >= accidentThreshold;
                                            return (
                                                <button
                                                    key={`${index}-${bucket.timestamp}`}
                                                    type="button"
                                                    onClick={() => jumpToTimestamp(bucket.timestamp)}
                                                    title={`${formatTime(bucket.timestamp)} • ${(bucket.score * 100).toFixed(1)}%`}
                                                    className={`w-full rounded-sm transition-opacity ${isHot ? 'bg-red-500' : 'bg-slate-600'} hover:opacity-90`}
                                                    style={{ height: `${height}%` }}
                                                />
                                            );
                                        })}
                                    </div>
                                    <div className="text-xs text-slate-500 flex items-center justify-between">
                                        <span>Click bars to jump</span>
                                        <span>Threshold {Math.round(accidentThreshold * 100)}%</span>
                                    </div>
                                </div>
                            ) : (
                                <p className="text-sm text-slate-500">No timeline data yet.</p>
                            )}

                            {accidentPeaks.length > 0 && (
                                <div className="mt-4 space-y-2">
                                    {accidentPeaks.map((peak, index) => (
                                        <div key={`${peak.timestamp}-${index}`} className="flex items-center justify-between text-sm">
                                            <div className="text-slate-300">
                                                Peak {index + 1} • {formatTime(peak.timestamp)} • {(peak.score * 100).toFixed(1)}%
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => jumpToTimestamp(peak.timestamp)}
                                                className="btn-secondary px-3 py-1 text-xs"
                                            >
                                                Jump
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {accidentClip && (
                                <div className="mt-4">
                                    <div className="flex items-center justify-between text-sm text-slate-300">
                                        <span>
                                            Accident Clip ({formatTime(accidentClip.start)} - {formatTime(accidentClip.end)})
                                        </span>
                                        <a
                                            href={accidentClip.url}
                                            download={`accident_clip_${Math.round(accidentClip.best_timestamp)}s.mp4`}
                                            className="text-blue-400 hover:text-blue-300 text-xs"
                                        >
                                            Download Clip
                                        </a>
                                    </div>
                                    <video
                                        src={accidentClip.url}
                                        controls
                                        className="mt-2 w-full max-h-64"
                                    />
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
