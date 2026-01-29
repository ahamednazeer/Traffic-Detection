'use client';

import React, { useState, useRef } from 'react';
import ModelSelector from '@/components/ModelSelector';
import DetectionStatsPanel from '@/components/DetectionStats';
import type { DetectionStats } from '@/lib/api';
import {
    Upload,
    Play,
    ArrowsClockwise,
    VideoCamera,
} from '@phosphor-icons/react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const CHUNK_SIZE = 1024 * 1024; // 1MB chunks

export default function VideoDetectionPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [resultVideo, setResultVideo] = useState<string | null>(null);
    const [stats, setStats] = useState<DetectionStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [confidence, setConfidence] = useState(0.5);
    const [skipFrames, setSkipFrames] = useState(2);
    const [progress, setProgress] = useState(0);
    const [progressStatus, setProgressStatus] = useState<string | null>(null);
    const [videoInfo, setVideoInfo] = useState<any>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResultVideo(null);
            setStats(null);
            setError(null);
            setVideoInfo(null);
            setProgress(0);
        }
    };

    const handleProcess = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);
        setProgress(0);
        setProgressStatus('Connecting...');

        try {
            const ws = new WebSocket(`${WS_URL}/api/video/process`);
            wsRef.current = ws;

            ws.onopen = async () => {
                setProgressStatus('Sending video...');

                // Send metadata first
                ws.send(JSON.stringify({
                    size: selectedFile.size,
                    confidence,
                    skip_frames: skipFrames
                }));
            };

            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);

                if (data.error) {
                    setError(data.error);
                    setLoading(false);
                    ws.close();
                    return;
                }

                if (data.type === 'ready') {
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
                    setProgress(data.progress);
                    setProgressStatus(`Processing: ${data.frame}/${data.total} frames`);
                }

                if (data.type === 'complete') {
                    setResultVideo(`data:video/mp4;base64,${data.video_base64}`);
                    setStats(data.statistics);
                    setVideoInfo(data.video_info);
                    setProgress(100);
                    setProgressStatus('Complete!');
                    setLoading(false);
                    ws.close();
                }
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('Connection failed. Is the backend running?');
                setLoading(false);
            };

            ws.onclose = () => {
                wsRef.current = null;
            };

        } catch (err: any) {
            setError(err.message || 'Video processing failed');
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6 animate-slide-up">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-chivo font-bold text-2xl uppercase tracking-wider">
                        Video Detection
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Process video files for object detection
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Controls */}
                <div className="space-y-4">
                    <ModelSelector />

                    {/* Confidence Slider */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Confidence Threshold
                        </h3>
                        <input
                            type="range"
                            min="0.1"
                            max="1"
                            step="0.05"
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
                    <button
                        onClick={handleProcess}
                        disabled={!selectedFile || loading}
                        className="btn-primary w-full flex items-center justify-center gap-2"
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

                    {/* Stats */}
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
                                <div className="text-center">
                                    <ArrowsClockwise size={40} className="text-blue-400 animate-spin mx-auto mb-2" />
                                    <p className="text-blue-400">{progressStatus}</p>
                                    <p className="text-slate-500 text-sm mt-1">{progress}% complete</p>
                                </div>
                            ) : resultVideo ? (
                                <video
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
                </div>
            </div>
        </div>
    );
}
