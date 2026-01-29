'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import ModelSelector from '@/components/ModelSelector';
import DetectionStatsPanel from '@/components/DetectionStats';
import type { DetectionStats } from '@/lib/api';
import {
    Camera,
    Play,
    Stop,
    Warning,
} from '@phosphor-icons/react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export default function CameraDetectionPage() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [stats, setStats] = useState<DetectionStats | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [fps, setFps] = useState(0);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const resultCanvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const frameCountRef = useRef(0);
    const lastTimeRef = useRef(Date.now());

    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' }
            });

            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            return true;
        } catch (err: any) {
            setError(`Camera access denied: ${err.message}`);
            return false;
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    }, []);

    const connectWebSocket = useCallback(() => {
        const ws = new WebSocket(`${WS_URL}/api/camera`);

        ws.onopen = () => {
            console.log('WebSocket connected');
            setError(null);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.error('Detection error:', data.error);
                    return;
                }

                // Update result canvas with annotated frame
                if (data.frame && resultCanvasRef.current) {
                    const ctx = resultCanvasRef.current.getContext('2d');
                    const img = new Image();
                    img.onload = () => {
                        if (ctx && resultCanvasRef.current) {
                            resultCanvasRef.current.width = img.width;
                            resultCanvasRef.current.height = img.height;
                            ctx.drawImage(img, 0, 0);
                        }
                    };
                    img.src = `data:image/jpeg;base64,${data.frame}`;
                }

                // Update stats
                if (data.stats) {
                    setStats(data.stats);
                }

                // Calculate FPS
                frameCountRef.current++;
                const now = Date.now();
                if (now - lastTimeRef.current >= 1000) {
                    setFps(frameCountRef.current);
                    frameCountRef.current = 0;
                    lastTimeRef.current = now;
                }

            } catch (err) {
                console.error('Parse error:', err);
            }
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setError('WebSocket connection failed');
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            if (isStreaming) {
                // Reconnect after delay
                setTimeout(() => {
                    if (isStreaming) {
                        wsRef.current = connectWebSocket();
                    }
                }, 1000);
            }
        };

        return ws;
    }, [isStreaming]);

    const sendFrame = useCallback(() => {
        if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
        if (wsRef.current.readyState !== WebSocket.OPEN) return;

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        // Draw video frame to canvas
        canvasRef.current.width = videoRef.current.videoWidth || 640;
        canvasRef.current.height = videoRef.current.videoHeight || 480;
        ctx.drawImage(videoRef.current, 0, 0);

        // Convert to base64 and send
        canvasRef.current.toBlob((blob) => {
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64 = (reader.result as string).split(',')[1];
                    wsRef.current?.send(base64);
                };
                reader.readAsDataURL(blob);
            }
        }, 'image/jpeg', 0.8);
    }, []);

    const startStreaming = useCallback(async () => {
        setError(null);

        // Start camera
        const cameraStarted = await startCamera();
        if (!cameraStarted) return;

        // Connect WebSocket
        wsRef.current = connectWebSocket();

        // Start sending frames
        setIsStreaming(true);
    }, [startCamera, connectWebSocket]);

    const stopStreaming = useCallback(() => {
        setIsStreaming(false);

        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        // Stop camera
        stopCamera();

        // Reset stats
        setStats(null);
        setFps(0);
    }, [stopCamera]);

    // Send frames while streaming
    useEffect(() => {
        if (!isStreaming) return;

        const interval = setInterval(sendFrame, 100); // ~10 FPS

        return () => clearInterval(interval);
    }, [isStreaming, sendFrame]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopStreaming();
        };
    }, [stopStreaming]);

    return (
        <div className="space-y-6 animate-slide-up">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-chivo font-bold text-2xl uppercase tracking-wider">
                        Live Camera Detection
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Real-time detection from your webcam
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    {isStreaming && (
                        <div className="flex items-center gap-2 text-green-400">
                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            <span className="font-mono text-sm">{fps} FPS</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Controls */}
                <div className="space-y-4">
                    <ModelSelector />

                    {/* Start/Stop Button */}
                    {!isStreaming ? (
                        <button
                            onClick={startStreaming}
                            className="btn-success w-full flex items-center justify-center gap-2"
                        >
                            <Play size={20} weight="duotone" />
                            Start Camera
                        </button>
                    ) : (
                        <button
                            onClick={stopStreaming}
                            className="btn-danger w-full flex items-center justify-center gap-2"
                        >
                            <Stop size={20} weight="duotone" />
                            Stop Camera
                        </button>
                    )}

                    {error && (
                        <div className="bg-red-950/30 border border-red-800 rounded-sm p-3 flex items-center gap-2 text-red-400 text-sm">
                            <Warning size={18} />
                            {error}
                        </div>
                    )}

                    {/* Stats */}
                    <DetectionStatsPanel stats={stats} />
                </div>

                {/* Right Column - Camera Feeds */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Camera Input */}
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Camera Feed
                            </h3>
                            <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center relative">
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className={`max-h-full max-w-full ${isStreaming ? '' : 'hidden'}`}
                                />
                                {!isStreaming && (
                                    <div className="text-center">
                                        <Camera size={40} className="text-slate-600 mx-auto mb-2" />
                                        <p className="text-slate-600">Camera not started</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Detection Result */}
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Detection Result
                            </h3>
                            <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center">
                                <canvas
                                    ref={resultCanvasRef}
                                    className={`max-h-full max-w-full ${isStreaming ? '' : 'hidden'}`}
                                />
                                {!isStreaming && (
                                    <div className="text-center">
                                        <Camera size={40} className="text-slate-600 mx-auto mb-2" />
                                        <p className="text-slate-600">Start camera to see detections</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} className="hidden" />
                </div>
            </div>
        </div>
    );
}
