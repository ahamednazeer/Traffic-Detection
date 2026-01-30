'use client';

import React, { useState, useRef } from 'react';
import api, { DetectionStats, Detection } from '@/lib/api';
import ModelSelector from '@/components/ModelSelector';
import DetectionStatsPanel from '@/components/DetectionStats';
import {
    Upload,
    MagnifyingGlass,
    ArrowsClockwise,
} from '@phosphor-icons/react';

export default function ImageDetectionPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [resultImage, setResultImage] = useState<string | null>(null);
    const [stats, setStats] = useState<DetectionStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [confidence, setConfidence] = useState(0.5);
    const [processingTime, setProcessingTime] = useState<number | undefined>();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResultImage(null);
            setStats(null);
            setError(null);
        }
    };

    const handleDetect = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);
        const startTime = performance.now();

        try {
            const result = await api.detectImage(selectedFile, confidence);
            // Handle both formats: raw base64 or full data URL
            const imageData = result.annotated_image.startsWith('data:')
                ? result.annotated_image
                : `data:image/jpeg;base64,${result.annotated_image}`;
            setResultImage(imageData);
            setStats(result.statistics);
            setProcessingTime((performance.now() - startTime) / 1000);
        } catch (err: any) {
            setError(err.message || 'Detection failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6 animate-slide-up">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-chivo font-bold text-2xl uppercase tracking-wider">
                        Image Detection
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Upload an image to detect vehicles and pedestrians
                    </p>
                </div>
            </div>

            {/* Main Result Display - Full Width */}
            {(resultImage || loading) && (
                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider">
                            Detection Result
                        </h3>
                        {resultImage && (
                            <a
                                href={resultImage}
                                download="detection_result.jpg"
                                className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                            >
                                Download Image
                            </a>
                        )}
                    </div>
                    <div className="bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center min-h-[400px] max-h-[600px]">
                        {loading ? (
                            <div className="flex flex-col items-center gap-3">
                                <ArrowsClockwise size={32} className="text-blue-400 animate-spin" />
                                <span className="text-blue-400">Processing image...</span>
                            </div>
                        ) : resultImage ? (
                            <img
                                src={resultImage}
                                alt="Detection Result"
                                className="max-h-[600px] w-auto object-contain cursor-zoom-in"
                                onClick={() => window.open(resultImage, '_blank')}
                                title="Click to view full size"
                            />
                        ) : null}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
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
                        />
                        <div className="text-center text-blue-400 font-mono text-lg mt-2">
                            {(confidence * 100).toFixed(0)}%
                        </div>
                    </div>

                    {/* Upload */}
                    <div className="card">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Upload Image
                        </h3>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="btn-secondary w-full flex items-center justify-center gap-2"
                        >
                            <Upload size={18} weight="duotone" />
                            {selectedFile ? selectedFile.name : 'Choose File'}
                        </button>
                    </div>

                    {/* Detect Button */}
                    <button
                        onClick={handleDetect}
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
                                <MagnifyingGlass size={18} weight="duotone" />
                                Detect Objects
                            </>
                        )}
                    </button>

                    {error && (
                        <div className="bg-red-950/30 border border-red-800 rounded-sm p-3 text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>

                {/* Middle - Original Image */}
                <div className="lg:col-span-2">
                    <div className="card h-full">
                        <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                            Original Image
                        </h3>
                        <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center">
                            {previewUrl ? (
                                <img src={previewUrl} alt="Original" className="max-h-full max-w-full object-contain" />
                            ) : (
                                <div className="flex flex-col items-center gap-2 text-slate-600">
                                    <Upload size={32} weight="duotone" />
                                    <p>No image selected</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right - Stats */}
                <div>
                    <DetectionStatsPanel stats={stats} processingTime={processingTime} />
                </div>
            </div>
        </div>
    );
}
