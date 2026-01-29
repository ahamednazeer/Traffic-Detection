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
            setResultImage(`data:image/jpeg;base64,${result.annotated_image}`);
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

                    {/* Stats */}
                    <DetectionStatsPanel stats={stats} processingTime={processingTime} />
                </div>

                {/* Right Column - Images */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Original */}
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Original Image
                            </h3>
                            <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center">
                                {previewUrl ? (
                                    <img src={previewUrl} alt="Original" className="max-h-full max-w-full object-contain" />
                                ) : (
                                    <p className="text-slate-600">No image selected</p>
                                )}
                            </div>
                        </div>

                        {/* Result */}
                        <div className="card">
                            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                                Detection Result
                            </h3>
                            <div className="aspect-video bg-slate-900 rounded-sm overflow-hidden flex items-center justify-center">
                                {loading ? (
                                    <div className="text-blue-400 animate-pulse">Processing...</div>
                                ) : resultImage ? (
                                    <img src={resultImage} alt="Result" className="max-h-full max-w-full object-contain" />
                                ) : (
                                    <p className="text-slate-600">Run detection to see results</p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
