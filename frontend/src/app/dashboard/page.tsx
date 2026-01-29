'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import DataCard from '@/components/DataCard';
import api from '@/lib/api';
import {
    Image,
    VideoCamera,
    Camera,
    Robot,
    Tag,
    Package,
    Pulse,
    CheckCircle,
    Warning,
} from '@phosphor-icons/react';

export default function DashboardOverview() {
    const router = useRouter();
    const [health, setHealth] = useState<{ status: string; active_model: string } | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        api.health()
            .then(setHealth)
            .catch(() => setError('Backend not connected'));
    }, []);

    const features = [
        {
            title: 'Image Detection',
            description: 'Upload images for object detection',
            icon: Image,
            path: '/dashboard/image',
        },
        {
            title: 'Video Detection',
            description: 'Process video files frame by frame',
            icon: VideoCamera,
            path: '/dashboard/video',
        },
        {
            title: 'Live Camera',
            description: 'Real-time detection from webcam',
            icon: Camera,
            path: '/dashboard/camera',
        },
    ];

    return (
        <div className="space-y-6 animate-slide-up">
            {/* Status Banner */}
            {error ? (
                <div className="bg-red-950/30 border border-red-800 rounded-sm p-4 flex items-center gap-3">
                    <Warning size={24} weight="duotone" className="text-red-400" />
                    <div>
                        <p className="text-red-400 font-medium">{error}</p>
                        <p className="text-red-400/70 text-sm">
                            Start the backend with: cd backend && python run.py
                        </p>
                    </div>
                </div>
            ) : health ? (
                <div className="bg-green-950/30 border border-green-800 rounded-sm p-4 flex items-center gap-3">
                    <CheckCircle size={24} weight="duotone" className="text-green-400" />
                    <div>
                        <p className="text-green-400 font-medium">Backend Connected</p>
                        <p className="text-green-400/70 text-sm">
                            Active model: {health.active_model.toUpperCase()}
                        </p>
                    </div>
                </div>
            ) : (
                <div className="bg-slate-800/40 border border-slate-700 rounded-sm p-4 animate-pulse">
                    <p className="text-slate-400">Connecting to backend...</p>
                </div>
            )}

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <DataCard title="Detection Models" value="2" icon={Robot} color="blue" />
                <DataCard title="YOLO Classes" value="8" icon={Tag} color="green" />
                <DataCard title="SSD Classes" value="COCO" icon={Package} color="purple" />
                <DataCard title="Status" value={health ? 'Online' : 'Offline'} icon={Pulse} color={health ? 'green' : 'red'} />
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {features.map((feature) => {
                    const Icon = feature.icon;
                    return (
                        <button
                            key={feature.path}
                            onClick={() => router.push(feature.path)}
                            className="card card-hover text-left group"
                        >
                            <Icon size={40} weight="duotone" className="text-blue-400 mb-4 group-hover:scale-110 transition-transform" />
                            <h3 className="font-chivo font-bold text-lg uppercase tracking-wide mb-2">
                                {feature.title}
                            </h3>
                            <p className="text-slate-400 text-sm">{feature.description}</p>
                        </button>
                    );
                })}
            </div>

            {/* Quick Info */}
            <div className="card">
                <h3 className="font-chivo font-bold text-sm uppercase tracking-wider mb-4">
                    Supported Detection Classes
                </h3>
                <div className="flex flex-wrap gap-2">
                    {['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Person_sitting'].map((cls) => (
                        <span
                            key={cls}
                            className="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-sm text-slate-300"
                        >
                            {cls}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}
