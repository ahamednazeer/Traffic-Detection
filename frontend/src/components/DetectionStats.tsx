'use client';

import React from 'react';
import type { DetectionStats } from '@/lib/api';
import {
    Warning,
    CheckCircle,
} from '@phosphor-icons/react';

interface DetectionStatsProps {
    stats: DetectionStats | null;
    processingTime?: number;
}

const CLASS_COLORS: Record<string, string> = {
    Car: '#3b82f6',
    Pedestrian: '#22c55e',
    Van: '#6366f1',
    Cyclist: '#eab308',
    Truck: '#ec4899',
    Misc: '#06b6d4',
    Tram: '#a855f7',
    Person_sitting: '#f97316',
};

export default function DetectionStatsPanel({ stats, processingTime }: DetectionStatsProps) {
    if (!stats || stats.total_objects === 0) {
        return (
            <div className="card text-center py-8">
                <p className="text-slate-500">No detections yet</p>
            </div>
        );
    }

    return (
        <div className="card space-y-4">
            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider">
                Detection Analytics
            </h3>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-900/50 rounded-sm p-3 text-center">
                    <div className="text-2xl font-bold text-blue-400">{stats.total_objects}</div>
                    <div className="text-xs text-slate-500">Total Objects</div>
                </div>
                <div className="bg-slate-900/50 rounded-sm p-3 text-center">
                    <div className="text-2xl font-bold text-green-400">{stats.unique_classes}</div>
                    <div className="text-xs text-slate-500">Unique Classes</div>
                </div>
                <div className="bg-slate-900/50 rounded-sm p-3 text-center">
                    <div className="text-2xl font-bold text-yellow-400">
                        {(stats.avg_confidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-slate-500">Avg Confidence</div>
                </div>
                {processingTime !== undefined && (
                    <div className="bg-slate-900/50 rounded-sm p-3 text-center">
                        <div className="text-2xl font-bold text-purple-400">{processingTime.toFixed(2)}s</div>
                        <div className="text-xs text-slate-500">Process Time</div>
                    </div>
                )}
            </div>

            {/* Class Breakdown */}
            <div className="space-y-2">
                <h4 className="text-xs font-mono text-slate-500 uppercase">Class Breakdown</h4>
                {Object.entries(stats.class_counts).map(([className, count]) => {
                    const percentage = (count / stats.total_objects) * 100;
                    const color = CLASS_COLORS[className] || '#64748b';

                    return (
                        <div key={className} className="space-y-1">
                            <div className="flex justify-between text-sm">
                                <span className="flex items-center gap-2">
                                    <span
                                        className="w-3 h-3 rounded-full"
                                        style={{ backgroundColor: color }}
                                    />
                                    {className}
                                </span>
                                <span className="text-slate-400">{count}</span>
                            </div>
                            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full transition-all"
                                    style={{ width: `${percentage}%`, backgroundColor: color }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Safety Alert */}
            {stats.has_pedestrians && stats.has_vehicles && (
                <div className="bg-yellow-950/30 border border-yellow-800/50 rounded-sm p-3">
                    <div className="flex items-center gap-2 text-yellow-400 text-sm">
                        <Warning size={18} weight="duotone" />
                        <span className="font-medium">Mixed Traffic Alert</span>
                    </div>
                    <p className="text-xs text-yellow-400/70 mt-1">
                        Pedestrians and vehicles detected in the same frame
                    </p>
                </div>
            )}
        </div>
    );
}
