'use client';

import React from 'react';
import type { AccidentResult } from '@/lib/api';
import { Warning, CheckCircle } from '@phosphor-icons/react';

interface AccidentStatusProps {
    accident?: AccidentResult | null;
}

export default function AccidentStatus({ accident }: AccidentStatusProps) {
    if (!accident) {
        return (
            <div className="card text-center py-8">
                <p className="text-slate-500">No accident analysis yet</p>
            </div>
        );
    }

    const detected = accident.detected;
    const score = (accident.score * 100).toFixed(1);
    const threshold = (accident.threshold * 100).toFixed(1);
    const timestamp = accident.best_timestamp !== null && accident.best_timestamp !== undefined
        ? `${accident.best_timestamp.toFixed(2)}s`
        : null;

    return (
        <div className={`card ${detected ? 'border-red-800 bg-red-950/30' : 'border-green-800 bg-green-950/30'}`}>
            <div className="flex items-center gap-2">
                {detected ? (
                    <Warning size={20} weight="duotone" className="text-red-400" />
                ) : (
                    <CheckCircle size={20} weight="duotone" className="text-green-400" />
                )}
                <h3 className={`text-sm font-mono uppercase tracking-wider ${detected ? 'text-red-400' : 'text-green-400'}`}>
                    {detected ? 'Accident Detected' : 'No Accident Detected'}
                </h3>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-900/40 rounded-sm p-2">
                    <div className="text-slate-500">Score</div>
                    <div className={detected ? 'text-red-300' : 'text-green-300'}>{score}%</div>
                </div>
                <div className="bg-slate-900/40 rounded-sm p-2">
                    <div className="text-slate-500">Threshold</div>
                    <div className="text-slate-200">{threshold}%</div>
                </div>
                {timestamp && (
                    <div className="bg-slate-900/40 rounded-sm p-2">
                        <div className="text-slate-500">Best Time</div>
                        <div className="text-slate-200">{timestamp}</div>
                    </div>
                )}
            </div>
        </div>
    );
}
