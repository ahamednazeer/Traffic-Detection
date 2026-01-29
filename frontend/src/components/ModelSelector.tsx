'use client';

import React, { useState, useEffect } from 'react';
import api from '@/lib/api';
import {
    Lightning,
    Target,
    ArrowsClockwise,
} from '@phosphor-icons/react';

interface ModelSelectorProps {
    onModelChange?: (model: string) => void;
}

export default function ModelSelector({ onModelChange }: ModelSelectorProps) {
    const [activeModel, setActiveModel] = useState('yolo');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        api.getModels().then(data => {
            setActiveModel(data.active);
        }).catch(console.error);
    }, []);

    const handleModelChange = async (model: string) => {
        setLoading(true);
        try {
            await api.selectModel(model);
            setActiveModel(model);
            onModelChange?.(model);
        } catch (error) {
            console.error('Failed to change model:', error);
        } finally {
            setLoading(false);
        }
    };

    const models = [
        { id: 'yolo', name: 'YOLO v11', icon: Lightning, description: 'Fast & accurate' },
        { id: 'ssd', name: 'SSD300', icon: Target, description: 'COCO pre-trained' },
        { id: 'ensemble', name: 'Ensemble', icon: ArrowsClockwise, description: 'Both models' },
    ];

    return (
        <div className="card">
            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                Detection Model
            </h3>
            <div className="grid grid-cols-3 gap-2">
                {models.map((model) => {
                    const Icon = model.icon;
                    return (
                        <button
                            key={model.id}
                            onClick={() => handleModelChange(model.id)}
                            disabled={loading}
                            className={`p-3 rounded-sm border transition-all text-center ${activeModel === model.id
                                    ? 'border-blue-500 bg-blue-950/50 text-blue-400'
                                    : 'border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200'
                                } ${loading ? 'opacity-50' : ''}`}
                        >
                            <Icon size={24} weight="duotone" className="mx-auto mb-1" />
                            <div className="text-xs font-medium">{model.name}</div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
