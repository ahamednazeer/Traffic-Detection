'use client';

import React, { useState, useEffect, useRef } from 'react';
import api from '@/lib/api';
import {
    Lightning,
    Target,
    ArrowsClockwise,
    Trophy,
    CaretDown,
    Check,
} from '@phosphor-icons/react';

interface ModelSelectorProps {
    onModelChange?: (model: string) => void;
}

const YOLO11_SIZES = [
    { id: 'yolo11n', name: 'Nano', size: '~5MB', desc: 'Fastest' },
    { id: 'yolo11s', name: 'Small', size: '~18MB', desc: 'Fast' },
    { id: 'yolo11m', name: 'Medium', size: '~40MB', desc: 'Balanced' },
    { id: 'yolo11l', name: 'Large', size: '~75MB', desc: 'Accurate' },
    { id: 'yolo11x', name: 'XLarge', size: '~140MB', desc: 'Best accuracy' },
];

export default function ModelSelector({ onModelChange }: ModelSelectorProps) {
    const [activeModel, setActiveModel] = useState('yolo11x');
    const [loading, setLoading] = useState(false);
    const [showYoloDropdown, setShowYoloDropdown] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        api.getModels().then(data => {
            setActiveModel(data.active);
        }).catch(console.error);
    }, []);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowYoloDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleModelChange = async (model: string) => {
        setLoading(true);
        setShowYoloDropdown(false);
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

    const isYolo11Active = activeModel.startsWith('yolo11');
    const activeYolo11 = YOLO11_SIZES.find(s => s.id === activeModel);

    return (
        <div className="card">
            <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-3">
                Detection Model
            </h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                {/* Custom YOLO */}
                <button
                    onClick={() => handleModelChange('yolo')}
                    disabled={loading}
                    className={`p-3 rounded-sm border transition-all text-center ${activeModel === 'yolo'
                        ? 'border-blue-500 bg-blue-950/50 text-blue-400'
                        : 'border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200'
                        } ${loading ? 'opacity-50' : ''}`}
                >
                    <Lightning size={22} weight="duotone" className="mx-auto mb-1" />
                    <div className="text-xs font-medium">Custom</div>
                </button>

                {/* YOLO11 with Dropdown */}
                <div className="relative" ref={dropdownRef}>
                    <button
                        onClick={() => setShowYoloDropdown(!showYoloDropdown)}
                        disabled={loading}
                        className={`w-full p-3 rounded-sm border transition-all text-center ${isYolo11Active
                            ? 'border-blue-500 bg-blue-950/50 text-blue-400'
                            : 'border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200'
                            } ${loading ? 'opacity-50' : ''}`}
                    >
                        <Trophy size={22} weight="duotone" className="mx-auto mb-1" />
                        <div className="text-xs font-medium">
                            YOLO11
                        </div>
                        <div className="text-[10px] text-slate-400 flex items-center justify-center gap-1 mt-0.5">
                            {activeYolo11 ? activeYolo11.name : 'Select'}
                            <CaretDown size={10} className={`transition-transform ${showYoloDropdown ? 'rotate-180' : ''}`} />
                        </div>
                    </button>

                    {/* Dropdown */}
                    {showYoloDropdown && (
                        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 min-w-[200px] bg-slate-800 border border-slate-600 rounded-sm shadow-xl z-[100] overflow-hidden">
                            {YOLO11_SIZES.map((size) => (
                                <button
                                    key={size.id}
                                    onClick={() => handleModelChange(size.id)}
                                    className={`w-full px-3 py-2 text-left text-sm flex items-center justify-between hover:bg-slate-700 transition-colors ${activeModel === size.id ? 'bg-blue-950/50 text-blue-400' : 'text-slate-300'
                                        }`}
                                >
                                    <div>
                                        <span className="font-medium">{size.name}</span>
                                        <span className="text-xs text-slate-500 ml-2">{size.size}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-slate-500">{size.desc}</span>
                                        {activeModel === size.id && <Check size={14} weight="bold" />}
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* SSD */}
                <button
                    onClick={() => handleModelChange('ssd')}
                    disabled={loading}
                    className={`p-3 rounded-sm border transition-all text-center ${activeModel === 'ssd'
                        ? 'border-blue-500 bg-blue-950/50 text-blue-400'
                        : 'border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200'
                        } ${loading ? 'opacity-50' : ''}`}
                >
                    <Target size={22} weight="duotone" className="mx-auto mb-1" />
                    <div className="text-xs font-medium">SSD300</div>
                </button>

                {/* Ensemble */}
                <button
                    onClick={() => handleModelChange('ensemble')}
                    disabled={loading}
                    className={`p-3 rounded-sm border transition-all text-center ${activeModel === 'ensemble'
                        ? 'border-blue-500 bg-blue-950/50 text-blue-400'
                        : 'border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200'
                        } ${loading ? 'opacity-50' : ''}`}
                >
                    <ArrowsClockwise size={22} weight="duotone" className="mx-auto mb-1" />
                    <div className="text-xs font-medium">Ensemble</div>
                </button>
            </div>
        </div>
    );
}
