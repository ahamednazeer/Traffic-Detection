'use client';

import { useRouter } from 'next/navigation';
import {
  Car,
  Lightning,
  Target,
  ArrowsClockwise,
  ArrowRight,
} from '@phosphor-icons/react';

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center">
      <div className="scanlines" />

      <div className="text-center space-y-8 animate-slide-up relative z-10">
        {/* Logo */}
        <Car size={96} weight="duotone" className="text-blue-400 mx-auto" />

        {/* Title */}
        <h1 className="font-chivo font-bold text-5xl uppercase tracking-wider text-gradient">
          Traffic Detection
        </h1>
        <p className="text-slate-400 text-lg max-w-md mx-auto">
          Real-time traffic sign and pedestrian detection using YOLO v11 and SSD
        </p>

        {/* Features */}
        <div className="flex justify-center gap-8 text-sm text-slate-500">
          <div className="flex items-center gap-2">
            <Lightning size={18} weight="duotone" className="text-blue-400" />
            YOLO v11
          </div>
          <div className="flex items-center gap-2">
            <Target size={18} weight="duotone" className="text-green-400" />
            SSD300
          </div>
          <div className="flex items-center gap-2">
            <ArrowsClockwise size={18} weight="duotone" className="text-purple-400" />
            Ensemble
          </div>
        </div>

        {/* CTA Button */}
        <button
          onClick={() => router.push('/dashboard')}
          className="btn-primary text-lg px-8 py-3 inline-flex items-center gap-2"
        >
          Launch Dashboard
          <ArrowRight size={20} weight="bold" />
        </button>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-8 mt-12 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-400">8</div>
            <div className="text-xs text-slate-500 uppercase tracking-wider">Classes</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-green-400">2</div>
            <div className="text-xs text-slate-500 uppercase tracking-wider">Models</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-purple-400">30+</div>
            <div className="text-xs text-slate-500 uppercase tracking-wider">FPS</div>
          </div>
        </div>
      </div>
    </div>
  );
}