import React from 'react';

interface DataCardProps {
    title: string;
    value: string | number;
    icon?: React.ElementType;
    className?: string;
    color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple';
}

const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    purple: 'text-purple-400',
};

export function DataCard({
    title,
    value,
    icon: Icon,
    className = '',
    color = 'blue',
}: DataCardProps) {
    return (
        <div className={`bg-slate-800/40 border border-slate-700/60 rounded-sm p-6 transition-all duration-200 hover:border-slate-500 ${className}`}>
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-slate-500 text-xs uppercase tracking-wider font-mono mb-2">{title}</p>
                    <p className="text-3xl font-bold font-mono text-slate-100">{value}</p>
                </div>
                {Icon && (
                    <div className={colorClasses[color]}>
                        <Icon size={28} weight="duotone" />
                    </div>
                )}
            </div>
        </div>
    );
}

export default DataCard;
