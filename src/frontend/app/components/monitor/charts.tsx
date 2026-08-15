"use client";

import React from "react";

/** 概览统计卡片（支持暗色）。 */
export function StatCard({
  title,
  value,
  unit,
  hint,
  accent = "text-blue-500",
  dark = false,
}: {
  title: string;
  value: string | number;
  unit?: string;
  hint?: string;
  accent?: string;
  dark?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-4 shadow-sm ${
        dark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
      }`}
    >
      <div className={`text-xs font-medium ${dark ? "text-gray-400" : "text-gray-500"}`}>{title}</div>
      <div className={`mt-1 text-2xl font-semibold ${accent}`}>
        {value}
        {unit ? <span className="ml-1 text-sm font-normal text-gray-400">{unit}</span> : null}
      </div>
      {hint ? (
        <div className={`mt-1 text-xs ${dark ? "text-gray-500" : "text-gray-400"}`}>{hint}</div>
      ) : null}
    </div>
  );
}

/** 单条水平进度条（用于耗时分布）。 */
export function BarRow({
  label,
  value,
  max,
  unit,
  color = "#3b82f6",
  dark = false,
}: {
  label: string;
  value: number;
  max: number;
  unit?: string;
  color?: string;
  dark?: boolean;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className={`w-14 shrink-0 text-right ${dark ? "text-gray-400" : "text-gray-500"}`}>{label}</div>
      <div className={`relative h-3 flex-1 rounded ${dark ? "bg-gray-700" : "bg-gray-100"}`}>
        <div className="h-3 rounded" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <div className={`w-20 shrink-0 text-right tabular-nums ${dark ? "text-gray-200" : "text-gray-700"}`}>
        {Number(value).toFixed(2)}
        {unit ? ` ${unit}` : ""}
      </div>
    </div>
  );
}

/** 轻量 SVG 折线/面积图（实时趋势 sparkline）。 */
export function LineChart({
  data,
  color = "#10b981",
  height = 80,
  unit,
}: {
  data: number[];
  color?: string;
  height?: number;
  unit?: string;
}) {
  if (!data || data.length === 0) return <div className="text-xs text-gray-400">暂无数据</div>;
  const w = 260;
  const h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = data.length > 1 ? w / (data.length - 1) : 0;
  const pts = data.map(
    (v, i) => [i * stepX, h - ((v - min) / range) * (h - 8) - 4] as [number, number]
  );
  const poly = pts.map((p) => p.join(",")).join(" ");
  const area = `0,${h} ${poly} ${w},${h}`;
  const last = data[data.length - 1];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height }}>
      <polygon points={area} fill={color} opacity={0.15} />
      <polyline points={poly} fill="none" stroke={color} strokeWidth={2} />
      <text x={w - 4} y={14} textAnchor="end" fontSize={11} fill="#9ca3af">
        {Number(last).toFixed(1)}
        {unit ? ` ${unit}` : ""}
      </text>
    </svg>
  );
}

/** 健康状态徽标。 */
export function HealthBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    up: "bg-green-100 text-green-700",
    healthy: "bg-green-100 text-green-700",
    degraded: "bg-yellow-100 text-yellow-700",
    down: "bg-red-100 text-red-700",
    unhealthy: "bg-red-100 text-red-700",
  };
  const cls = map[status] || "bg-gray-100 text-gray-600";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>;
}
