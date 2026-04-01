"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { DriftScore } from "@/lib/api";

interface ChartPoint {
  filing_date: string;
  cosine_distance: number;
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-[#1a1a1a] bg-[#111111] px-3 py-2 shadow-sm">
      <p className="text-xs text-[#737373] mb-1">{label}</p>
      <p className="text-sm font-mono text-[#c8a97e]">
        {Number(payload[0].value).toFixed(4)}
      </p>
    </div>
  );
}

export default function DriftChart({ data }: { data: DriftScore[] }) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-sm text-[#737373]">
        No timeline data available.
      </div>
    );
  }

  const chartData: ChartPoint[] = [...data]
    .sort(
      (a, b) =>
        new Date(a.filing_date).getTime() - new Date(b.filing_date).getTime()
    )
    .map((d) => ({
      filing_date: d.filing_date,
      cosine_distance: d.cosine_distance,
    }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart
        data={chartData}
        margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#1a1a1a"
          vertical={false}
        />
        <XAxis
          dataKey="filing_date"
          tick={{ fontSize: 11, fill: "#737373" }}
          axisLine={{ stroke: "#1a1a1a" }}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#737373" }}
          axisLine={{ stroke: "#1a1a1a" }}
          tickLine={false}
          width={50}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="cosine_distance"
          stroke="#c8a97e"
          strokeWidth={2}
          dot={{ r: 3, fill: "#c8a97e", stroke: "#0a0a0a", strokeWidth: 2 }}
          activeDot={{ r: 5, fill: "#c8a97e", stroke: "#0a0a0a", strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
