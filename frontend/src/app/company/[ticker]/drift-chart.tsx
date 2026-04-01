"use client";

import { useTheme } from "next-themes";
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
    <div className="rounded-md border border-border bg-card px-3 py-2 shadow-sm">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className="text-sm font-mono text-accent">
        {Number(payload[0].value).toFixed(4)}
      </p>
    </div>
  );
}

export default function DriftChart({ data }: { data: DriftScore[] }) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  // Use CSS variable-compatible colors for chart elements
  const gridColor = isDark ? "#1a1a1a" : "#e8e5e0";
  const tickColor = isDark ? "#737373" : "#6b6b6b";
  const dotStrokeColor = isDark ? "#0a0a0a" : "#fafaf8";

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-sm text-muted-foreground">
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
          stroke={gridColor}
          vertical={false}
        />
        <XAxis
          dataKey="filing_date"
          tick={{ fontSize: 11, fill: tickColor }}
          axisLine={{ stroke: gridColor }}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 11, fill: tickColor }}
          axisLine={{ stroke: gridColor }}
          tickLine={false}
          width={50}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="cosine_distance"
          stroke="#c8a97e"
          strokeWidth={2}
          dot={{ r: 3, fill: "#c8a97e", stroke: dotStrokeColor, strokeWidth: 2 }}
          activeDot={{ r: 5, fill: "#c8a97e", stroke: dotStrokeColor, strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
