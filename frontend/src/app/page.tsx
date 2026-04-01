"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { DataTable, type Column } from "@/components/ui/data-table";
import {
  getAlerts,
  getScreener,
  type Alert,
  type ScreenerEntry,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({
  title,
  value,
  loading,
}: {
  title: string;
  value: string | number;
  loading: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-24" />
        ) : (
          <p className="text-2xl font-semibold text-foreground">{value}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Severity helper
// ---------------------------------------------------------------------------

function severityOf(s: string): "critical" | "high" | "medium" | "low" {
  const l = s.toLowerCase();
  if (l === "critical") return "critical";
  if (l === "high") return "high";
  if (l === "medium") return "medium";
  return "low";
}

// ---------------------------------------------------------------------------
// Top drifters columns
// ---------------------------------------------------------------------------

const driftColumns: Column<ScreenerEntry & Record<string, unknown>>[] = [
  {
    key: "ticker",
    header: "Ticker",
    sortable: true,
    render: (row) => (
      <span className="font-medium text-foreground">{row.ticker}</span>
    ),
  },
  {
    key: "name",
    header: "Company",
    sortable: true,
  },
  {
    key: "cosine_distance",
    header: "Drift",
    sortable: true,
    render: (row) => {
      const score = row.cosine_distance as number;
      const severity: "critical" | "high" | "medium" | "low" =
        score >= 0.5 ? "critical" : score >= 0.3 ? "high" : score >= 0.15 ? "medium" : "low";
      return (
        <Badge severity={severity}>
          {(score * 100).toFixed(1)}%
        </Badge>
      );
    },
  },
  {
    key: "filing_date",
    header: "Filed",
    sortable: true,
    render: (row) => (
      <span className="text-muted-foreground">{row.filing_date}</span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Dashboard page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const router = useRouter();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [screener, setScreener] = useState<(ScreenerEntry & Record<string, unknown>)[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [alertsData, screenerData] = await Promise.allSettled([
          getAlerts(),
          getScreener("risk_factors", "cosine_distance", 10),
        ]);

        if (cancelled) return;

        if (alertsData.status === "fulfilled") setAlerts(alertsData.value);
        if (screenerData.status === "fulfilled")
          setScreener(
            screenerData.value.map((e) => ({ ...e } as ScreenerEntry & Record<string, unknown>))
          );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const recentAlerts = alerts.slice(0, 5);
  const unreadCount = alerts.filter((a) => !a.read).length;
  const avgDrift =
    screener.length > 0
      ? (
          screener.reduce((sum, s) => sum + (s.cosine_distance as number), 0) /
          screener.length *
          100
        ).toFixed(1) + "%"
      : "--";

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      {/* Page heading */}
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Overview of filing activity and language drift signals.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Companies"
          value={loading ? "" : screener.length > 0 ? new Set(screener.map((s) => s.ticker)).size : "--"}
          loading={loading}
        />
        <StatCard
          title="Total Filings"
          value={loading ? "" : "--"}
          loading={loading}
        />
        <StatCard
          title="Active Alerts"
          value={loading ? "" : unreadCount}
          loading={loading}
        />
        <StatCard
          title="Avg Drift"
          value={loading ? "" : avgDrift}
          loading={loading}
        />
      </div>

      {/* Two-column layout */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent alerts */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Recent Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : recentAlerts.length === 0 ? (
              <p className="text-sm text-muted-foreground">No alerts yet.</p>
            ) : (
              <ul className="space-y-3">
                {recentAlerts.map((alert) => (
                  <li
                    key={alert.id}
                    className="flex items-start gap-3 rounded border border-border p-3 transition-colors duration-200 hover:bg-muted/30 cursor-pointer"
                    onClick={() => router.push(`/company/${alert.ticker}`)}
                  >
                    <Badge severity={severityOf(alert.severity)} className="mt-0.5 shrink-0">
                      {alert.severity}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-foreground truncate">
                        {alert.ticker}
                      </p>
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {alert.message}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {/* Top drifting companies */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Top Drifting Companies</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {loading ? (
              <div className="space-y-2 p-5">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <DataTable
                columns={driftColumns}
                data={screener}
                keyExtractor={(row) => `${row.ticker}-${row.section_type}`}
                onRowClick={(row) => router.push(`/company/${row.ticker}`)}
                emptyMessage="No drift data available. Ingest some filings to get started."
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
