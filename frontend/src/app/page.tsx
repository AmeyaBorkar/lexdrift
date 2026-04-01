"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Search,
  TrendingUp,
  FileText,
  Bell,
  Building2,
} from "lucide-react";
import {
  getAlerts,
  getScreener,
  searchCompanies,
  type Alert,
  type ScreenerEntry,
  type Company,
} from "@/lib/api";

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
// Dashboard page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const router = useRouter();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [screener, setScreener] = useState<ScreenerEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Search state
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Company[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [searchTimeout, setSearchTimeout] = useState<ReturnType<typeof setTimeout> | null>(null);

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
          setScreener(screenerData.value);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  function handleSearch(value: string) {
    setQuery(value);
    if (searchTimeout) clearTimeout(searchTimeout);
    if (!value.trim()) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }
    const timeout = setTimeout(async () => {
      try {
        const data = await searchCompanies(value.trim());
        setSearchResults(data);
        setShowResults(true);
      } catch {
        setSearchResults([]);
      }
    }, 250);
    setSearchTimeout(timeout);
  }

  function selectCompany(ticker: string) {
    setShowResults(false);
    setQuery("");
    router.push(`/company/${ticker}`);
  }

  const unreadCount = alerts.filter((a) => !a.read).length;
  const companiesTracked = screener.length > 0 ? new Set(screener.map((s) => s.ticker)).size : 0;
  const topMovers = [...screener]
    .sort((a, b) => b.cosine_distance - a.cosine_distance)
    .slice(0, 5);
  const recentAlerts = alerts.slice(0, 3);
  const hasData = screener.length > 0 || alerts.length > 0;

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      {/* Page heading */}
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Track language drift across SEC filings.
        </p>
      </div>

      {/* Search bar - prominent */}
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            onFocus={() => searchResults.length > 0 && setShowResults(true)}
            placeholder="Enter a ticker to get started (e.g. AAPL, TSLA, MSFT)"
            className="h-12 w-full rounded-lg border border-border bg-card pl-12 pr-4 text-foreground text-base placeholder:text-muted-foreground outline-none focus:border-accent/50 transition-colors"
          />
        </div>

        {/* Search results dropdown */}
        {showResults && searchResults.length > 0 && (
          <div className="absolute top-full left-0 mt-1 w-full rounded-lg border border-border bg-card shadow-sm z-50">
            {searchResults.map((company) => (
              <button
                key={company.ticker}
                onClick={() => selectCompany(company.ticker)}
                className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm transition-colors hover:bg-muted/50 first:rounded-t-lg last:rounded-b-lg"
              >
                <span className="font-mono font-medium text-accent">
                  {company.ticker}
                </span>
                <span className="truncate text-muted-foreground">
                  {company.name}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Close dropdown on outside click */}
        {showResults && (
          <div
            className="fixed inset-0 z-40"
            onClick={() => setShowResults(false)}
          />
        )}
      </div>

      {/* Empty state */}
      {!loading && !hasData && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Building2 className="h-10 w-10 text-muted-foreground/40 mb-4" />
            <p className="text-foreground font-medium mb-1">No companies analyzed yet</p>
            <p className="text-sm text-muted-foreground">
              Search for a ticker above to begin tracking language drift.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Loading state */}
      {loading && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            <Skeleton className="h-64 lg:col-span-2" />
            <Skeleton className="h-64" />
          </div>
        </div>
      )}

      {/* Data view */}
      {!loading && hasData && (
        <>
          {/* Quick Stats row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardContent className="flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10">
                  <Building2 className="h-5 w-5 text-accent" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Companies Tracked</p>
                  <p className="text-xl font-semibold text-foreground">{companiesTracked}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10">
                  <FileText className="h-5 w-5 text-accent" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Filings Analyzed</p>
                  <p className="text-xl font-semibold text-foreground">{screener.length}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-danger/10">
                  <Bell className="h-5 w-5 text-danger" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Unread Alerts</p>
                  <p className="text-xl font-semibold text-foreground">{unreadCount}</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Two-column: Top Movers + Recent Alerts */}
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Top Movers */}
            <Card className="lg:col-span-2">
              <CardHeader className="flex flex-row items-center gap-2">
                <TrendingUp className="h-4 w-4 text-accent" />
                <CardTitle>Top Movers by Drift</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                {topMovers.length === 0 ? (
                  <p className="px-5 pb-5 text-sm text-muted-foreground">
                    No drift data available. Ingest some filings to get started.
                  </p>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-5 py-2.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Ticker</th>
                        <th className="px-5 py-2.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Company</th>
                        <th className="px-5 py-2.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Drift</th>
                        <th className="px-5 py-2.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Filed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topMovers.map((row) => {
                        const score = row.cosine_distance;
                        const severity: "critical" | "high" | "medium" | "low" =
                          score >= 0.5 ? "critical" : score >= 0.3 ? "high" : score >= 0.15 ? "medium" : "low";
                        return (
                          <tr
                            key={`${row.ticker}-${row.filing_date}`}
                            className="border-b border-border cursor-pointer hover:bg-muted/50 transition-colors"
                            onClick={() => router.push(`/company/${row.ticker}`)}
                          >
                            <td className="px-5 py-3 font-mono font-medium text-accent">{row.ticker}</td>
                            <td className="px-5 py-3 text-foreground">{row.name}</td>
                            <td className="px-5 py-3">
                              <Badge severity={severity}>
                                {(score * 100).toFixed(1)}%
                              </Badge>
                            </td>
                            <td className="px-5 py-3 text-muted-foreground">{row.filing_date}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </CardContent>
            </Card>

            {/* Recent Alerts */}
            <Card>
              <CardHeader className="flex flex-row items-center gap-2">
                <Bell className="h-4 w-4 text-muted-foreground" />
                <CardTitle>Recent Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                {recentAlerts.length === 0 ? (
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
          </div>
        </>
      )}
    </div>
  );
}
